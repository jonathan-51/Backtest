import pandas as pd
from config import BacktestConfig
from strategy.base import Strategy
import logging
from typing import Dict
from order import Position

class BacktestEngine:
    """Simulates trading strategy execution against historical data"""
    def __init__(self,config:BacktestConfig,strategy:Strategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        self.capital = self.config.initial_capital
        self.position = {}
        self.equity_curve = []
        self.trade_log = []

    def run(self,data:Dict[str,pd.DataFrame],strategy_config) -> dict:
        """Execute strategy signals against historical data and return results."""
        self.strategy_config = strategy_config

        # Loop through all symbols in dictionary
        for symbol,timeframes in data.items():
            
            df, orders = self.strategy.generate_signals(timeframes,self.strategy_config)
            # Loop through all rows from each timeframe per symbol
            for i, row in df.iterrows():
                price = row['close']
                high = row['high']
                low = row['low']
                signal = row['signal']
                date = row['date']

                # Check for stops
                if self.position:
                    # Check for stop-loss / take-profit
                    self._check_exits(price,high,low,symbol,date)


                # Enter long position
                if (not self.position and signal == 'buy'):
                    direction = 'long'
                    
                    
                    shares,fill_price,commission = self._enter_position(symbol,price,date,'buy')

                    if shares is None:
                        continue

                    if i in orders:
                        stop_loss=orders[i].stop_loss
                        take_profit=orders[i].take_profit

                    else:
                        stop_loss,take_profit = self._add_exits(shares,direction,fill_price)
                    
                    self.position = Position(
                        symbol=symbol,
                        direction=direction,
                        entry_date=date,
                        entry_price=fill_price,
                        shares=shares,
                        commission_in=commission,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                
                # Enter short position
                elif (not self.position and signal == 'short'):
                    direction = 'short'
                    
                    shares,fill_price,commission = self._enter_position(symbol,price,date,'short')

                    if shares is None:
                        continue

                    if i in orders:
                        stop_loss=orders[i].stop_loss
                        take_profit=orders[i].take_profit

                    else:
                        stop_loss,take_profit = self._add_exits(shares,direction,fill_price)
                    
                    self.position = Position(
                        symbol=symbol,
                        direction=direction,
                        entry_date=date,
                        entry_price=fill_price,
                        shares=shares,
                        commission_in=commission,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                self._log_equity(price,date)
        
        # Build results
        results = self._build_results()

        return results

    def _log_equity(self,price,date):
        
        if self.position:
            multiplier = -1 if self.position.direction == 'short' else 1

            entry_position_size = self.position.shares * self.position.entry_price
            unrealized_pnl = multiplier * self.position.shares * (price - self.position.entry_price)
            market_value = entry_position_size + unrealized_pnl
            
            self.equity_curve.append({'date':date,'equity':self.capital + market_value})
        else:
            self.equity_curve.append({'date':date,'equity':self.capital})
    
    def _check_exits(self,price,high,low,symbol,date):
        """Check if current price triggers stop-loss or take-profit.
        If hit, calculate PnL, log the trade, update capital, and clear position."""
        if not self.position:
            return
        hit = False

        # Check if price breached SL or TP
        if self.position.direction == 'long':
            if high >= self.position.take_profit or low <= self.position.stop_loss:
                hit = True
        else:
            if low <= self.position.take_profit or high >= self.position.stop_loss:
                hit = True

        if hit:
            # Calculate PnL: positive for profitable trades, negative for losses
            multiplier = 1 if self.position.direction == 'long' else -1
            pnl = multiplier * (price - self.position.entry_price) * self.position.shares

            # Log completed trade
            self.trade_log.append({
                    'symbol':symbol,
                    'direction':self.position.direction,
                    'entry_date':self.position.entry_date,
                    'exit_date':date,
                    'entry_price':self.position.entry_price,
                    'exit_price':price,
                    'shares': self.position.shares,
                    'pnl':pnl,
                    'commission':self.position.commission_in*2,
                })

            # Update capital and clear position
            self.capital += pnl - self.position.commission_in + self.position.entry_price * self.position.shares
            self.position = {}
        return

    def _add_exits(self,shares,direction,fill_price):
        """Calculate capital-based SL/TP when the strategy doesn't provide per-trade exits.
        Risk is a percentage of current capital."""

        # How much capital we're willing to lose on this trade
        risk_val = self.capital * self.strategy_config.stop_loss_percent
        
        # Risk value per share
        risk_per_share = risk_val / shares

        if direction == 'long':
            # Long: SL below entry, TP above entry (2:1 RRR)
            stop_loss = fill_price - risk_per_share
            take_profit = fill_price + 2 * risk_per_share
        else:
            # Short: SL above entry, TP below entry (2:1 RRR)
            stop_loss = fill_price + risk_per_share
            take_profit = fill_price - 2 * risk_per_share

        return stop_loss,take_profit

    def _enter_position(self,symbol:str,price:float,date:str,signal:str):
        """Open a long position.
        Calculates fill price (with slippage/spreads), number of shares,
        commission, and updates the position dict."""

        # Budget available for this position
        position_size = self.capital * self.config.position_size
        # Simulate real fill price: market price + slippage + spread
        fill_price = self._apply_slippage(price,signal)
        # Maximum whole shares we can afford
        shares = int(position_size // fill_price)

        if shares <= 0:
            self.logger.warning(f"Insufficient budget to buy {symbol} at {price:.2f} on {date}")
            return None, None, None

        # Total cost and commission for this trade
        cost = shares * fill_price
        commission = self._calculate_commission(shares)


        if cost + commission > self.capital:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.capital:.2f}")
            return None, None, None
        
        self.capital -= (cost + commission)

        return shares,fill_price,commission
    
    def _apply_slippage(self,stock_price:float,direction:str) -> float:
        """Simulate slippage: buying pushes price up, selling pushes price down."""
        if direction == 'buy':
            price_after_slippage = stock_price + (stock_price * self.config.slippage)
            return self._apply_spreads(price_after_slippage,direction)
        else:
            price_after_slippage = stock_price - (stock_price * self.config.slippage)
            return self._apply_spreads(price_after_slippage,direction)

    def _apply_spreads(self,price:float,direction:str) -> float:
        """Add bid-ask spread cost: buyers pay more, sellers receive less."""
        if direction == 'buy':
            return price + self.config.spreads
        else:
            return price - self.config.spreads

    def _calculate_commission(self,shares:int) -> float:
        """Calculate IBKR-style commission: per-share rate, min $1.00, max $9.79."""
        commission = self.config.commission_rate * shares

        if commission < 1:
            return 1.00
        elif commission > 9.79:
            return 9.79
        else:
            return commission
        
    def _build_results(self) -> dict:
        """Package trade log"""

        if not self.trade_log:
            return {
                'trade_log': [],
                'equity_curve': [],
                'summary': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'return_pct': 0,
                    'final_equity': self.config.initial_capital
                }
            }

        # Convert list of dicts to Panda's Dataframe, each unique key is a column header and each entry is a row
        df = pd.DataFrame(self.trade_log)

        return {
            'trade_log':self.trade_log,
            'equity_curve':pd.DataFrame(self.equity_curve),
            'summary': {
                'total_trades':len(self.trade_log),
                'winning_trades': (df['pnl'] > 0).sum(),
                'losing_trades': (df['pnl'] < 0).sum(),
                'win_rate': (df['pnl'] > 0).sum() / len(self.trade_log) if self.trade_log else 0,
                'total_pnl':df['pnl'].sum(),
                'return_pct':df['pnl'].sum() / self.config.initial_capital,
                'final_equity': self.equity_curve[-1]['equity']
            }
        }