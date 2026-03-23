import pandas as pd
from config import BacktestConfig
from strategy.base import Strategy
import logging
from typing import Dict

class BacktestEngine:
    """Simulates trading strategy execution against historical data"""
    def __init__(self,config:BacktestConfig,strategy:Strategy):
        self.config = config
        self.strategy = strategy()
        self.logger = logging.getLogger(__name__)

        self.capital = self.config.initial_capital
        self.position = {}
        self.trade_log = []
        self.equity_curve = []

    def run(self,data:Dict[str,pd.DataFrame]) -> dict:
        """Execute strategy signals against historical data and return results."""

        # Loop through all symbols in dictionary
        for symbol,timeframes in data.items():

            df, orders = self.strategy.generate_signals(timeframes)

            # Loop through all rows from each timeframe per symbol
            for i, row in df.iterrows():
                price = row['close']
                signal = row['signal']
                date = row['date']

                # Check for stops
                if self.position:
                    # Check for stop-loss / take-profit
                    self._check_exits(price,symbol,date)

                # Enter long position
                if (not self.position and signal == 'buy'):
                    self.position = {
                        'entry_date':date,
                        'direction':'long'}
                    
                    
                    self._execute_buy(symbol,price,date)

                    if i in orders:
                        self.position = {
                            **self.position,
                            'stop_loss':orders[i].stop_loss,
                            'take_profit':orders[i].take_profit
                        }
                    else:
                        self._add_exits()
                
                # Enter short position
                elif (not self.position and signal == 'short'):

                    self.position = {
                        'entry_date':date,
                        'direction':'short'}
                    
                    self._execute_short(symbol,price,date)

                    if i in orders:
                        self.position = {
                            **self.position,
                            'stop_loss':orders[i].stop_loss,
                            'take_profit':orders[i].take_profit
                        }
                    else:
                        self._add_exits()

    def _check_exits(self,price,symbol,date):
        """Check if current price triggers stop-loss or take-profit.
        If hit, calculate PnL, log the trade, update capital, and clear position."""
        if not self.position:
            return
        hit = False

        # Check if price breached SL or TP
        if self.position['direction'] == 'long':
            if price >= self.position['take_profit'] or price <= self.position['stop_loss']:
                hit = True
        else:
            if price <= self.position['take_profit'] or price >= self.position['stop_loss']:
                hit = True

        if hit:
            # Calculate PnL: positive for profitable trades, negative for losses
            multiplier = 1 if self.position['direction'] == 'long' else -1
            pnl = multiplier * (price - self.position['entry_price']) * self.position['shares']

            # Log completed trade
            self.trade_log.append({
                    'symbol':symbol,
                    'direction':self.position['direction'],
                    'entry_date':self.position['entry_date'],
                    'exit_date':date,
                    'entry_price':self.position['entry_price'],
                    'exit_price':price,
                    'shares': self.position['shares'],
                    'pnl':pnl,
                    'commission':self.position['commission_in']*2,
                })

            # Update capital and clear position
            self.capital += pnl
            self.position = {}
        return

    def _add_exits(self):
        """Calculate capital-based SL/TP when the strategy doesn't provide per-trade exits.
        Risk is a percentage of current capital."""

        # How much capital we're willing to lose on this trade
        risk_val = self.capital * self.strategy.stop_loss_percent

        # Convert capital risk into a price movement percentage
        position_size_risk_percent = (self.position['shares'] * self.position['entry_price']) / risk_val

        if self.position['direction'] == 'long':
            stop_loss = self.position['entry_price'] - (self.position['entry_price'] * position_size_risk_percent)
            take_profit = self.position['entry_price'] + (2 * self.position['entry_price'] * position_size_risk_percent)
        else:
            stop_loss = self.position['entry_price'] + (self.position['entry_price'] * position_size_risk_percent)
            take_profit = self.position['entry_price'] - (2 * self.position['entry_price'] * position_size_risk_percent)

        self.position = {
            **self.position,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        return

    def _execute_buy(self,symbol:str,price:float,date:str):
        """Open a long position.
        Calculates fill price (with slippage/spreads), number of shares,
        commission, and updates the position dict."""

        # Budget available for this position
        position_size = self.capital * self.config.position_size
        # Simulate real fill price: market price + slippage + spread
        fill_price = self._apply_slippage(price,'buy')
        # Maximum whole shares we can afford
        shares = int(position_size // fill_price)

        if shares <= 0:
            self.logger.warning(f"Insufficient budget to buy {symbol} at {price:.2f} on {date}")
            return

        # Total cost and commission for this trade
        cost = shares * fill_price
        commission = self._calculate_commission(shares)

        if cost + commission > self.capital:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.capital:.2f}")
            return

        self.position = {
            **self.position,
            'shares':shares,
            'entry_price':fill_price,
            'commission_in':commission,
        }

    def _execute_short(self,symbol:str,price:float,date:str):
        """Open a short position.
        Calculates fill price (with slippage/spreads), number of shares,
        commission, and updates the position dict."""

        # Budget available for this position
        position_size = self.capital * self.config.position_size
        # Simulate real fill price: market price - slippage - spread
        fill_price = self._apply_slippage(price,'sell')
        # Maximum whole shares we can afford
        shares = int(position_size // fill_price)

        if shares <= 0:
            self.logger.warning(f"Insufficient budget to short {symbol} at {price:.2f} on {date}")
            return

        # Total cost and commission for this trade
        cost = shares * fill_price
        commission = self._calculate_commission(shares)

        if cost + commission > self.capital:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.capital:.2f}")
            return

        self.position = {
            **self.position,
            'shares':shares,
            'entry_price':fill_price,
            'commission_in':commission,
        }
    
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
        