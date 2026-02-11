import pandas as pd
import logging
from typing import Dict,List
from config import BacktestConfig
from strategies.base import Strategy

class BacktestEngine:
    """Simulates trading strategy execution against historical data."""
    def __init__(self,config:BacktestConfig,strategy:Strategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        self.cash = self.config.initial_capital
        self.position = None
        self.trade_log = []
        self.equity_curve = []
        

    def run(self,data:Dict[str,pd.DataFrame]) -> dict:
        """Execute strategy signals against historical data and return results."""

        # Generate signals
        signals = self.strategy.generate_signals(data)

        # Loop through each bar
        for _, row in signals.iterrows():
            date = row['date']
            price = row['close']
            signal = row['signal']

            # Execute signals
            if signal == 'buy' and self.position is None:
                self._execute_buy(date,price)
            elif signal == 'sell' and self.position is not None:
                self._execute_sell(date,price)

            # Record equity
            equity = self.cash
            if self.position is not None:
                equity += self.position['shares'] * price
            self.equity_curve.append({'date':date,'equity':equity})

        # Force close position at end of data
        if self.position is not None:
            last = signals.iloc[-1]
            self._execute_sell(last['date'],last['close'])
        
        return self._build_results()

    def _build_results(self) -> dict:
        """Package trade log"""

        equity_df = pd.DataFrame(self.equity_curve)
        wins = [trade for trade in self.trade_log if trade['pnl'] > 0]
        losses = [trade for trade in self.trade_log if trade['pnl'] < 0]
        total_pnl = sum(trade['pnl'] for trade in self.trade_log)

        # Max Drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'trade_log':self.trade_log,
            'equity_curve': equity_df,
            'summary': {
                'total_trades': len(self.trade_log),
                'winning_trades':len(wins),
                'losing_trades':len(losses),
                'win_rate':len(wins) / len(self.trade_log) if self.trade_log else 0,
                'total_pnl': total_pnl,
                'total_return_pct': total_pnl / self.config.initial_capital,
                'max_drawdown': max_drawdown,
                'final_equity': equity_df['equity'].iloc[-1],
            }
        }

    def _execute_buy(self,date:str,price:float):
        """Open a long position at the given price with costs applied"""

        # Calculate number of shares
        shares = self._calculate_shares(price)
        if shares <= 0:
            return
        
        fill_price = self._apply_slippage(price,'buy')
        cost = shares * fill_price
        commission = self._calculate_commission(cost)

        self.cash -= (cost + commission)
        self.position = {
            'shares': shares,
            'entry_price':fill_price,
            'entry_date':date,
            'commission_in':commission
            }

    def _execute_sell(self,date:str,price:float):
        """Close the open position at the given price with costs applied."""
        fill_price = self._apply_slippage(price,'sell')
        proceeds = self.position['shares'] * fill_price
        commission = self._calculate_commission(proceeds)

        self.cash += (proceeds - commission)

        # Calculate P&L
        entry_cost = self.position['shares'] * self.position['entry_price']
        total_commision = self.position['commission_in'] + commission
        pnl = proceeds - entry_cost - total_commision

        # Append to trade log
        self.trade_log.append({
            'entry_date':self.position['entry_date'],
            'exit_date':date,
            'entry_price':self.position['entry_price'],
            'exit_price': fill_price,
            'shares': self.position['shares'],
            'pnl':pnl,
            'return_pct':pnl / entry_cost,
            'commission_paid': total_commision,
        })

        self.position = None

    def _apply_slippage(self,stock_price:float,direction:str) -> float:
        if direction == 'buy':
            # Calculate share price after slippage
            price_after_slippage = stock_price + (stock_price * self.config.slippage)

            # Returns share price after slippage and spreads
            return self._apply_spreads(price_after_slippage,direction)
        else:
            # Calculate share price after slippage
            price_after_slippage = stock_price - (stock_price * self.config.slippage)

            # Returns share price after slippage and spreads
            return self._apply_spreads(price_after_slippage,direction)

    def _apply_spreads(self,price:float,direction:str) -> float:
        if direction == 'buy':
            return price + self.config.spreads
        else:
            return price - self.config.spreads
        

    def _calculate_commission(self,trade_value: float) -> float:
        return trade_value * self.config.commission_rate

    def _calculate_shares(self,price:float) -> int:
        """Calculate number of whole shares to buy based on position sizing"""

        # Calculate position size
        position_size = self.cash * self.config.position_size

        # Calculate filled price
        fill_price = self._apply_slippage(price,'buy')

        # Calculate number of shares
        shares = int(position_size // fill_price)

        return shares
        