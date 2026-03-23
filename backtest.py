import pandas as pd
from config import BacktestConfig
from strategy.base import Strategy
import logging
from typing import Dict

class BacktestEngine:
    """Simulates trading strategy execution against historical data"""
    def __init__(self,config:BacktestConfig,strategy:Strategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        self.capital = self.config.initial_capital
        self.position = {}
        self.trade_log = []
        self.equity_curve = []

    def run(self,data:Dict[str,pd.DataFrame]) -> dict:
        """Execute strategy signals against historical data and return results."""

        # Loop through all symbols in dictionary
        for symbol,timeframes in data.items():
            # Loop through all timeframes per symbol
            for timeframe, df in timeframes.items():
                # Loop through all rows from each timeframe per symbol
                for i, row in df.iterrows():
                    price = row['close']
                    signal = row['signal']
                    date = row['date']


                    # Enter long position
                    if (not self.position and signal == 'buy'):
                        self.position[symbol] = {
                            'entry_date':date,
                            'direction':'long'}
                        
                        self._execute_buy(symbol,price,date)
                    
                    # Enter short position
                    elif (not self.position and signal == 'short'):

                        self.position[symbol] = {
                            'entry_date':date,
                            'direction':'short'}
                        
                        self._execute_sell(symbol,price,date)       


    def _execute_buy(self,symbol:str,price:float,date:str):
        """Open a long position"""

        # Calculating position size
        position_size = self.capital * self.config.position_size
        # Calculating fill price of 1 share after accounting for slippage and spreads
        fill_price = self._apply_slippage(price,'buy')

        shares = int(position_size // fill_price)
        
        if shares <= 0:
            self.logger.warning(f"Insufficient budget to buy {symbol} at {price:.2f} on {date}")
            return
        
        # Calculating total market size position to the nearest whole share
        cost = shares * fill_price
        commission = self._calculate_commission(shares)

        if cost + commission > self.capital:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.capital:.2f}")
            return

        self.position[symbol] = {
            **self.position[symbol],
            'shares':shares,
            'entry_price':fill_price,
            'commission_in':commission,
        }

    def _execute_short(self,symbol:str,price:float,date:str):
        """Open a short position"""

        # Calculating position size
        position_size = self.capital * self.config.position_size
        # Calculating fill price of 1 share after accounting for slippage and spreads
        fill_price = self._apply_slippage(price,'sell')

        shares = int(position_size // fill_price)
        
        if shares <= 0:
            self.logger.warning(f"Insufficient budget to buy {symbol} at {price:.2f} on {date}")
            return
        
        # Calculating total market size position to the nearest whole share
        cost = shares * fill_price
        commission = self._calculate_commission(shares)

        if cost + commission > self.capital:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.capital:.2f}")
            return

        self.position[symbol] = {
            **self.position[symbol],
            'shares':shares,
            'entry_price':fill_price,
            'commission_in':commission,
        }
    
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
        
    def _calculate_commission(self,shares:int) -> float:

        commission = self.config.commission_rate * shares

        if commission < 1:
            return 1.00
        elif commission > 9.79:
            return 9.79
        else:
            return commission
        