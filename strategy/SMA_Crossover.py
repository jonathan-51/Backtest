import pandas as pd
from strategy.base import Strategy
from config import SMACrossoverConfig
from typing import Dict
from indicators import Indicators
from order import Order

class SMACrossover(Strategy):

    def __init__(self,
                 sma_slow_length = SMACrossoverConfig.slow_length,
                 sma_fast_length = SMACrossoverConfig.fast_length,
                 timeframe = SMACrossoverConfig.timeframe,
                 stop_loss_percent = SMACrossoverConfig.stop_loss_percent,
                 take_profit_percent = SMACrossoverConfig.take_profit_percent):
        self.sma_slow_length = sma_slow_length
        self.sma_fast_length = sma_fast_length
        self.timeframe = timeframe
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.indicator = Indicators()

    def generate_signals(self, data:Dict[str,pd.DataFrame],config) -> tuple[pd.DataFrame,dict]:

        sma_slow_length = config.slow_length
        sma_fast_length = config.fast_length
        stop_loss_percent = config.stop_loss_percent
        take_profit_percent = config.take_profit_percent
        timeframe = config.timeframe

        df = data[timeframe].copy()

        # Computing indicators
        df['sma_slow'] = self.indicator.sma(df['close'],sma_slow_length)
        df['sma_fast'] = self.indicator.sma(df['close'],sma_fast_length)

        # Initializing signals
        df['signal'] = 'hold_cash'
        orders = {}

        # Vectorized buy condition: fast crosses above slow
        buy_mask =  ((df['sma_fast'].shift(1) < df['sma_slow'].shift(1)) & (df['sma_fast'] > df['sma_slow']))
        # Vectorized sell condition: fast crosses below slow
        sell_mask = ((df['sma_fast'].shift(1) > df['sma_slow'].shift(1)) & (df['sma_fast'] < df['sma_slow']))

        df.loc[buy_mask,'signal'] = 'buy'
        df.loc[sell_mask,'signal'] = 'short'

        return df[['date','signal','close','high','low']].copy(), orders