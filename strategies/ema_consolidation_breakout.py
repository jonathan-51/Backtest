import pandas as pd
from strategies.base import Strategy
from config import EMAConsolidationBreakoutConfig
from indicators import Indicators
from typing import Dict
from order import Order

class EMAConsolidationBreakout(Strategy):
    def __init__(self,
                 ema_fast_length = EMAConsolidationBreakoutConfig.ema_fast_length,
                 ema_slow_length = EMAConsolidationBreakoutConfig.ema_slow_length,
                 ema_trend_length = EMAConsolidationBreakoutConfig.ema_trend_length,
                 ema_filter_length = EMAConsolidationBreakoutConfig.ema_filter_length,
                 atr_length = EMAConsolidationBreakoutConfig.atr_length,
                 consolidation_bar_length = EMAConsolidationBreakoutConfig.consolidation_bar_length,
                 consolidation_mult = EMAConsolidationBreakoutConfig.consolidation_mult,
                 timeframe = EMAConsolidationBreakoutConfig.timeframe):
        self.ema_fast_length = ema_fast_length
        self.ema_slow_length = ema_slow_length
        self.ema_trend_length = ema_trend_length
        self.ema_filter_length = ema_filter_length
        self.atr_length = atr_length
        self.consolidation_bar_length = consolidation_bar_length
        self.consolidation_mult = consolidation_mult
        self.indicator = Indicators()
        self.timeframe = timeframe

    def generate_signals(self,data: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame,dict]:
        df = data[self.timeframe].copy()

        # Compute indicators
        df['ema_fast'] = self.indicator.ema(df['close'],self.ema_fast_length)
        df['ema_slow'] = self.indicator.ema(df['close'],self.ema_slow_length)
        df['ema_trend'] = self.indicator.ema(df['close'],self.ema_trend_length)
        df['ema_filter'] = self.indicator.ema(df['close'],self.ema_filter_length)

        df['atr'] = self.indicator.atr(df,self.atr_length)

        df['consolidation_high'] = df['high'].shift(1).rolling(self.consolidation_bar_length).max()
        df['consolidation_low'] = df['low'].shift(1).rolling(self.consolidation_bar_length).min()
        
        df['up_trending'] = df['ema_filter'] > df['ema_filter'].shift(60)

        df['signal'] = 'hold_cash'
        orders = {}

        # Vectorized buy condition
        buy_mask = (
                df['close'] > df['ema_fast']
            ) & (
                df['close'] > df['ema_slow']
            ) & (
                df['close'] > df['ema_trend']
            ) & (
                df['close'] > df['ema_filter']
            ) & (
                df['close'] > df['consolidation_high']
            ) & (
                (df['consolidation_high'] - df['consolidation_low']) < self.consolidation_mult * df['atr']
            ) & (
                df['up_trending']
            )

        df.loc[buy_mask, 'signal'] = 'buy'

        # Build orders only for buy rows (far fewer iterations than full loop)
        for idx, row in df[buy_mask].iterrows():
            orders[idx] = Order(
                stop_loss=row['ema_slow'] - 0.5 * row['atr'],
                trail_offset=2.0 * row['atr'],
            )

        return df[['date','signal','close','high','low']].copy(), orders