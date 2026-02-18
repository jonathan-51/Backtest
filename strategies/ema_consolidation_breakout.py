import pandas as pd
import logging
from strategies.base import Strategy
from config import EMAConsolidationBreakoutConfig, RegimeFilterConfig
from indicators import Indicators
from regime_filter import RegimeFilter
from typing import Dict
from order import Order

logger = logging.getLogger(__name__)

class EMAConsolidationBreakout(Strategy):
    def __init__(self,
                 ema_fast_length = EMAConsolidationBreakoutConfig.ema_fast_length,
                 ema_slow_length = EMAConsolidationBreakoutConfig.ema_slow_length,
                 ema_trend_length = EMAConsolidationBreakoutConfig.ema_trend_length,
                 ema_filter_length = EMAConsolidationBreakoutConfig.ema_filter_length,
                 atr_length = EMAConsolidationBreakoutConfig.atr_length,
                 consolidation_bar_length = EMAConsolidationBreakoutConfig.consolidation_bar_length,
                 consolidation_mult = EMAConsolidationBreakoutConfig.consolidation_mult,
                 timeframe = EMAConsolidationBreakoutConfig.timeframe,
                 use_regime_filter = EMAConsolidationBreakoutConfig.use_regime_filter,
                 regime_score_threshold = EMAConsolidationBreakoutConfig.regime_score_threshold):
        self.ema_fast_length = ema_fast_length
        self.ema_slow_length = ema_slow_length
        self.ema_trend_length = ema_trend_length
        self.ema_filter_length = ema_filter_length
        self.atr_length = atr_length
        self.consolidation_bar_length = consolidation_bar_length
        self.consolidation_mult = consolidation_mult
        self.use_regime_filter = use_regime_filter
        self.regime_score_threshold = regime_score_threshold
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

        # Regime filter: suppress buy signals when market environment is unfavourable
        if self.use_regime_filter and 'regime_data' in data:
            regime_filter = RegimeFilter(
                sma_long=RegimeFilterConfig.sma_long,
                sma_short=RegimeFilterConfig.sma_short,
                sma_ratio=RegimeFilterConfig.sma_ratio,
                vix_threshold=RegimeFilterConfig.vix_threshold,
            )
            scores = regime_filter.compute(data['regime_data'])  # date-indexed Series

            scores_aligned = df['date'].map(scores).fillna(0).astype(int)
            logger.info(f"Regime score distribution: {scores_aligned.value_counts().sort_index().to_dict()}")
            logger.info(f"Days below threshold ({self.regime_score_threshold}): {(scores_aligned < self.regime_score_threshold).sum()} / {len(scores_aligned)}")
            suppress_mask = buy_mask & (scores_aligned < self.regime_score_threshold)

            df.loc[suppress_mask, 'signal'] = 'hold_cash'
            for idx in df.index[suppress_mask]:
                orders.pop(idx, None)

        return df[['date','signal','close','high','low']].copy(), orders
