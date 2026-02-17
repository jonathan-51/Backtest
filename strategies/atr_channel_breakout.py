import pandas as pd
from typing import Dict,Tuple
from indicators import Indicators
from config import ATRChannelBreakoutConfig
from order import Order
from strategies.base import Strategy

class ATRChannelBreakout(Strategy):
    """Trend-following strategy that enters long when price breaks above an ATR envelope around the SMA.

    Optionally applies a SPY SMA regime filter: buy signals are suppressed when SPY is
    trading below its spy_sma_length moving average, keeping the strategy in cash during
    bear market regimes. Set spy_sma_length=0 to disable.
    """
    def __init__(self,
                 sma_length = ATRChannelBreakoutConfig.sma_length,
                 atr_length = ATRChannelBreakoutConfig.atr_length,
                 envelope_mult = ATRChannelBreakoutConfig.envelope_mult,
                 stop_mult = ATRChannelBreakoutConfig.stop_mult,
                 trail_mult = ATRChannelBreakoutConfig.trail_mult,
                 timeframe = ATRChannelBreakoutConfig.timeframe,
                 spy_sma_length = ATRChannelBreakoutConfig.spy_sma_length):
        self.sma_length = sma_length
        self.atr_length = atr_length
        self.envelope_mult = envelope_mult
        self.stop_mult = stop_mult
        self.trail_mult = trail_mult
        self.spy_sma_length = spy_sma_length
        self.indicator = Indicators()
        self.timeframe = timeframe

    def generate_signals(self,data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame,dict]:
        """Generate buy/sell signals based on ATR Channel Breakout"""

        df = data[self.timeframe].copy()

        # Compute Channel
        df['sma'] = self.indicator.sma(df['close'],self.sma_length)
        df['atr'] = self.indicator.atr(df,self.atr_length)
        df['upper'] = df['sma'] + self.envelope_mult * df['atr']
        df['lower'] = df['sma'] - self.envelope_mult * df['atr']

        # Previous close for crossover detection
        df['prev_close'] = df['close'].shift(1)

        df['signal'] = 'hold_cash'
        orders = {}

        # Vectorized buy condition: close crosses above upper envelope
        buy_mask = (
            df['prev_close'] <= df['upper']
        ) & (
            df['close'] > df['upper']
        ) & df['upper'].notna()

        df.loc[buy_mask, 'signal'] = 'buy'

        # Build orders only for buy rows (far fewer iterations than full loop)
        for idx, row in df[buy_mask].iterrows():
            orders[idx] = Order(
                stop_loss=max(row['lower'], row['close'] - self.stop_mult * row['atr']),
                trail_offset=self.trail_mult * row['atr'],
            )

        # Regime filter: suppress buy signals when SPY is below its SMA
        if self.spy_sma_length > 0 and 'spy_1d' in data:
            spy_df = data['spy_1d'].copy().set_index('date')
            spy_sma = spy_df['close'].rolling(self.spy_sma_length).mean()

            spy_sma_aligned = df['date'].map(spy_sma)
            spy_close_aligned = df['date'].map(spy_df['close'])
            regime_fail = spy_sma_aligned.isna() | (spy_close_aligned < spy_sma_aligned)

            suppress_mask = (df['signal'] == 'buy') & regime_fail
            df.loc[suppress_mask, 'signal'] = 'hold_cash'
            for idx in df.index[suppress_mask]:
                orders.pop(idx, None)

        return df[['date','signal','close','high','low']].copy(), orders
