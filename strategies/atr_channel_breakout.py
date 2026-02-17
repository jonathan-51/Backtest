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

        for i in range(1,len(df)):
            if pd.isna(df['upper'].iloc[i]):
                continue

            prev_close = df['prev_close'].iloc[i]
            curr_close = df['close'].iloc[i]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]
            curr_atr = df['atr'].iloc[i]

            # Entry: close crosses above upper envelope
            if prev_close <= upper and curr_close > upper:
                df.loc[df.index[i],'signal'] = 'buy'
                orders[df.index[i]] = Order(
                    stop_loss = max(lower, curr_close - self.stop_mult * curr_atr),
                    trail_offset=self.trail_mult * curr_atr,
                )

        # Regime filter: suppress buy signals when SPY is below its SMA
        if self.spy_sma_length > 0 and 'spy_1d' in data:
            spy_df = data['spy_1d'].copy().set_index('date')
            spy_sma = spy_df['close'].rolling(self.spy_sma_length).mean()

            for i in range(len(df)):
                if df['signal'].iloc[i] != 'buy':
                    continue
                bar_date = df['date'].iloc[i]
                if bar_date not in spy_sma.index:
                    continue
                sma_val = spy_sma.loc[bar_date]
                spy_close = spy_df.loc[bar_date, 'close']
                if pd.isna(sma_val) or spy_close < sma_val:
                    df.loc[df.index[i], 'signal'] = 'hold_cash'
                    orders.pop(df.index[i], None)

        return df[['date','signal','close','high','low']].copy(), orders
