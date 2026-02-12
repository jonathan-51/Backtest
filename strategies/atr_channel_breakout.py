import pandas as pd
from typing import Dict
from indicators import Indicators
from config import ATRChannelBreakoutConfig

class ATRChannelBreakout:

    def __init__(self, 
                 sma_length = ATRChannelBreakoutConfig.sma_length, 
                 atr_length = ATRChannelBreakoutConfig.atr_length, 
                 envelope_mult = ATRChannelBreakoutConfig.envelope_mult,
                 stop_mult = ATRChannelBreakoutConfig.stop_mult,
                 trail_mult = ATRChannelBreakoutConfig.trail_mult ):
        
        self.sma_length = sma_length
        self.atr_length = atr_length
        self.envelope_mult = envelope_mult
        self.stop_mult = stop_mult
        self.trail_mult = trail_mult
        self.indicator = Indicators()

    def generate_signals(self,data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate buy/sell signals based on ATR Channel Breakout"""

        df = data['1d'].copy()

        # Compute Channel
        df['sma'] = self.indicator.sma(df['close'],self.sma_length)
        df['atr'] = self.indicator.atr(df,self.atr_length)
        df['upper'] = df['sma'] + self.envelope_mult * df['atr']
        df['lower'] = df['sma'] - self.envelope_mult * df['atr']

        # Previous close for crossover detection
        df['prev_close'] = df['close'].shift(1)

        df['signal'] = 'hold_cash'
        in_position = False

        for i in range(1,len(df)):
            if pd.isna(df['upper'].iloc[i]):
                continue

            prev_close = df['prev_close'].iloc[i]
            curr_close = df['close'].iloc[i]
            upper = df['upper'].iloc[i]

            # Entry: close crosses above upper envelope
            if not in_position and prev_close <= upper and curr_close > upper:
                in_position = True
                df.loc[df.index[i],'signal'] = 'buy'

            elif in_position:
                df.loc[df.index[i],'signal'] = 'hold_long'

        # Stop columns for engine
        df['initial_stop'] = df['lower']
        df['stop_offset'] = self.stop_mult * df['atr']
        df['trail_offset'] = self.trail_mult * df['atr']

        return df[['date','signal','close','high','low','inital_stop','stop_offset','trail_offset']].copy()