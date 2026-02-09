import pandas as pd
from typing import Dict

class MACrossoverStrategy:
    """Simple Moving Average Crossover Strategy (Stub for Testing)"""
    
    def __init__(self,fast_period: int = 5,slow_period: int =20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data: Dict[str,pd.DataFrame]) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover"""

        df = data['1d'].copy()

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        # Initialize signal column
        df['signal'] = 'hold_cash'

        # Track position status
        in_position = False

        # Generate signals based on crossovers
        for i in range(1,len(df)):
            prev_fast = df['fast_ma'].iloc[i-1]
            prev_slow = df['slow_ma'].iloc[i-1]
            curr_fast = df['fast_ma'].iloc[i]
            curr_slow = df['slow_ma'].iloc[i]

            # Skip if we don't have enough data yet
            if pd.isna(prev_fast) or pd.isna(prev_slow):
                continue
            
            # Buy signal: slow MA crosses above slow MA
            if not in_position and prev_fast <= prev_slow and curr_fast > curr_slow:
                df.loc[df.index[i],'signal'] = 'buy'
                in_position = True

            # Sell signal: fast MA crosses below slow MA
            elif in_position and prev_fast >= prev_slow and curr_fast < curr_slow:
                df.loc[df.index[i],'signal'] = 'sell'
                in_position = False

            # Hold states based on current position
            elif in_position:
                df.loc[df.index[i],'signal'] = 'hold_long'
            else:
                df.loc[df.index[i],'signal'] = 'hold_cash'

        # Return only date, signal and close columns
        return df[['date','signal','close']].copy()