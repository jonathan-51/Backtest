import pandas as pd
from typing import Dict,Tuple
from config import RSIPullbackUptrendConfig
from indicators import Indicators
from order import Order

class RSIPullbackUptrend:
    """Connors-style RSI(2) pullback in uptrend strategy."""
    def __init__(self,
                 entry_sma_length = RSIPullbackUptrendConfig.entry_sma_length,
                 exit_sma_length = RSIPullbackUptrendConfig.exit_sma_length,
                 atr_length = RSIPullbackUptrendConfig.atr_length,
                 rsi_length = RSIPullbackUptrendConfig.rsi_length,
                 rsi_entry_threshold = RSIPullbackUptrendConfig.rsi_entry_threshold,
                 rsi_exit_threshold = RSIPullbackUptrendConfig.rsi_exit_threshold,
                 stop_mult = RSIPullbackUptrendConfig.stop_mult):
        self.entry_sma_length = entry_sma_length
        self.exit_sma_length = exit_sma_length
        self.atr_length = atr_length
        self.rsi_length = rsi_length
        self.rsi_entry_threshold = rsi_entry_threshold
        self.rsi_exit_threshold = rsi_exit_threshold
        self.stop_mult = stop_mult
        self.indicator = Indicators()

    def generate_signals(self,data: Dict[str,pd.DataFrame]) -> Tuple[pd.DataFrame,dict]:
        """Generate buy/sell signals based on RSI uptrend pullbacks"""

        df = data['1d'].copy()

        # Compute Indicators
        df['entry_sma'] = self.indicator.sma(df['close'],self.entry_sma_length)
        df['exit_sma'] = self.indicator.sma(df['close'],self.exit_sma_length)
        df['atr'] = self.indicator.atr(df,self.atr_length)
        df['rsi'] = self.indicator.rsi(df['close'],self.rsi_length)

        df['signal'] = 'hold_cash'
        orders = {}

        for i in range(1,len(df)):
            if pd.isna(df['entry_sma'].iloc[i]):
                continue

            curr_close = df['close'].iloc[i]
            curr_atr = df['atr'].iloc[i]
            curr_entry_sma = df['entry_sma'].iloc[i]
            curr_exit_sma = df['exit_sma'].iloc[i]
            curr_rsi = df['rsi'].iloc[i]

            # Entry: uptrend confirmed (close > long SMA) + short-term oversold (low RSI)
            if curr_close > curr_entry_sma and curr_rsi <= self.rsi_entry_threshold:
                df.loc[df.index[i],'signal'] = 'buy'
                orders[df.index[i]] = Order(stop_loss=curr_close - (self.stop_mult * curr_atr))

            # Exit: price reverted to short-term mean (close > short SMA) or momentum recovered (RSI high)
            elif curr_close > curr_exit_sma or curr_rsi >= self.rsi_exit_threshold:
                df.loc[df.index[i],'signal'] = 'sell'

        return df[['date','signal','close','high','low']].copy(), orders