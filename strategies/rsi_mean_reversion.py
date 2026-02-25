import pandas as pd
from strategies.base import Strategy
from config import RSIMeanReversionConfig
from indicators import Indicators
from typing import Dict
from order import Order

class RSIMeanReversion(Strategy):
    def __init__(self,
                top_threshold = RSIMeanReversionConfig.top_threshold,
                bottom_threshold = RSIMeanReversionConfig.bottom_threshold,
                rsi_ema_length = RSIMeanReversionConfig.rsi_ema_length,
                rsi_length = RSIMeanReversionConfig.rsi_length,
                atr_length = RSIMeanReversionConfig.atr_length,
                rsi_ema_gap_threshold = RSIMeanReversionConfig.rsi_ema_gap_threshold,
                ema_slow_length = RSIMeanReversionConfig.ema_slow_length,
                ema_trend_length = RSIMeanReversionConfig.ema_trend_length,
                stop_mult = RSIMeanReversionConfig.stop_mult,
                timeframe = RSIMeanReversionConfig.timeframe):
        self.top_threshold = top_threshold
        self.bottom_threshold = bottom_threshold
        self.rsi_ema_length = rsi_ema_length
        self.rsi_length = rsi_length
        self.atr_length = atr_length
        self.rsi_ema_gap_threshold = rsi_ema_gap_threshold
        self.ema_slow_length = ema_slow_length
        self.ema_trend_length = ema_trend_length
        self.stop_mult = stop_mult
        self.timeframe = timeframe
        self.indicator = Indicators()

    def generate_signals(self,data: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame,dict]:
        df = data[self.timeframe].copy()

        # Compute Indicators
        df['rsi'] = self.indicator.rsi(df['close'],self.rsi_length)
        df['rsi_ema'] = self.indicator.ema(df['rsi'],self.rsi_ema_length)
        df['atr'] = self.indicator.atr(df,self.atr_length)
        df['ema_slow'] = self.indicator.ema(df['close'],self.ema_slow_length)
        df['ema_trend'] = self.indicator.ema(df['close'],self.ema_trend_length)

        df['signal'] = 'hold_cash'
        position_state = None
        orders = {}

        for i in range (1,len(df)):
            curr_rsi = df['rsi'].iloc[i]
            curr_rsi_ema = df['rsi_ema'].iloc[i]
            curr_atr = df['atr'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_ema_slow = df['ema_slow'].iloc[i]
            curr_ema_trend = df['ema_trend'].iloc[i]
            rsi_gap = curr_rsi - curr_rsi_ema
            
            if position_state is None:
                # Entry: go long
                if (curr_rsi < self.bottom_threshold or rsi_gap < -self.rsi_ema_gap_threshold) and curr_close < curr_ema_slow and curr_close < curr_ema_trend:
                    df.loc[df.index[i],'signal'] = 'buy'
                    orders[df.index[i]] = Order(stop_loss=curr_close - (curr_atr * self.stop_mult))
                    position_state='long'

                # Entry: go short
                elif (curr_rsi > self.top_threshold or rsi_gap > self.rsi_ema_gap_threshold) and curr_close > curr_ema_slow and curr_close > curr_ema_trend:
                    df.loc[df.index[i],'signal'] = 'short'
                    orders[df.index[i]] = Order(stop_loss=curr_close + (curr_atr * self.stop_mult))
                    position_state='short'

            elif position_state == 'long':
                # Exit long: RSI reverts back to EMA
                if curr_rsi >= curr_rsi_ema:
                    df.loc[df.index[i],'signal'] = 'sell'
                    position_state = None
                
            elif position_state == 'short':
                # Exit Short: RSI reverts back to EMA
                if curr_rsi <= curr_rsi_ema:
                    df.loc[df.index[i],'signal'] = 'cover'
                    position_state = None


        return df[['date','signal','close','high','low']].copy(), orders