import pandas as pd

class Indicators:
    """Technical indicator library for strategy signal generation."""
    def __init__(self):
        pass

    def sma(self, series:pd.Series,period:int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def true_range(self,df:pd.DataFrame) -> pd.Series:
        """True Range - accounts for gaps between bars"""

        prev_close = df['close'].shift(1)
        tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
                ],axis=1).max(axis=1)
        
        return tr
    
    def atr(self,df:pd.DataFrame,period:int) -> pd.Series:
        """Average True Range (simple rolling mean of True Range)"""

        return self.true_range(df).rolling(window=period).mean()
    
    def rsi(self,series:pd.Series,period: int) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing (EWM with alpha=1/period)."""
        delta = series.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(alpha=1/period,min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1/period,min_periods=period).mean()

        rs = avg_gain/avg_loss

        return 100 - (100 / (1 + rs))