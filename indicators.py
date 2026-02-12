import pandas as pd

class Indicators:

    def __init__(self):
        pass

    def sma(self, series:pd.DataFrame,period:int) -> pd.DataFrame:
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