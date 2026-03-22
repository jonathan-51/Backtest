import pandas as pd

class Indicators:
    """Technical indicator library for strategy signal generation"""
    def __init__(self):
        pass

    def sma(self, series:pd.Series,period:int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()