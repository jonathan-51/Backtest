import pandas as pd
from base import Strategy
from config import SMACrossoverConfig
from typing import Dict

class SMACrossover(Strategy):

    def __init__(self,
                 sma_slow_length = SMACrossoverConfig.slow_length,
                 sma_fast_length = SMACrossoverConfig.fast_length,
                 timeframe = SMACrossoverConfig.timeframe):
        self.sma_slow_length = sma_slow_length
        self.sma_fast_length = sma_fast_length
        self.timeframe = timeframe

    def generate_signals(self, data:Dict[str,pd.DataFrame]) -> tuple[pd.DataFrame,dict]:
        return 