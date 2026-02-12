from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self,data: Dict[str,pd.DataFrame]) -> Tuple[pd.DataFrame,dict]:
        """Generate trading signals from market data.
        
        Returns:
            signals: Dataframe [date, signal, close, high, low]
            orders: {row_index: Order} - only for 'buy rows
        """
        pass