from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self,data: Dict[str,pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals from market data."""
        pass