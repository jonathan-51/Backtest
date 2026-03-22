from abc import ABC,abstractmethod
from typing import Dict,Tuple
import pandas as pd
class Strategy(ABC):
    @abstractmethod
    def generate_signals(self,data: Dict[str,pd.DataFrame]) -> Tuple[pd.DataFrame,dict]:

        pass