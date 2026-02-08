from dataclasses import dataclass,field
from datetime import datetime, timedelta
from typing import List
import os

@dataclass
class BacktestConfig:
    initial_capital: float = 30000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    spreads: float = 0.1
    position_size: float = 0.33
    max_positions: int = 3

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("Initial Capital must be positive")
        if not (0 <= self.position_size <= 1):
            raise ValueError("Position size must be between 0 and 1")
        
@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda:['AAPL', 'MSFT', 'GOOGL','TSLA'])
    timeframe: str = '1d'
    data_path: str = './data'
    cache_data:bool = True
    lookback_period: str = "1 Y"
    end_date: datetime = datetime(2025,1,1)

    def __post_init__(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

@dataclass
class DataFetcherConfig:
    tws_host: str = "127.0.0.1"
    tws_port: int = 7497
    client_id: int = 1
    request_pause: int = 11

