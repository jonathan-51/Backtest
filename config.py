from dataclasses import dataclass,field
from typing import Optional, List
from datetime import datetime
import os

@dataclass
class BacktestConfig:
    """Configuration for backtest simulation parameters including 
    capital, costs, and position sizing"""
    
    
    initial_capital: float = 30000
    commission_rate: float = 0.005
    slippage: float = 0.0001
    spreads: float = 0.01
    position_size: float = 0.9
    max_positions: int = 1

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("Initial Capital must be positive")
        if not (0 <= self.position_size <= 1):
            raise ValueError("Position size must be between 0 and 1")
        
@dataclass
class DataConfig:
    """
    Configuration for market data retrieval including 
    ticker symbols, timeframe, storage path, and date range
    """
    symbols: List[str] = field(default_factory=lambda: ['XOM','KOS'])
    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    data_path: str = './data'
    cache_data:bool = True
    lookback_period: str = "1 Y"
    end_date: datetime = datetime(2025,1,1)
    min_bars_required: int = 20

    def __post_init__(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

@dataclass
class DataFetcherConfig:
    """Configuration for Interactive Brokers TWS connection settings
    
    Attributes:
        request_pause: Seconds to wait between API requests to avoid rate limiting
    """
    tws_host: str = "127.0.0.1"
    tws_port: int = 7497
    client_id: int = 1
    request_pause: int = 11

@dataclass
class SMACrossoverConfig:
    """Configuration for SMA Crossover Strategy"""
    fast_length: int = 20
    slow_length: int = 50
    timeframe: str = '1d'
    stop_loss_percent: float = 0.01
    take_profit_percent: float = 0.02