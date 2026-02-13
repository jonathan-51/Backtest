from dataclasses import dataclass,field
from datetime import datetime, timedelta
from typing import List
import os

@dataclass
class BacktestConfig:
    """Configuration for backtest simulation parameters including 
    capital, costs, and position sizing"""
    
    
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
    """
    Configuration for market data retrieval including 
    ticker symbols, timeframe, storage path, and date range
    """
    symbol: str = 'AAPL'
    timeframes: List[str] = field(default_factory=lambda: ['1d'])
    data_path: str = './data'
    cache_data:bool = True
    lookback_period: str = "5 Y"
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
class MetricsConfig:
    risk_free_rate: float = 0.02
    VaR_confidence: float = 0.95
    trading_days: float = 252
    calender_days: float = 365

@dataclass
class ATRChannelBreakoutConfig:
    sma_length: int = 20
    atr_length: int = 20
    envelope_mult: float = 1.5
    stop_mult: int = 2
    trail_mult: int = 3

@dataclass
class RSIPullbackUptrendConfig:
    """Configuration for Connors-style RSI pullback strategy."""
    entry_sma_length: int = 200
    rsi_length: int = 2
    rsi_entry_threshold: int = 25
    rsi_exit_threshold: int = 70
    exit_sma_length: int = 5
    atr_length: int = 15
    stop_mult: float = 2

    def __post_init__(self):
        if not (0 < self.rsi_entry_threshold < self.rsi_exit_threshold < 100):
            raise ValueError("RSI thresholds must be 0 < entry < exit < 100")
        if self.stop_mult <= 0:
            raise ValueError("stop_mult must be positive")
        
@dataclass
class WalkForwardValidatorConfig:
    """Configuration for walk-forward validation"""
    warm_up_bars: int = 200
    train_bars: int = 700
    test_bars: int = 200
    min_remaining_bars: int = 50
    avg_degradation_threshold_pass: float = -0.25
    avg_degradation_threshold_marginal: float = -0.50

