from dataclasses import dataclass,field
from datetime import datetime, timedelta
from typing import List, Optional
import os

@dataclass
class BacktestConfig:
    """Configuration for backtest simulation parameters including 
    capital, costs, and position sizing"""
    
    
    initial_capital: float = 30000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    spreads: float = 0.1
    position_size: float = 0.16
    max_positions: int = 6

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
    symbols: List[str] = field(default_factory=lambda: [
    # Tech / Growth
    'AAPL',   # Apple - mega cap tech
    'MSFT',   # Microsoft - cloud/AI
    'NVDA',   # Nvidia - GPU/AI
    'META',   # Meta - social media
    'GOOGL',  # Alphabet - search/cloud
    'AMZN',   # Amazon - ecommerce/cloud
    'AMAT',   # Applied Materials - semis equipment
    'PANW',   # Palo Alto - cybersecurity
    # Finance
    'JPM',    # JPMorgan - large bank
    'GS',     # Goldman Sachs - investment bank
    'V',      # Visa - payments
    'MA',     # Mastercard - payments
    'BAC',    # Bank of America
    # Healthcare
    'UNH',    # UnitedHealth - managed care
    'JNJ',    # Johnson & Johnson - diversified
    'LLY',    # Eli Lilly - pharma/GLP-1
    'ABBV',   # AbbVie - biotech
    # Energy
    'XOM',    # ExxonMobil
    'CVX',    # Chevron
    'SLB',    # SLB - oilfield services
    # Industrials
    'CAT',    # Caterpillar - machinery
    'DE',     # Deere - ag equipment
    'LMT',    # Lockheed Martin - defense
    'UNP',    # Union Pacific - rail
    # Consumer Discretionary
    'HD',     # Home Depot
    'NKE',    # Nike
    'TSLA',   # Tesla - EV
    # Consumer Staples
    'COST',   # Costco
    'PG',     # Procter & Gamble
    # Materials
    'FCX',    # Freeport-McMoRan - copper/gold
])
    timeframes: List[str] = field(default_factory=lambda: ['1d'])
    data_path: str = './data'
    cache_data:bool = True
    lookback_period: str = "10 Y"
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
    calendar_days: float = 365

@dataclass
class ATRChannelBreakoutConfig:
    timeframe: str = '1d'
    sma_length: int = 20
    atr_length: int = 20
    envelope_mult: float = 1.5
    stop_mult: int = 2
    trail_mult: int = 3
    spy_sma_length: int = 0   # set to 0 to disable regime filter

@dataclass
class RSIPullbackUptrendConfig:
    """Configuration for Connors-style RSI pullback strategy."""
    timeframe: str = '1d'
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
    timeframe: str = '1d'
    warm_up_bars: int = 200
    train_bars: int = 700
    test_bars: int = 200
    min_remaining_bars: int = 50
    avg_degradation_threshold_pass: float = -0.25
    avg_degradation_threshold_marginal: float = -0.50

@dataclass
class ATRChannelBreakoutOptimizerConfig:
    """Configuration for walk-forward optimizer"""
    sma_length: list = field(default_factory=lambda: [20,30])
    atr_length: list = field(default_factory=lambda: [20,30])
    envelope_mult: list = field(default_factory=lambda: [1.5,2.0])
    stop_mult: list = field(default_factory=lambda: [2,3])
    trail_mult: list = field(default_factory=lambda: [3,4])

@dataclass
class RSIPullbackOptimizerConfig:
    rsi_length: list = field(default_factory=lambda: [2, 3, 5])
    rsi_entry_threshold: list = field(default_factory=lambda: [15, 25, 35])
    rsi_exit_threshold: list = field(default_factory=lambda: [60, 70, 80])
    exit_sma_length: list = field(default_factory=lambda: [5, 10])
    atr_length: list = field(default_factory=lambda: [10, 15, 20])
    stop_mult: list = field(default_factory=lambda: [1.5, 2, 3])

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo trade resampling simulation."""
    n_simulations: int = 1000
    percentiles: list = field(default_factory=lambda: [5, 25, 50, 75, 95])
    ruin_threshold: float = 0.5
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not (0 < self.ruin_threshold < 1):
            raise ValueError("ruin_threshold must be between 0 and 1")
        
@dataclass
class RegimeFilterConfig:
    """Configuration for the market regime scoring filter."""
    sma_long: int = 200          # SMA period for SPY/QQQ trend checks
    sma_short: int = 50          # SMA period for HYG credit check
    sma_ratio: int = 20          # Rolling window for rotation ratio trend
    vix_threshold: float = 20.0  # VIX level below which regime is risk-on
    regime_tickers: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'RSP',
        'HYG', 'TLT',
        'SPHB', 'SPLV', 'XLY', 'XLP',
    ])  # VIX excluded: IB cannot fetch index data as a standard security

@dataclass
class EMAConsolidationBreakoutConfig:
    ema_fast_length: int = 8
    ema_slow_length: int = 20
    ema_trend_length: int = 50
    ema_filter_length: int = 200
    atr_length: int = 14
    consolidation_bar_length: int = 15
    consolidation_mult: float = 3.0
    timeframe: str = '1d'
    use_regime_filter: bool = True
    regime_score_threshold: int = 3  # Minimum score (0-8) to allow buy signals

@dataclass
class EMAConsolidationBreakoutOptimizerConfig:
    ema_fast_length: list = field(default_factory=lambda: [2, 3, 5])
    ema_slow_length: list = field(default_factory=lambda: [15, 25, 35])
    ema_trend_length: list = field(default_factory=lambda: [60, 70, 80])
    ema_filter_length: list = field(default_factory=lambda: [5, 10])
    atr_length: list = field(default_factory=lambda: [10, 15, 20])
    consolidation_bar_length: list = field(default_factory=lambda: [1.5, 2, 3])