from data_fetcher import DataFetcher
from config import DataConfig, DataFetcherConfig, BacktestConfig, SMACrossoverConfig, MetricsConfig
import logging
from backtest import BacktestEngine
from strategy.SMA_Crossover import SMACrossover
from metrics import PerformanceMetrics

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.getLogger("ib_insync").setLevel(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main(Strategy,StrategyConfig) -> None:
    """Computes complete backtest results"""

    # Initialize configs
    data_config = DataConfig()
    data_fetcher_config = DataFetcherConfig()
    metric_config = MetricsConfig()

    # Create fetcher
    fetcher = DataFetcher(data_fetcher_config,data_config)
    
    # Create backtest engine
    backtest = BacktestEngine(BacktestConfig(),Strategy())

    # Fetch data for all symbols
    all_data = fetcher.fetch_all_symbols()   

    if not all_data:
        return
    
    # Run backtest
    results = backtest.run(all_data,StrategyConfig())

    # Create metrics engine
    get_metrics = PerformanceMetrics(results,metric_config)

    metrics = get_metrics.generate_metrics()

    print(metrics)
    return

if __name__ == "__main__":
    main(SMACrossover,SMACrossoverConfig)