from data_fetcher import DataFetcher
from config import DataConfig, DataFetcherConfig, BacktestConfig, SMACrossoverConfig
import logging
from backtest import BacktestEngine
from strategy.SMA_Crossover import SMACrossover

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

    print(results['summary']['final_equity'])
    return

if __name__ == "__main__":
    main(SMACrossover,SMACrossoverConfig)