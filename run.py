from config import DataConfig, DataFetcherConfig, BacktestConfig
import logging
from data_fetcher import DataFetcher
from backtest import BacktestEngine
from strategies.test_ma_crossover import MACrossoverStrategy

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.getLogger("ib_insync").setLevel(level=logging.WARNING)

def main():
    """Test the data fetching pipeline end-to-end"""

    # Initialize configs
    data_config = DataConfig()
    data_fetcher_config = DataFetcherConfig()
    

    # Create fetcher
    fetcher = DataFetcher(data_fetcher_config,data_config)

    # Create backtest engine
    backtest = BacktestEngine(BacktestConfig,MACrossoverStrategy())

    
    # Fetch data
    aapl_data = fetcher.fetch_all_timeframes('AAPL')

    if aapl_data is None:
        return    

    results = backtest.run(aapl_data)

    print(results)
if __name__ == "__main__":
    main()
