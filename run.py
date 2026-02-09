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
    backtest = BacktestEngine(BacktestConfig,MACrossoverStrategy)

    # Connect
    if not fetcher.connect():
        return
    
    # Fetch data
    aapl_data = fetcher.fetch_all_timeframes('AAPL')

    if aapl_data is None:
        fetcher.disconnect()
        return    
    
    # Disconnect
    fetcher.disconnect()

    backtest.run(aapl_data['1d'])

if __name__ == "__main__":
    main()
