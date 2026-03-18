from data_fetcher import DataFetcher
from config import DataConfig, DataFetcherConfig
import logging

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

    # Fetch data for all symbols
    all_data = fetcher.fetch_all_symbols()   

    return

if __name__ == "__main__":
    main(None,None)