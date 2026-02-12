from config import DataConfig, DataFetcherConfig, BacktestConfig, MetricsConfig
import logging
from data_fetcher import DataFetcher
from backtest import BacktestEngine
from strategies.test_ma_crossover import MACrossoverStrategy
from strategies.atr_channel_breakout import ATRChannelBreakout
from metrics import PerformanceMetrics
from visualizer import BacktestVisualizer
import numpy as np

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
    metric_config = MetricsConfig()

    # Create fetcher
    fetcher = DataFetcher(data_fetcher_config,data_config)

    # Create backtest engine
    backtest = BacktestEngine(BacktestConfig,ATRChannelBreakout())
    
    # Fetch data
    aapl_data = fetcher.fetch_all_timeframes('AAPL')

    if aapl_data is None:
        return    

    results = backtest.run(aapl_data)

    # Create metrics engine
    get_metrics = PerformanceMetrics(results,metric_config)

    metrics = get_metrics.generate_metrics()

    summary = results['summary']
    for key, value in summary.items():
        if isinstance(value, np.number):
            value = float(value)
        print(f"{key}: {value}")
    print(50 * '=')
    for key, value in metrics.items():
        if isinstance(value, np.number):
            value = float(value)
        print(f"{key}: {value}")

    plotter = BacktestVisualizer(results,metrics)

    plotter.generate_report()

if __name__ == "__main__":
    main()
