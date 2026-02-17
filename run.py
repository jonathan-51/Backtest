from dataclasses import asdict
from config import DataConfig, DataFetcherConfig, BacktestConfig, MetricsConfig, MonteCarloConfig, ATRChannelBreakoutConfig,RSIPullbackUptrendConfig, WalkForwardValidatorConfig,ATRChannelBreakoutOptimizerConfig,RSIPullbackOptimizerConfig
import logging
from data_fetcher import DataFetcher
from backtest import BacktestEngine
from metrics import PerformanceMetrics
from visualizer import BacktestVisualizer
from monte_carlo import MonteCarloSimulator
import numpy as np
import csv
import os
from datetime import datetime
from strategies.test_ma_crossover import MACrossoverStrategy
from strategies.atr_channel_breakout import ATRChannelBreakout
from strategies.rsi_pullback_uptrend import RSIPullbackUptrend
from validation import WalkForwardValidator, WalkForwardOptimizer

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
    results = backtest.run(all_data)

    # Create metrics engine
    get_metrics = PerformanceMetrics(results,metric_config)

    # Analyse results and compute metrics
    metrics = get_metrics.generate_metrics()

    summary = results['summary']
    for key, value in summary.items():
        if isinstance(value, np.number):
            value = float(value)
        logger.info(f"{key}: {value}")
    logger.info(50 * '=')
    for key, value in metrics.items():
        if isinstance(value, np.number):
            value = float(value)
        logger.info(f"{key}: {value}")

    # Log results and metrics in CSV file
    log_experiment(
        strategy_name=Strategy.__name__,
        symbol=','.join(data_config.symbols),
        summary=results['summary'],
        metrics=metrics,
        strategy_config=StrategyConfig(),
        data=list(all_data.values())[0][StrategyConfig().timeframe]
    )

    # Run Monte Carlo simulation
    mc_config = MonteCarloConfig()
    mc_results = MonteCarloSimulator(results, mc_config).run()

    logger.info(50 * '=')
    for key, value in mc_results['statistics'].items():
        logger.info(f"{key}: {value}")

    # Create plotter object
    plotter = BacktestVisualizer(results, metrics, mc_results)

    # Plot metrics
    plotter.generate_report()

def log_experiment(strategy_name,symbol,summary,metrics,strategy_config,data) -> None:
    """Append backtest results to CSV log."""

    row = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'lookback_period':DataConfig.lookback_period,
        'start_date':data['date'].iloc[0],
        'end_date':data['date'].iloc[-1],
        'bars':len(data),
        'strategy':strategy_name,
        'change_made':'NaN',
        'symbol': symbol,
        'initial_capital': BacktestConfig.initial_capital,
        'commission_rate': BacktestConfig.commission_rate,
        'slippage': BacktestConfig.slippage,
        'position_size': BacktestConfig.position_size,
        **{f'param_{k}':v for k,v in strategy_config.__dict__.items()},
        **summary,
        **metrics,
    }

    os.makedirs('logs', exist_ok=True)
    file_exists = os.path.exists(f'logs/{strategy_name}_log.csv')

    with open(f'logs/{strategy_name}_log.csv','a',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def validate(Strategy) -> None:
    """Run walk-forward validation on strategy"""
    # Create data config and data fetcher config objects
    data_config = DataConfig()
    data_fetcher_config = DataFetcherConfig()

    # Create data fetcher object
    fetcher = DataFetcher(data_fetcher_config,data_config)

    # Fetch data
    data = fetcher.fetch_all_timeframes(data_config.symbol)

    if data is None:
        return
    
    # Create validator object
    validator = WalkForwardValidator(Strategy(),data)

    # Computing validation results
    results = validator.run()

    # Log per-window results
    for r in results['windows']:
        logger.info(f"Window {r['window']}: train_sharpe={r['train_sharpe']:.4f}, "
              f"test_sharpe={r['test_sharpe']:.4f}, degradation={r['degradation']:.2%}")

    logger.info(f"Avg Degradation: {results['avg_degradation']:.2%}")
    logger.info(f"Verdict: {results['verdict']}")

def optimize(Strategy, param_grid) -> None:
    """Run walk-forward optimization with parameter sweeps."""
    data_config = DataConfig()
    data_fetcher_config = DataFetcherConfig()

    fetcher = DataFetcher(data_fetcher_config, data_config)
    data = fetcher.fetch_all_timeframes(data_config.symbol)

    if data is None:
        return

    optimizer = WalkForwardOptimizer(Strategy, param_grid, data)
    results = optimizer.run()

    for r in results['windows']:
        logger.info(f"Window {r['window']}: best_params={r['best_params']}, "
                    f"train_sharpe={r['train_sharpe']:.4f}, test_sharpe={r['test_sharpe']:.4f}, "
                    f"degradation={r['degradation']:.2%}")

    logger.info(f"Avg Degradation: {results['avg_degradation']:.2%}")
    logger.info(f"Verdict: {results['verdict']}")

if __name__ == "__main__":
    main(ATRChannelBreakout,ATRChannelBreakoutConfig)
    #validate(ATRChannelBreakout)
    #validate(RSIPullbackUptrend)
    #optimize(ATRChannelBreakout, asdict(ATRChannelBreakoutOptimizerConfig()))
