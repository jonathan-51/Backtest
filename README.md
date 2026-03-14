# Backtest

A Python backtesting framework for evaluating trading strategies against historical market data sourced from Interactive Brokers.

## Features

- Long and short selling support
- Configurable slippage, commissions, and spread modeling
- Stop-loss, trailing stop, and take-profit order types
- Multi-symbol and multi-timeframe support
- Walk-forward validation
- Monte Carlo simulation
- Market regime filtering
- Performance metrics and equity curve visualization

## Prerequisites

- Python 3.8+
- Interactive Brokers TWS or IB Gateway (for fetching historical data)

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
Backtest/
├── run.py               # Main entry point
├── backtest.py          # Core backtesting engine
├── config.py            # Configuration for all modules
├── order.py             # Order types (stop-loss, trailing, take-profit)
├── data_fetcher.py      # IB historical data retrieval
├── indicators.py        # Technical indicators
├── metrics.py           # Performance metrics
├── monte_carlo.py       # Monte Carlo trade resampling
├── regime_filter.py     # Market regime scoring
├── validation.py        # Walk-forward validation
├── visualizer.py        # Equity curve and result plotting
├── strategies/          # Trading strategies
│   └── base.py          # Abstract strategy interface
└── data/                # Cached historical data
```

## Adding a Strategy

1. Create a new file in `strategies/`
2. Subclass `Strategy` from `strategies/base.py` and implement `generate_signals()`
3. Add configuration parameters to `config.py`
4. Import and use the strategy in `run.py`
