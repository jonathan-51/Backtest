# Backtest

A Python-based backtesting framework for trading strategies using Interactive Brokers data.

### Prerequisites
- Python 3.8+
- Interactive Brokers account (for Historical data)

### Installation

1. Clone the repository
2. Create a virtual environment: 
    python -m venv venv

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
    pip install -r requirements.txt


5. Create a `.env` file for IB credentials (if needed):
   IB_HOST=127.0.0.1
   IB_PORT=7497
   IB_CLIENT_ID=1

## Project Structure
Backtest/
├── backtest.py          # Core backtesting engine
├── run.py               # Main entry point
├── config.py            # Configuration settings
├── indicators.py        # Technical indicators
├── analytics.py         # Performance analytics
├── strategies/          # Trading strategies
│   └── undercut_breakout.py
├── data/                # Historical data storage
└── requirements.txt     # Python dependencies

## Adding New Strategies

1. Create a new file in `strategies/` folder
2. Implement your strategy class with required methods
3. Import and register it in `run.py`
4. Configure parameters in `config.py`

## Available Strategies

- **Undercut Breakout**: Identifies breakout patterns with undercut confirmation