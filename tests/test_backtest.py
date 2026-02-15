import pandas as pd
import sys
sys.path.insert(0, '.')
from backtest import BacktestEngine
from config import BacktestConfig

# Zero slippage/spreads/commission config for predictable math
config = BacktestConfig(
    initial_capital=10000,
    commission_rate=0.0,
    slippage=0.0,
    spreads=0.0,
    position_size=0.5,
)

# ── SLIPPAGE & SPREADS ─────────────────────────────────

def make_engine():
    """Helper to create a fresh engine with a dummy strategy"""
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    return BacktestEngine(config, DummyStrategy())

def test_apply_slippage_buy():
    engine = make_engine()
    # Buy slippage should increase price
    result = engine._apply_slippage(100.0, 'buy')
    assert result >= 100.0

def test_apply_slippage_sell():
    engine = make_engine()
    # Sell slippage should decrease price
    result = engine._apply_slippage(100.0, 'sell')
    assert result <= 100.0

def test_slippage_with_config():
    slippage_config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.0,
        slippage=0.01,   # 1%
        spreads=0.0,
        position_size=0.5,
    )
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    engine = BacktestEngine(slippage_config, DummyStrategy())

    # Buy at 100 with 1% slippage = 101
    assert engine._apply_slippage(100.0, 'buy') == 101.0
    # Sell at 100 with 1% slippage = 99
    assert engine._apply_slippage(100.0, 'sell') == 99.0

# ── COMMISSION ─────────────────────────────────────────

def test_calculate_commission():
    comm_config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.001,  # 0.1%
        slippage=0.0,
        spreads=0.0,
        position_size=0.5,
    )
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    engine = BacktestEngine(comm_config, DummyStrategy())

    # 0.1% of $10,000 = $10
    assert engine._calculate_commission(10000) == 10.0

# ── POSITION SIZING ────────────────────────────────────

def test_calculate_shares():
    engine = make_engine()
    # cash=10000, position_size=0.5, so budget=5000
    # price=100, no slippage → 5000 // 100 = 50 shares
    shares = engine._calculate_shares(100.0)
    assert shares == 50

def test_calculate_shares_insufficient_cash():
    tiny_config = BacktestConfig(
        initial_capital=10,
        commission_rate=0.0,
        slippage=0.0,
        spreads=0.0,
        position_size=0.5,
    )
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    engine = BacktestEngine(tiny_config, DummyStrategy())

    # cash=10, position_size=0.5, budget=5, price=100 → 0 shares
    shares = engine._calculate_shares(100.0)
    assert shares == 0

# ── EXECUTE BUY ────────────────────────────────────────

def test_execute_buy():
    engine = make_engine()
    engine._execute_buy('2024-01-01', 100.0)

    # Should have a position
    assert engine.position is not None
    assert engine.position['shares'] == 50
    assert engine.position['entry_price'] == 100.0

    # Cash should decrease: 10000 - (50 * 100) = 5000
    assert engine.cash == 5000.0

def test_execute_buy_insufficient_cash():
    tiny_config = BacktestConfig(
        initial_capital=10,
        commission_rate=0.0,
        slippage=0.0,
        spreads=0.0,
        position_size=0.5,
    )
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    engine = BacktestEngine(tiny_config, DummyStrategy())
    engine._execute_buy('2024-01-01', 100.0)

    # Should NOT open a position
    assert engine.position is None
    assert engine.cash == 10

# ── EXECUTE SELL ───────────────────────────────────────

def test_execute_sell_profit():
    engine = make_engine()
    engine._execute_buy('2024-01-01', 100.0)
    engine._execute_sell('2024-01-02', 110.0)

    # Position should be closed
    assert engine.position is None

    # PnL: 50 shares * (110 - 100) = 500
    assert engine.trade_log[-1]['pnl'] == 500.0

    # Cash: started 10000, bought 50@100, sold 50@110 = 10500
    assert engine.cash == 10500.0

def test_execute_sell_loss():
    engine = make_engine()
    engine._execute_buy('2024-01-01', 100.0)
    engine._execute_sell('2024-01-02', 90.0)

    # PnL: 50 shares * (90 - 100) = -500
    assert engine.trade_log[-1]['pnl'] == -500.0
    assert engine.cash == 9500.0

# ── STATE RESET ────────────────────────────────────────

def test_run_resets_state():
    from strategies.base import Strategy
    class BuyAndHoldStrategy(Strategy):
        def generate_signals(self, data):
            df = data['1d'].copy()
            df['signal'] = 'hold_cash'
            df.loc[df.index[0], 'signal'] = 'buy'
            return df[['date', 'signal', 'close', 'low']], {}

    engine = BacktestEngine(config, BuyAndHoldStrategy())
    data = {'1d': pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'close': [100.0, 105.0, 110.0],
        'low': [99.0, 104.0, 109.0],
        'high': [101.0, 106.0, 111.0],
    })}

    engine.run(data)

    # After run, state should be reset
    assert engine.cash == config.initial_capital
    assert engine.position is None
    assert engine.trade_log == []
    assert engine.equity_curve == []