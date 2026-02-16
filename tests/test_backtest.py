import pandas as pd
import sys
sys.path.insert(0, '.')
from backtest import BacktestEngine
from config import BacktestConfig

# Zero slippage/spreads/commission config for predictable math
# max_positions=3, so budget per trade = 10000 / 3 = 3333
config = BacktestConfig(
    initial_capital=10000,
    commission_rate=0.0,
    slippage=0.0,
    spreads=0.0,
    position_size=0.5,
    max_positions=3,
)

# ── SLIPPAGE & SPREADS ─────────────────────────────────

def make_engine(custom_config=None):
    """Helper to create a fresh engine with a dummy strategy"""
    from strategies.base import Strategy
    class DummyStrategy(Strategy):
        def generate_signals(self, data):
            return pd.DataFrame(), {}
    return BacktestEngine(custom_config or config, DummyStrategy())

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
    engine = make_engine(slippage_config)

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
    engine = make_engine(comm_config)

    # 0.1% of $10,000 = $10
    assert engine._calculate_commission(10000) == 10.0

# ── EXECUTE BUY ────────────────────────────────────────

def test_execute_buy():
    engine = make_engine()
    engine._execute_buy('AAPL', '2024-01-01', 100.0)

    # Should have a position
    assert 'AAPL' in engine.position
    assert engine.position['AAPL']['shares'] == 33  # 10000/3 = 3333, 3333//100 = 33
    assert engine.position['AAPL']['entry_price'] == 100.0

    # Cash should decrease: 10000 - (33 * 100) = 6700
    assert engine.cash == 6700.0

def test_execute_buy_insufficient_cash():
    tiny_config = BacktestConfig(
        initial_capital=10,
        commission_rate=0.0,
        slippage=0.0,
        spreads=0.0,
        position_size=0.5,
        max_positions=3,
    )
    engine = make_engine(tiny_config)
    engine._execute_buy('AAPL', '2024-01-01', 100.0)

    # Should NOT open a position (budget = 10/3 = 3.33, can't buy any shares at 100)
    assert 'AAPL' not in engine.position
    assert engine.cash == 10

# ── EXECUTE SELL ───────────────────────────────────────

def test_execute_sell_profit():
    engine = make_engine()
    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    engine._execute_sell('AAPL', '2024-01-02', 110.0)

    # Position should be closed
    assert 'AAPL' not in engine.position

    # PnL: 33 shares * (110 - 100) = 330
    assert engine.trade_log[-1]['pnl'] == 330.0

    # Cash: started 10000, bought 33@100, sold 33@110 = 10330
    assert engine.cash == 10330.0

def test_execute_sell_loss():
    engine = make_engine()
    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    engine._execute_sell('AAPL', '2024-01-02', 90.0)

    # PnL: 33 shares * (90 - 100) = -330
    assert engine.trade_log[-1]['pnl'] == -330.0
    assert engine.cash == 9670.0

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
    assert engine.position == {}
    assert engine.trade_log == []
    assert engine.equity_curve == []

# ── MULTI-SYMBOL ───────────────────────────────────────

def test_equal_allocation():
    """Each position gets the same budget: initial_capital / max_positions"""
    engine = make_engine()

    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    engine._execute_buy('XOM', '2024-01-01', 100.0)

    # Both should have the same number of shares
    assert engine.position['AAPL']['shares'] == engine.position['XOM']['shares']
    assert engine.position['AAPL']['shares'] == 33  # 10000/3 // 100

def test_max_positions_enforced():
    """Cannot open more positions than max_positions"""
    two_slot_config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.0,
        slippage=0.0,
        spreads=0.0,
        position_size=0.5,
        max_positions=2,
    )
    from strategies.base import Strategy
    class ThreeBuyStrategy(Strategy):
        def generate_signals(self, data):
            df = data['1d'].copy()
            df['signal'] = 'buy'
            return df[['date', 'signal', 'close', 'low']], {}

    engine = BacktestEngine(two_slot_config, ThreeBuyStrategy())
    data = {
        'AAPL': {'1d': pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'close': [100.0, 105.0], 'low': [99.0, 104.0], 'high': [101.0, 106.0],
        })},
        'XOM': {'1d': pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'close': [80.0, 85.0], 'low': [79.0, 84.0], 'high': [81.0, 86.0],
        })},
        'JPM': {'1d': pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'close': [150.0, 155.0], 'low': [149.0, 154.0], 'high': [151.0, 156.0],
        })},
    }

    results = engine.run(data)

    # max_positions=2, so only 2 symbols should have traded
    symbols_traded = set(t['symbol'] for t in results['trade_log'])
    assert len(symbols_traded) <= 2

def test_trade_log_has_symbol():
    """Each trade log entry should include the symbol"""
    engine = make_engine()
    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    engine._execute_sell('AAPL', '2024-01-02', 110.0)

    assert engine.trade_log[-1]['symbol'] == 'AAPL'

def test_shared_cash_pool():
    """Buying multiple symbols depletes the same cash pool"""
    engine = make_engine()

    cash_before = engine.cash
    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    cash_after_first = engine.cash
    engine._execute_buy('XOM', '2024-01-01', 100.0)
    cash_after_second = engine.cash

    # Each buy costs 33 * 100 = 3300
    assert cash_after_first == cash_before - 3300
    assert cash_after_second == cash_after_first - 3300

def test_combined_equity():
    """Equity = cash + sum of all position market values"""
    engine = make_engine()

    engine._execute_buy('AAPL', '2024-01-01', 100.0)
    engine._execute_buy('XOM', '2024-01-01', 50.0)

    # AAPL: 33 shares @ 100 = 3300
    # XOM: 10000/3 = 3333, 3333//50 = 66 shares @ 50 = 3300
    # Cash: 10000 - 3300 - 3300 = 3400
    # Total equity at current prices: 3400 + 33*100 + 66*50 = 3400 + 3300 + 3300 = 10000
    expected_equity = engine.cash + (33 * 100) + (66 * 50)
    assert expected_equity == 10000
