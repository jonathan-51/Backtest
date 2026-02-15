import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from metrics import PerformanceMetrics
from config import MetricsConfig

config = MetricsConfig()

def make_results(trade_pnls, equity_values, dates=None):
    """Helper to build a fake results dict from trade PnLs and equity values"""
    if dates is None:
        dates = pd.date_range('2024-01-01', periods=len(equity_values), freq='D')

    trade_log = [{'pnl': pnl} for pnl in trade_pnls]
    winners = [t for t in trade_pnls if t > 0]
    losers = [t for t in trade_pnls if t < 0]

    return {
        'trade_log': trade_log,
        'equity_curve': pd.DataFrame({'date': dates, 'equity': equity_values}),
        'summary': {
            'max_drawdown': (min(equity_values) - max(equity_values)) / max(equity_values),
            'win_rate': len(winners) / len(trade_pnls) if trade_pnls else 0,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
        }
    }

# ── WIN/LOSS RATIO ─────────────────────────────────────

def test_win_loss_ratio_basic():
    results = make_results([200, -100], [10000, 10200, 10100])
    pm = PerformanceMetrics(results, config)

    # avg win=200, avg loss=100 → ratio=2.0
    assert pm.win_loss_ratio() == 2.0

def test_win_loss_ratio_no_losers():
    results = make_results([100, 200], [10000, 10100, 10300])
    pm = PerformanceMetrics(results, config)
    assert pm.win_loss_ratio() == float('inf')

def test_win_loss_ratio_no_winners():
    results = make_results([-100, -200], [10000, 9900, 9700])
    pm = PerformanceMetrics(results, config)
    assert pm.win_loss_ratio() == 0.0

# ── PROFIT FACTOR ──────────────────────────────────────

def test_profit_factor_basic():
    results = make_results([300, -100], [10000, 10300, 10200])
    pm = PerformanceMetrics(results, config)

    # gross_profit=300, gross_loss=100 → 3.0
    assert pm.profit_factor() == 3.0

def test_profit_factor_no_losers():
    results = make_results([100, 200], [10000, 10100, 10300])
    pm = PerformanceMetrics(results, config)
    assert pm.profit_factor() == float('inf')

# ── EXPECTANCY ─────────────────────────────────────────

def test_expectancy_basic():
    # 2 wins of 200 each, 2 losses of -100 each
    results = make_results([200, -100, 200, -100], [10000, 10200, 10100, 10300, 10200])
    pm = PerformanceMetrics(results, config)

    # win_rate=0.5, avg_win=200, avg_loss=100
    # expectancy = (0.5 * 200) - (0.5 * 100) = 50
    assert pm.expectancy() == 50.0

def test_expectancy_no_losers():
    results = make_results([100, 200], [10000, 10100, 10300])
    pm = PerformanceMetrics(results, config)
    assert pm.expectancy() == 0.0

# ── DRAWDOWN DURATION ──────────────────────────────────

def test_drawdown_avg_no_drawdown():
    # Equity only goes up → no drawdown
    results = make_results([], [100, 110, 120, 130])
    pm = PerformanceMetrics(results, config)
    assert pm.drawdown_avg(results['equity_curve']) == 0

def test_drawdown_avg_single():
    # Peak at 120, then 2 days in drawdown
    results = make_results([], [100, 120, 110, 105, 130])
    pm = PerformanceMetrics(results, config)

    # One drawdown lasting 2 days → avg = 2
    assert pm.drawdown_avg(results['equity_curve']) == 2.0

# ── VaR / CVaR ─────────────────────────────────────────

def test_var_is_negative():
    """VaR at 95% should be a negative number (it's a loss)"""
    equity = [10000 + i * 10 + ((-1)**i * 50) for i in range(100)]
    results = make_results([], equity)
    pm = PerformanceMetrics(results, config)
    assert pm.VaR(results['equity_curve']) < 0

def test_cvar_worse_than_var():
    """CVaR should be worse (more negative) than VaR"""
    equity = [10000 + i * 10 + ((-1)**i * 50) for i in range(100)]
    results = make_results([], equity)
    pm = PerformanceMetrics(results, config)
    assert pm.CVaR(results['equity_curve']) < pm.VaR(results['equity_curve'])

# ── ULCER INDEX ────────────────────────────────────────

def test_ulcer_index_no_drawdown():
    # Straight up → ulcer index = 0
    results = make_results([], [100, 110, 120, 130])
    pm = PerformanceMetrics(results, config)
    assert pm.ulcer_index(results['equity_curve']) == 0.0

def test_ulcer_index_with_drawdown():
    results = make_results([], [100, 120, 110, 130])
    pm = PerformanceMetrics(results, config)
    assert pm.ulcer_index(results['equity_curve']) > 0.0

# ── CALMAR RATIO ───────────────────────────────────────

def test_calmar_no_drawdown():
    results = make_results([], [100, 110, 120, 130])
    results['summary']['max_drawdown'] = 0
    pm = PerformanceMetrics(results, config)
    assert pm.calmar_ratio(results['equity_curve'], 0) == float('inf')

# ── SHARPE / SORTINO ───────────────────────────────────

def test_sharpe_zero_volatility():
    # Flat equity → zero volatility → returns 0.0
    results = make_results([], [100, 100, 100, 100])
    pm = PerformanceMetrics(results, config)
    assert pm.sharpe_ratio(results['equity_curve']) == 0.0

def test_sortino_zero_volatility():
    # Only up → no downside → returns 0.0
    results = make_results([], [100, 110, 120, 130])
    pm = PerformanceMetrics(results, config)
    assert pm.sortino_ratio(results['equity_curve']) == 0.0
