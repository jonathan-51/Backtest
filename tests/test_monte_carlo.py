import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from monte_carlo import MonteCarloSimulator
from config import MonteCarloConfig


def make_results(pnls, initial_capital=10000):
    """Build a minimal results dict from a list of PnLs."""
    trade_log = [{'pnl': p} for p in pnls]
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity_curve = pd.DataFrame({
        'date': range(len(equity)),
        'equity': equity,
    })
    return {'trade_log': trade_log, 'equity_curve': equity_curve}


# ── BASIC FUNCTIONALITY ──────────────────────────────────

def test_simulation_count():
    results = make_results([100, -50, 200, -30, 150])
    config = MonteCarloConfig(n_simulations=500, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    assert len(mc['final_equities']) == 500
    assert len(mc['max_drawdowns']) == 500
    assert mc['equity_curves'].shape[0] == 500
    assert mc['equity_curves'].shape[1] == 6  # initial + 5 trades


def test_reproducibility():
    results = make_results([100, -50, 200, -30, 150])
    config = MonteCarloConfig(n_simulations=100, random_seed=42)

    mc1 = MonteCarloSimulator(results, config).run()
    mc2 = MonteCarloSimulator(results, config).run()

    np.testing.assert_array_equal(mc1['final_equities'], mc2['final_equities'])
    np.testing.assert_array_equal(mc1['max_drawdowns'], mc2['max_drawdowns'])


def test_different_seeds_differ():
    results = make_results([100, -50, 200, -30, 150])

    mc1 = MonteCarloSimulator(results, MonteCarloConfig(n_simulations=100, random_seed=1)).run()
    mc2 = MonteCarloSimulator(results, MonteCarloConfig(n_simulations=100, random_seed=2)).run()

    assert not np.array_equal(mc1['final_equities'], mc2['final_equities'])


# ── PROBABILITY METRICS ──────────────────────────────────

def test_all_profitable_trades():
    results = make_results([100, 200, 300, 150, 250])
    config = MonteCarloConfig(n_simulations=1000, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    # All trades are winners, every resampling must be profitable
    assert mc['statistics']['prob_profit'] == 1.0


def test_all_losing_trades():
    results = make_results([-100, -200, -300, -150, -250])
    config = MonteCarloConfig(n_simulations=1000, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    assert mc['statistics']['prob_profit'] == 0.0


def test_prob_ruin_all_losers():
    """Large losses should trigger ruin (equity drops below 50% of initial)."""
    results = make_results([-2000, -2000, -2000, -2000, -2000], initial_capital=10000)
    config = MonteCarloConfig(n_simulations=1000, random_seed=42, ruin_threshold=0.5)
    mc = MonteCarloSimulator(results, config).run()

    # Every sim loses $10k from $10k starting, equity goes to 0 or below
    assert mc['statistics']['prob_ruin'] == 1.0


def test_prob_ruin_no_ruin():
    """Small gains should never trigger ruin."""
    results = make_results([10, 20, 30, 10, 20], initial_capital=10000)
    config = MonteCarloConfig(n_simulations=1000, random_seed=42, ruin_threshold=0.5)
    mc = MonteCarloSimulator(results, config).run()

    assert mc['statistics']['prob_ruin'] == 0.0


# ── PERCENTILE ORDERING ──────────────────────────────────

def test_percentile_ordering():
    results = make_results([100, -50, 200, -30, 150, -80, 300, -120])
    config = MonteCarloConfig(n_simulations=1000, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    stats = mc['statistics']
    assert stats['p5_final_equity'] <= stats['p25_final_equity']
    assert stats['p25_final_equity'] <= stats['p50_final_equity']
    assert stats['p50_final_equity'] <= stats['p75_final_equity']
    assert stats['p75_final_equity'] <= stats['p95_final_equity']


# ── EQUITY CURVE SHAPE ───────────────────────────────────

def test_equity_curves_start_at_initial_capital():
    results = make_results([100, -50, 200], initial_capital=5000)
    config = MonteCarloConfig(n_simulations=100, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    # All sims should start at initial capital
    np.testing.assert_array_equal(mc['equity_curves'][:, 0], 5000)


def test_max_drawdown_is_negative():
    """Max drawdown should be <= 0 for any path with a loss."""
    results = make_results([100, -200, 50, -100, 300])
    config = MonteCarloConfig(n_simulations=500, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    assert np.all(mc['max_drawdowns'] <= 0)


def test_zero_drawdown_all_gains():
    """If every trade is a gain, drawdown should be 0."""
    results = make_results([100, 100, 100, 100, 100])
    config = MonteCarloConfig(n_simulations=500, random_seed=42)
    mc = MonteCarloSimulator(results, config).run()

    np.testing.assert_array_equal(mc['max_drawdowns'], 0.0)
