import pandas as pd
import sys
sys.path.insert(0, '.')
from strategies.atr_channel_breakout import ATRChannelBreakout


def make_price_df(closes, start='2020-01-01'):
    """Build a minimal OHLCV DataFrame with enough bars to generate a crossover signal."""
    dates = pd.date_range(start=start, periods=len(closes), freq='B')
    return pd.DataFrame({
        'date': dates,
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low':  [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000] * len(closes),
    })


def make_spy_df(closes, start='2020-01-01'):
    """Build a SPY DataFrame with the same date range."""
    dates = pd.date_range(start=start, periods=len(closes), freq='B')
    return pd.DataFrame({
        'date': dates,
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low':  [c * 0.99 for c in closes],
        'close': closes,
        'volume': [50_000_000] * len(closes),
    })


def make_strategy(spy_sma_length=5):
    """Short-param strategy so signals appear in small test DataFrames."""
    return ATRChannelBreakout(
        sma_length=3,
        atr_length=3,
        envelope_mult=0.5,
        stop_mult=1,
        trail_mult=1,
        spy_sma_length=spy_sma_length,
    )


# ── FILTER DISABLED ────────────────────────────────────

def test_regime_filter_disabled_when_spy_not_in_data():
    """When spy_1d is absent, the filter is skipped and signals pass through unchanged."""
    closes = [100] * 10 + [120]   # price jumps above envelope on last bar
    df = make_price_df(closes)
    strategy = make_strategy(spy_sma_length=5)

    signals, _ = strategy.generate_signals({'1d': df})

    # Strategy ran without error; signals are present (some 'buy' or 'hold_cash')
    assert set(signals['signal'].unique()).issubset({'buy', 'hold_cash'})


def test_regime_filter_disabled_when_spy_sma_length_zero():
    """spy_sma_length=0 disables the regime filter entirely regardless of SPY data."""
    closes = [100] * 10 + [120]
    df = make_price_df(closes)
    spy = make_spy_df([50] * 11)   # SPY far below any SMA — would block trades if filter active

    strategy = make_strategy(spy_sma_length=0)

    signals_no_spy, _ = strategy.generate_signals({'1d': df})
    signals_with_spy, _ = strategy.generate_signals({'1d': df, 'spy_1d': spy})

    # Both runs should produce the same signals (filter is off)
    pd.testing.assert_series_equal(
        signals_no_spy['signal'].reset_index(drop=True),
        signals_with_spy['signal'].reset_index(drop=True),
    )


# ── BULL REGIME (allow buys) ───────────────────────────

def test_regime_filter_allows_buy_in_bull_market():
    """When SPY close > SMA, buy signals should pass through."""
    # Stock: big jump to trigger a breakout signal
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 150]
    df = make_price_df(closes)

    # SPY: steadily rising — close always above SMA(5)
    spy_closes = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
    spy = make_spy_df(spy_closes)

    strategy = make_strategy(spy_sma_length=5)
    signals, _ = strategy.generate_signals({'1d': df, 'spy_1d': spy})

    # At least one buy signal should exist (not all filtered out)
    assert 'buy' in signals['signal'].values


# ── BEAR REGIME (suppress buys) ───────────────────────

def test_regime_filter_suppresses_buy_in_bear_market():
    """When SPY close < SMA, buy signals are suppressed to hold_cash."""
    # Stock: big jump to trigger a breakout signal
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 150]
    df = make_price_df(closes)

    # SPY: plummeting — close always below SMA(5)
    spy_closes = [390, 380, 370, 360, 350, 100, 90, 80, 70, 60]
    spy = make_spy_df(spy_closes)

    strategy = make_strategy(spy_sma_length=5)
    signals, orders = strategy.generate_signals({'1d': df, 'spy_1d': spy})

    # No buy signals should survive the regime filter
    assert 'buy' not in signals['signal'].values
    # No orders should be attached to suppressed signals
    assert len(orders) == 0


def test_regime_filter_removes_associated_order():
    """When a buy signal is suppressed, its Order is also removed from the orders dict."""
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 150]
    df = make_price_df(closes)

    # Bear regime SPY
    spy_closes = [390, 380, 370, 360, 350, 100, 90, 80, 70, 60]
    spy = make_spy_df(spy_closes)

    strategy = make_strategy(spy_sma_length=5)
    _, orders = strategy.generate_signals({'1d': df, 'spy_1d': spy})

    assert len(orders) == 0
