import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from indicators import Indicators

indicator = Indicators()

# ── SMA ──────────────────────────────────────────────

def test_sma_basic():
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = indicator.sma(prices, 3)

    # SMA(3) at index 2: (10+20+30)/3 = 20
    # SMA(3) at index 3: (20+30+40)/3 = 30
    # SMA(3) at index 4: (30+40+50)/3 = 40
    assert result.iloc[2] == 20.0
    assert result.iloc[3] == 30.0
    assert result.iloc[4] == 40.0

def test_sma_leading_nans():
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = indicator.sma(prices, 3)

    # First 2 values should be NaN (not enough data for period=3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])

def test_sma_period_equals_length():
    prices = pd.Series([10.0, 20.0, 30.0])
    result = indicator.sma(prices, 3)

    # Only the last value should be valid: (10+20+30)/3 = 20
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 20.0

# ── TRUE RANGE ───────────────────────────────────────

def test_true_range_no_gap():
    """When there's no gap, TR = high - low"""
    df = pd.DataFrame({
        'high':  [110.0, 115.0],
        'low':   [100.0, 105.0],
        'close': [105.0, 110.0]
    })
    result = indicator.true_range(df)

    # Bar 1: high-low=10, |high-prev_close|=10, |low-prev_close|=0 → TR=10
    assert result.iloc[1] == 10.0

def test_true_range_gap_up():
    """Gap up: previous close below current low, TR = high - prev_close"""
    df = pd.DataFrame({
        'high':  [50.0, 120.0],
        'low':   [40.0, 110.0],
        'close': [45.0, 115.0]
    })
    result = indicator.true_range(df)

    # Bar 1: high-low=10, |high-prev_close|=|120-45|=75, |low-prev_close|=|110-45|=65 → TR=75
    assert result.iloc[1] == 75.0

def test_true_range_gap_down():
    """Gap down: previous close above current high, TR = prev_close - low"""
    df = pd.DataFrame({
        'high':  [150.0, 60.0],
        'low':   [140.0, 50.0],
        'close': [145.0, 55.0]
    })
    result = indicator.true_range(df)

    # Bar 1: high-low=10, |high-prev_close|=|60-145|=85, |low-prev_close|=|50-145|=95 → TR=95
    assert result.iloc[1] == 95.0

# ── ATR ──────────────────────────────────────────────

def test_atr_basic():
    """ATR = rolling mean of True Range"""
    df = pd.DataFrame({
        'high':  [12.0, 13.0, 14.0, 15.0, 16.0],
        'low':   [10.0, 11.0, 12.0, 13.0, 14.0],
        'close': [11.0, 12.0, 13.0, 14.0, 15.0]
    })
    result = indicator.atr(df, period=3)

    # No gaps, so each TR = high - low = 2.0
    # ATR(3) at index 3 = mean(2, 2, 2) = 2.0
    assert result.iloc[3] == 2.0

def test_atr_leading_nans():
    """ATR needs period bars of TR, and TR needs 1 bar for prev_close"""
    df = pd.DataFrame({
        'high':  [12.0, 13.0, 14.0, 15.0, 16.0],
        'low':   [10.0, 11.0, 12.0, 13.0, 14.0],
        'close': [11.0, 12.0, 13.0, 14.0, 15.0]
    })
    result = indicator.atr(df, period=3)

    # Index 0: TR is NaN (no prev_close), so ATR NaN
    # Index 1,2: not enough TRs yet for period=3
    # Index 3: first valid ATR
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert not pd.isna(result.iloc[2])
    assert not pd.isna(result.iloc[3])

# ── RSI ──────────────────────────────────────────────

def test_rsi_all_gains():
    """Straight up prices → RSI near 100"""
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    result = indicator.rsi(prices, period=5)

    # All moves are positive → avg_loss ≈ 0 → RSI → 100
    assert result.iloc[-1] > 99.0

def test_rsi_all_losses():
    """Straight down prices → RSI near 0"""
    prices = pd.Series([20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
    result = indicator.rsi(prices, period=5)

    # All moves are negative → avg_gain ≈ 0 → RSI → 0
    assert result.iloc[-1] < 1.0

def test_rsi_range():
    """RSI should always be between 0 and 100"""
    prices = pd.Series([44.0, 44.3, 44.1, 43.6, 44.3, 44.8, 45.1, 43.9, 44.2, 44.5,
                        43.8, 44.6, 45.2, 44.0, 43.5])
    result = indicator.rsi(prices, period=5).dropna()

    assert (result >= 0).all()
    assert (result <= 100).all()

def test_rsi_leading_nans():
    """RSI needs `period` bars before producing values"""
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = indicator.rsi(prices, period=5)

    # First few values should be NaN
    assert pd.isna(result.iloc[0])