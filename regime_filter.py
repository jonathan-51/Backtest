import pandas as pd


class RegimeFilter:
    """
    Scores the daily market regime from 0-9 using 9 binary conditions
    across trend, rotation, credit, and volatility signals.

    A score >= threshold means the environment is favourable for long entries.
    A score < threshold suppresses buy signals entirely.
    """

    def __init__(self,
                 sma_long: int = 200,
                 sma_short: int = 50,
                 sma_ratio: int = 20,
                 vix_threshold: float = 20.0):
        self.sma_long = sma_long
        self.sma_short = sma_short
        self.sma_ratio = sma_ratio
        self.vix_threshold = vix_threshold

    def compute(self, regime_data: dict) -> pd.Series:
        """
        Compute a daily regime score (0-9) from ETF data.

        Parameters
        ----------
        regime_data : dict
            Mapping of ticker -> DataFrame with at least a 'date' and 'close' column.
            Expected tickers: SPY, QQQ, IWM, RSP, HYG, TLT, VIX, SPHB, SPLV, XLY, XLP

        Returns
        -------
        pd.Series
            Date-indexed Series of integer scores. Missing ETFs contribute 0.
        """
        if 'SPY' not in regime_data:
            return pd.Series(dtype=int)

        spy_close = self._get_close(regime_data, 'SPY')
        base_index = spy_close.index

        components = [
            self._score_sma_above(regime_data, 'SPY', self.sma_long, base_index),   # 1: SPY trend
            self._score_sma_above(regime_data, 'QQQ', self.sma_long, base_index),   # 2: QQQ trend
            self._score_ratio_trending(regime_data, 'IWM', 'SPY', base_index),      # 3: small caps leading
            self._score_ratio_trending(regime_data, 'XLY', 'XLP', base_index),      # 4: cyclicals leading
            self._score_ratio_trending(regime_data, 'SPHB', 'SPLV', base_index),    # 5: high beta leading
            self._score_ratio_trending(regime_data, 'RSP', 'SPY', base_index),      # 6: breadth
            self._score_sma_above(regime_data, 'HYG', self.sma_short, base_index),  # 7: credit healthy
            self._score_tlt_declining(regime_data, base_index),                     # 8: rates rising = risk-on
        ]  # Max score = 8 (VIX excluded: not fetchable from IB as standard security)

        score = (
            pd.concat(components, axis=1)
            .fillna(0)
            .sum(axis=1)
            .astype(int)
        )

        return score

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _get_close(self, regime_data: dict, ticker: str) -> pd.Series | None:
        if ticker not in regime_data:
            return None
        df = regime_data[ticker]
        return df.set_index('date')['close']

    def _score_sma_above(self, regime_data, ticker, period, base_index) -> pd.Series:
        """Returns 1 where close > SMA(period), 0 otherwise."""
        close = self._get_close(regime_data, ticker)
        if close is None:
            return pd.Series(0, index=base_index)
        sma = close.rolling(period).mean()
        return (close > sma).astype(int).reindex(base_index, fill_value=0)

    def _score_ratio_trending(self, regime_data, num_ticker, den_ticker, base_index) -> pd.Series:
        """Returns 1 where the ratio num/den is above its own SMA(sma_ratio), else 0."""
        num = self._get_close(regime_data, num_ticker)
        den = self._get_close(regime_data, den_ticker)
        if num is None or den is None:
            return pd.Series(0, index=base_index)
        ratio = num / den
        ratio_sma = ratio.rolling(self.sma_ratio).mean()
        return (ratio > ratio_sma).astype(int).reindex(base_index, fill_value=0)

    def _score_tlt_declining(self, regime_data, base_index) -> pd.Series:
        """Returns 1 when TLT SMA(20) < SMA(50) — rising rates = risk-on environment."""
        close = self._get_close(regime_data, 'TLT')
        if close is None:
            return pd.Series(0, index=base_index)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(self.sma_short).mean()
        return (sma20 < sma50).astype(int).reindex(base_index, fill_value=0)

    def _score_vix_low(self, regime_data, base_index) -> pd.Series:
        """Returns 1 when VIX close < threshold (default 20)."""
        close = self._get_close(regime_data, 'VIX')
        if close is None:
            return pd.Series(0, index=base_index)
        return (close < self.vix_threshold).astype(int).reindex(base_index, fill_value=0)
