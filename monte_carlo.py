import numpy as np
import logging
from config import MonteCarloConfig

class MonteCarloSimulator:
    """Bootstrap resampling of trade PnLs to estimate outcome distributions.

    Takes the trade log from a backtest, resamples trades with replacement
    to generate thousands of synthetic equity paths, then computes percentile
    statistics for final equity, max drawdown, and probability of ruin.
    """

    def __init__(self, results: dict, config: MonteCarloConfig):
        self.trade_pnls = np.array([t['pnl'] for t in results['trade_log']])
        self.initial_capital = results['equity_curve']['equity'].iloc[0]
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self) -> dict:
        """Run Monte Carlo simulation and return distribution statistics.

        Returns:
            dict with keys:
                equity_curves: np.ndarray (n_sims, n_trades+1) — all simulated paths
                final_equities: np.ndarray (n_sims,)
                max_drawdowns: np.ndarray (n_sims,) — as negative fractions
                statistics: dict of percentile summaries
        """
        n_trades = len(self.trade_pnls)
        n_sims = self.config.n_simulations
        rng = np.random.default_rng(self.config.random_seed)

        self.logger.info(f"Running {n_sims} Monte Carlo simulations on {n_trades} trades")

        # Resample trade PnLs: (n_sims, n_trades)
        indices = rng.integers(0, n_trades, size=(n_sims, n_trades))
        resampled_pnls = self.trade_pnls[indices]

        # Build equity curves: prepend initial capital, then cumsum PnLs
        cumulative_pnls = np.cumsum(resampled_pnls, axis=1)
        equity_curves = np.column_stack([
            np.full(n_sims, self.initial_capital),
            self.initial_capital + cumulative_pnls
        ])

        final_equities = equity_curves[:, -1]
        max_drawdowns = self._compute_max_drawdowns(equity_curves)

        # Probability metrics
        prob_profit = np.mean(final_equities > self.initial_capital)
        ruin_level = self.initial_capital * self.ruin_threshold
        prob_ruin = np.mean(np.min(equity_curves, axis=1) <= ruin_level)

        statistics = {
            'median_final_equity': float(np.median(final_equities)),
            'mean_final_equity': float(np.mean(final_equities)),
            'prob_profit': float(prob_profit),
            'prob_ruin': float(prob_ruin),
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'p5_max_drawdown': float(np.percentile(max_drawdowns, 5)),
        }

        # Add final equity percentiles
        for p in self.config.percentiles:
            statistics[f'p{p}_final_equity'] = float(np.percentile(final_equities, p))

        self.logger.info(f"MC complete — median final equity: ${statistics['median_final_equity']:,.0f}, "
                         f"prob profit: {prob_profit:.1%}, prob ruin: {prob_ruin:.1%}")

        return {
            'equity_curves': equity_curves,
            'final_equities': final_equities,
            'max_drawdowns': max_drawdowns,
            'statistics': statistics,
        }

    @property
    def ruin_threshold(self) -> float:
        return self.config.ruin_threshold

    def _compute_max_drawdowns(self, equity_curves: np.ndarray) -> np.ndarray:
        """Compute max drawdown for each simulated equity path.

        Returns:
            np.ndarray of shape (n_sims,) with max drawdowns as negative fractions.
        """
        running_peak = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (equity_curves - running_peak) / running_peak
        return np.min(drawdowns, axis=1)
