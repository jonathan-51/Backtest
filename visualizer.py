import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional

class BacktestVisualizer:
    """Generates visual reports from backtest results and performance metrics."""

    def __init__(self,results:Dict[str,any],metrics:Dict[str,float],mc_results:Optional[dict]=None):
        self.results = results
        self.metrics = metrics
        self.mc_results = mc_results
        self.logger = logging.getLogger(__name__)

    def generate_report(self) -> None:
        """Calls all plots, arranges into a single figure with subplots"""

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(12,6))

        ax1 = self.plot_equity_curve(ax1)
        ax2 = self.plot_drawdowns(ax2)
        ax3 = self.plot_returns_distribution(ax3)
        if self.mc_results is not None:
            ax4 = self.plot_monte_carlo(ax4)

        plt.tight_layout()
        plt.show()
    
    def plot_equity_curve(self,ax):
        """Line chart of equity over time"""

        equity_curve = self.results['equity_curve']

        ax.plot(equity_curve['date'],equity_curve['equity'])

        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')

        return ax
        

    def plot_drawdowns(self,ax):
        """Shaded area chart showing underwater periods"""

        equity_curve = self.results['equity_curve']
        peak = equity_curve['equity'].cummax()

        drawdown = ((equity_curve['equity'] - peak) / peak) * 100

        ax.fill_between(equity_curve['date'], drawdown, 0, color='red', alpha=0.3)

        ax.set_title('Drawdown Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')

        return ax

    def plot_returns_distribution(self,ax):
        """Historgram of daily returns with normal dist overlay"""
        
        equity_curve = self.results['equity_curve']

        daily_returns = equity_curve['equity'].pct_change().dropna()

        ax.hist(daily_returns,bins=50)

        ax.set_title(f'Daily Return\'s Curve')
        ax.set_xlabel('Daily Returns (%)')
        ax.set_ylabel('Frequency')

    def plot_monte_carlo(self, ax):
        """Fan chart showing percentile bands of simulated equity paths."""
        curves = self.mc_results['equity_curves']
        trade_numbers = np.arange(curves.shape[1])

        p5 = np.percentile(curves, 5, axis=0)
        p25 = np.percentile(curves, 25, axis=0)
        p50 = np.percentile(curves, 50, axis=0)
        p75 = np.percentile(curves, 75, axis=0)
        p95 = np.percentile(curves, 95, axis=0)

        ax.fill_between(trade_numbers, p5, p95, alpha=0.15, color='blue', label='5th–95th')
        ax.fill_between(trade_numbers, p25, p75, alpha=0.3, color='blue', label='25th–75th')
        ax.plot(trade_numbers, p50, color='blue', linewidth=1.5, label='Median')

        stats = self.mc_results['statistics']
        ax.set_title(f'Monte Carlo ({curves.shape[0]} sims) — '
                     f'P(profit): {stats["prob_profit"]:.0%}')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Equity ($)')
        ax.legend(fontsize=8)

        return ax
