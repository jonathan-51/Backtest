import matplotlib.pyplot as plt
import logging
from typing import Dict

class BacktestVisualizer:

    def __init__(self,results:Dict[str,any],metrics:Dict[str,float]):
        self.results = results
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)

    def generate_report(self) -> None:
        """Calls all plots, arranges into a single figure with subplots"""

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(12,6))

        ax1 = self.plot_equity_curve(ax1)
        ax2 = self.plot_drawdowns(ax2)
        ax3 = self.plot_returns_distribution(ax3)

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

    