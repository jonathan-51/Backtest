import pandas as pd
import numpy as np
from config import MetricsConfig

class PerformanceMetrics:
    """Computes performance and risk metrics"""

    def __init__(self,results,metric_config:MetricsConfig):
        self.results = results
        self.config =  metric_config
        pass

    def generate_metrics(self):
        """Generate all performance and risk metrics from backtest results."""
        metrics = {}

        equity_curve = self.results['equity_curve']

        metrics['sharpe_ratio'] = self.sharpe_ratio(equity_curve)
        metrics['max_drawdown'] = self.max_drawdown(equity_curve)

        return metrics
    
    def sharpe_ratio(self,equity_curve):
        """Calculate Sharpe Ratio from equity curve"""

        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # Annualized mean return
        annualized_return = daily_returns.mean() * self.config.trading_days

        # Annualized volatility
        annualized_volatility = daily_returns.std() * np.sqrt(self.config.trading_days)

        return (annualized_return - self.config.risk_free_rate) / annualized_volatility

    def max_drawdown(self,equity_curve):

        # Running peak at each bar
        peak = equity_curve['equity'].cummax()

        # Drawdown at each bar
        drawdown = (equity_curve['equity'] - peak) / peak

        return drawdown.min()