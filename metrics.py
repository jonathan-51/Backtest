import logging
import pandas as pd
import numpy as np
from config import MetricsConfig

class PerformanceMetrics:
    """Computes performance and risk metrics from backtest results."""

    def __init__(self,results,metric_config:MetricsConfig):
        self.results = results
        self.metric_config = metric_config
        self.logger = logging.getLogger(__name__)

    def generate_metrics(self) -> dict:
        """Generate all performance and risk metrics from backtest results."""
        metrics = {}

        equity_curve = self.results['equity_curve']

        metrics['win_loss_ratio'] = self.win_loss_ratio()
        metrics['sharpe_ratio'] = self.sharpe_ratio(equity_curve)
        metrics['sortino_ratio'] = self.sortino_ratio(equity_curve)
        metrics['calmar_ratio'] = self.calmar_ratio(equity_curve,self.results['summary']['max_drawdown'])
        metrics['profit_factor'] = self.profit_factor()
        metrics['expectancy'] = self.expectancy()
        metrics['VaR'] = self.VaR(equity_curve)
        metrics['CVaR'] = self.CVaR(equity_curve)
        metrics['avg_drawdown_duration'] = self.drawdown_avg(equity_curve)
        metrics['ulcer_index'] = self.ulcer_index(equity_curve)
        


        return metrics
    
    def sharpe_ratio(self,equity_curve:pd.DataFrame) -> float:
        """Calculate annualized Sharpe Ratio from equity curve"""

        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        years = ((equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days)/self.metric_config.calendar_days

        # Calculate daily returs from equity curve
        daily_returns = equity_curve['equity'].pct_change().dropna()

        # Annualize volatility
        annual_volatility = daily_returns.std() * np.sqrt(self.metric_config.trading_days)

        if annual_volatility == 0:
            self.logger.warning("Zero volatility — cannot calculate Sharpe ratio")
            return 0.0

        # Annualized return (CAGR)
        annual_return = (final_equity / initial_equity) ** (1/years) - 1

        sharpe = (annual_return - self.metric_config.risk_free_rate) / annual_volatility

        return sharpe
    
    def sortino_ratio(self,equity_curve:pd.DataFrame) -> float:
        """Calculate annualized Sortino Ratio from equity curve"""

        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        years = ((equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days)/self.metric_config.calendar_days

        # Calculate daily returs from equity curve
        daily_returns = equity_curve['equity'].pct_change().dropna()

        # Calculate downside returns
        downside_returns = daily_returns[daily_returns < 0]

        # Annualize volatility
        annual_downside_volatility = downside_returns.std() * np.sqrt(self.metric_config.trading_days)

        if annual_downside_volatility == 0 or downside_returns.empty:
            self.logger.warning("Zero volatility — cannot calculate Sortino ratio")
            return 0.0

        # Annualized return (CAGR)
        annual_return = (final_equity / initial_equity) ** (1/years) - 1

        sortino = (annual_return - self.metric_config.risk_free_rate) / annual_downside_volatility

        return sortino
    
    def calmar_ratio(self,equity_curve:pd.DataFrame,max_drawdown:float) -> float:
        """CAGR divided by max drawdown — penalizes strategies with deep drawdowns."""

        if max_drawdown == 0:
            self.logger.warning("No drawdown — calmar ratio is infinite")
            return float('inf')

        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        years = ((equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days)/self.metric_config.calendar_days

        # Annualized return (CAGR)
        annual_return = (final_equity / initial_equity) ** (1/years) - 1

        Calmar = annual_return / abs(max_drawdown)

        return Calmar
    
    def profit_factor(self) -> float:
        """Gross profit divided by gross loss — above 1.0 means profitable."""

        trade_log = self.results['trade_log']

        gross_profit = sum([trade['pnl'] for trade in trade_log if trade['pnl'] > 0])
        gross_loss = abs(sum([trade['pnl'] for trade in trade_log if trade['pnl'] < 0]))

        if gross_loss == 0:
            self.logger.warning("No losing trades - profit factor is infinite")
            return float('inf')

        return gross_profit / gross_loss

    def expectancy(self) -> float:
        """Expected dollar amount per trade — weighted average of wins and losses."""

        trade_log = self.results['trade_log']
        win_rate = self.results['summary']['win_rate']
        loss_rate = 1 - win_rate

        if self.results['summary']['winning_trades'] == 0:
            self.logger.warning('No winning trades - can not calculate expectancy')
            return 0.0
        
        if self.results['summary']['losing_trades'] == 0:
            self.logger.warning('No losing trades - can not calculate expectancy')
            return 0.0

        gross_profit = sum([trade['pnl'] for trade in trade_log if trade['pnl'] > 0])
        average_win = gross_profit / self.results['summary']['winning_trades']

        gross_loss = abs(sum([trade['pnl'] for trade in trade_log if trade['pnl'] < 0]))
        average_loss = gross_loss / self.results['summary']['losing_trades']

        expectancy = (win_rate * average_win) - (loss_rate * average_loss)

        return expectancy

    def VaR(self,equity_curve:pd.DataFrame) -> float:
        """Value at Risk — worst expected daily loss at a given confidence level."""
        # 5th percentile at 95% confidence: only 5% of days are worse than this
        daily_returns = equity_curve['equity'].pct_change().dropna()

        return np.percentile(daily_returns,(1 - self.metric_config.VaR_confidence) * 100)

    def CVaR(self,equity_curve:pd.DataFrame) -> float:
        """Conditional VaR (Expected Shortfall) — average loss on days worse than VaR."""
        # Captures tail risk that VaR misses: how bad are the truly bad days
        daily_returns = equity_curve['equity'].pct_change().dropna()

        var_value = np.percentile(daily_returns,(1 - self.metric_config.VaR_confidence) * 100)

        return (daily_returns[daily_returns < var_value]).mean()
    
    def drawdown_avg(self,equity_curve:pd.DataFrame) -> float:
        """Average number of days spent in drawdown (below previous peak equity)."""
        # Counts consecutive days equity is below its running peak, then averages
        peak = equity_curve['equity'].cummax()

        in_drawdown = equity_curve['equity'] < peak

        current_duration = 0
        durations = []

        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        avg_duration = np.mean(durations) if durations else 0
        return avg_duration
    
    def ulcer_index(self,equity_curve:pd.DataFrame) -> float:
        """RMS of drawdown percentages — captures both depth and duration of pain."""
        # Root-mean-square penalizes large drawdowns disproportionately
        peak = equity_curve['equity'].cummax()

        drawdown_pct = (equity_curve['equity'] - peak) / peak

        ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))

        return ulcer_index
    
    def win_loss_ratio(self) -> float:
        """Average win / average loss — how big wins are relative to losses."""

        trade_log = self.results['trade_log']

        winners = [trade['pnl'] for trade in trade_log if trade['pnl'] > 0]
        losers = [trade['pnl'] for trade in trade_log if trade['pnl'] < 0]

        if not losers:
            self.logger.warning("No losing trades — win/loss ratio is infinite")
            return float('inf')
        if not winners:
            self.logger.warning("No winning trades - win/loss ratio is zero")
            return 0.0
        
        win_ratio = np.mean(winners)/abs(np.mean(losers))

        return win_ratio