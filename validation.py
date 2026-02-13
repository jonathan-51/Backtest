import pandas as pd
from typing import List,Tuple
from backtest import BacktestEngine
from config import BacktestConfig,MetricsConfig,WalkForwardValidatorConfig
from metrics import PerformanceMetrics
class WalkForwardValidator:
    """Walk-forward validation to test strategy edge on unseen data."""
    def __init__(self,strategy,data:dict):
        self.walk_forward_validator_config = WalkForwardValidatorConfig
        self.strategy = strategy
        self.data = data
        self.train_bars = self.walk_forward_validator_config.train_bars
        self.test_bars = self.walk_forward_validator_config.test_bars
        self.warm_up_bars = self.walk_forward_validator_config.warm_up_bars

    def create_windows(self) -> List[Tuple[pd.DataFrame,pd.DataFrame]]:
        """Split data into rolling train/test windows"""
        df = self.data['1d']

        windows=[]
        start = 0
        # Creating windows
        while start + self.train_bars + self.test_bars <= len(df):
            train = df.iloc[start:start+self.train_bars].copy()
            test = df.iloc[start+self.train_bars-self.warm_up_bars:start+self.train_bars+self.test_bars].copy()
            windows.append((train,test))
            start += self.test_bars


        # Handle last window if remaining data is smaller than test_bars
        remaining = len(df) - (start + self.train_bars)
        if remaining > self.walk_forward_validator_config.min_remaining_bars:
            train = df.iloc[start:start + self.train_bars].copy()
            test = df.iloc[start+self.train_bars-self.warm_up_bars:].copy()
            windows.append((train,test))

        return windows
    
    def run(self) -> List[dict]:
        """Run walk-forward validation across all windows"""
        windows = self.create_windows()
        results = []
        for i,(train_df,test_df) in enumerate(windows):
            # Fresh engine for train
            train_engine = BacktestEngine(BacktestConfig,self.strategy)
            train_results = train_engine.run({'1d':train_df})
            train_metrics = PerformanceMetrics(train_results,MetricsConfig()).generate_metrics()

            # Fresh engine for test
            test_engine = BacktestEngine(BacktestConfig,self.strategy)
            test_results = test_engine.run({'1d':test_df})
            test_metrics = PerformanceMetrics(test_results,MetricsConfig()).generate_metrics()

            # Compute degradation
            train_sharpe = train_metrics['sharpe_ratio']
            test_sharpe = test_metrics['sharpe_ratio']

            if abs(train_sharpe) <= 0.1:
                degradation = 0
            elif train_sharpe != 0:
                degradation = (test_sharpe - train_sharpe) / abs(train_sharpe)
            else:
                degradation = 0

            results.append({
                'window':i+1,
                'train_bars':len(train_df),
                'test_bars':len(test_df),
                'train_trades':train_results['summary']['total_trades'],
                'test_trades':test_results['summary']['total_trades'],
                'train_sharpe':train_sharpe,
                'test_sharpe':test_sharpe,
                'degradation':degradation
            })

        return self.summarize(results)
    
    def summarize(self,results: list[dict]) -> dict:
        """Compute aggregate walk-forward stats"""

        avg_train_sharpe = sum(result['train_sharpe'] for result in results) / len(results)
        avg_test_sharpe = sum(result['test_sharpe'] for result in results) / len(results)
        avg_degradation = sum(result['degradation'] for result in results) / len(results)

        # Return validation result for optimized parameter based on optimized parameter threshold
        if avg_degradation > self.walk_forward_validator_config.avg_degradation_threshold_pass:
            verdict = "PASS"
        elif avg_degradation > self.walk_forward_validator_config.avg_degradation_threshold_marginal:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        return {
            'windows':results,
            'avg_train_sharpe':avg_train_sharpe,
            'avg_test_sharpe':avg_test_sharpe,
            'avg_degradation':avg_degradation,
            'verdict':verdict
        }