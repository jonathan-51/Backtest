import pandas as pd
from typing import List,Tuple,Dict
from backtest import BacktestEngine
from config import BacktestConfig,MetricsConfig,WalkForwardValidatorConfig
from metrics import PerformanceMetrics
import logging
from itertools import product

class WalkForwardValidator:
    """Walk-forward validation to test strategy edge on unseen data."""
    def __init__(self,strategy,data:Dict[str,pd.DataFrame]):
        self.walk_forward_validator_config = WalkForwardValidatorConfig
        self.strategy = strategy
        self.data = data
        self.train_bars = self.walk_forward_validator_config.train_bars
        self.test_bars = self.walk_forward_validator_config.test_bars
        self.warm_up_bars = self.walk_forward_validator_config.warm_up_bars
        self.logger = logging.getLogger(__name__)

    def create_windows(self) -> List[Tuple[pd.DataFrame,pd.DataFrame]]:
        """Split data into rolling train/test windows"""
        df = self.data[WalkForwardValidatorConfig.timeframe]

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
        self.logger.info(f"{self.strategy.__class__.__name__} -> Running walk-forward validation: {len(windows)} windows")

        results = []
        for i,(train_df,test_df) in enumerate(windows):
            self.logger.info(f"Window {i+1}/{len(windows)}: train={len(train_df)} bars, test={len(test_df)} bars")

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

        self.logger.info(f"Window {i+1}: train_sharpe={train_sharpe:.4f}, test_sharpe={test_sharpe:.4f}, degradation={degradation:.2%}")
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

        self.logger.info(f"Walk-forward complete for {self.strategy.__class__.__name__}: avg_degradation={avg_degradation:.2%}, verdict={verdict}")

        return {
            'windows':results,
            'avg_train_sharpe':avg_train_sharpe,
            'avg_test_sharpe':avg_test_sharpe,
            'avg_degradation':avg_degradation,
            'verdict':verdict
        }
    

class WalkForwardOptimizer:
    """Sweeps aprameter combinations per train window, tests best on unseen data"""
    def __init__(self,strategy_class,param_grid:dict,data:Dict[str,pd.DataFrame]):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.config = WalkForwardValidatorConfig
        self.logger = logging.getLogger(__name__)

    def _generate_combos(self) -> list[dict]:
        """Generate all parameter combinations from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys,combo)) for combo in product(*values)]
    
    def _run_single(self, params: dict, data: dict) -> float:
        """Run a single backtest with given params, return Sharpe ratio."""
        strategy = self.strategy_class(**params)
        engine = BacktestEngine(BacktestConfig(), strategy)
        results = engine.run(data)

        if results['summary']['total_trades'] == 0:
            return float('-inf')

        metrics = PerformanceMetrics(results, MetricsConfig()).generate_metrics()
        return metrics['sharpe_ratio']
        
    def create_windows(self):
        """Split data into rolling train/test windows (same logic as validator)."""
        df = self.data[self.config.timeframe]
        windows = []
        start = 0

        while start + self.config.train_bars + self.config.test_bars <= len(df):
            train = df.iloc[start:start + self.config.train_bars].copy()
            test = df.iloc[start + self.config.train_bars - self.config.warm_up_bars:
                           start + self.config.train_bars + self.config.test_bars].copy()
            windows.append((train,test))
            start += self.config.test_bars

        remaining = len(df) - (start + self.config.train_bars)
        if remaining > self.config.min_remaining_bars:
            train = df.iloc[start:start + self.config.train_bars].copy()
            test = df.iloc[start + self.config.train_bars - self.config.warm_up_bars:].copy()
            windows.append((train, test))

        return windows
    
    def run(self) -> dict:
        """Run walk-forward optimization across all windows."""
        windows = self.create_windows()
        combos = self._generate_combos()
        self.logger.info(f"Walk-forward optimization: {len(windows)} windows, {len(combos)} param combos each")

        results = []
        for i, (train_df, test_df) in enumerate(windows):
            self.logger.info(f"Window {i+1}/{len(windows)}: sweeping {len(combos)} combos on {len(train_df)} train bars")

            # Sweep all combos on train data
            best_sharpe = float('-inf')
            best_params = None

            for params in combos:
                sharpe = self._run_single(params, {self.config.timeframe: train_df})
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params

            # Test best params on unseen data
            test_sharpe = self._run_single(best_params, {self.config.timeframe: test_df})

            # Compute degradation
            if abs(best_sharpe) <= 0.1:
                degradation = 0
            elif best_sharpe != 0:
                degradation = (test_sharpe - best_sharpe) / abs(best_sharpe)
            else:
                degradation = 0

            self.logger.info(f"Window {i+1}: best_params={best_params}, "
                           f"train_sharpe={best_sharpe:.4f}, test_sharpe={test_sharpe:.4f}, "
                           f"degradation={degradation:.2%}")
            
            results.append({
                'window': i + 1,
                'best_params': best_params,
                'train_bars': len(train_df),
                'test_bars': len(test_df),
                'train_sharpe': best_sharpe,
                'test_sharpe': test_sharpe,
                'degradation': degradation,
            })

        return self.summarize(results)
    
    def summarize(self, results: list[dict]) -> dict:
        """Compute aggregate optimization stats."""
        avg_train_sharpe = sum(r['train_sharpe'] for r in results) / len(results)
        avg_test_sharpe = sum(r['test_sharpe'] for r in results) / len(results)
        avg_degradation = sum(r['degradation'] for r in results) / len(results)

        if avg_degradation > self.config.avg_degradation_threshold_pass:
            verdict = "PASS"
        elif avg_degradation > self.config.avg_degradation_threshold_marginal:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        self.logger.info(f"Optimization complete: avg_degradation={avg_degradation:.2%}, verdict={verdict}")

        return {
            'windows': results,
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'avg_degradation': avg_degradation,
            'verdict': verdict,
        }