"""
AutoML Hyperparameter Optimization

Automated machine learning with Bayesian optimization using Optuna.

Author: Gabriel Demetrios Lafis
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Callable
from loguru import logger


class AutoMLOptimizer:
    """
    Automated ML with hyperparameter optimization.
    """
    
    def __init__(
        self,
        task: str = 'classification',
        metric: str = 'accuracy',
        n_trials: int = 100,
        timeout: int = None
    ):
        """
        Initialize AutoML optimizer.
        
        Args:
            task: Task type ('classification' or 'regression')
            metric: Optimization metric
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.task = task
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_model = None
        self.best_params = None
        self.study = None
        
        logger.info(f"Initialized AutoML for {task} with {n_trials} trials")
    
    def _objective_xgboost(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Objective function for XGBoost."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'random_state': 42
        }
        
        if self.task == 'classification':
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring=self.metric,
            n_jobs=-1
        )
        
        return scores.mean()
    
    def _objective_lightgbm(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Objective function for LightGBM."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42,
            'verbose': -1
        }
        
        if self.task == 'classification':
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring=self.metric,
            n_jobs=-1
        )
        
        return scores.mean()
    
    def _objective_random_forest(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Objective function for Random Forest."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        if self.task == 'classification':
            model = RandomForestClassifier(**params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring=self.metric,
            n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'xgboost'
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training labels
            model_type: Model type ('xgboost', 'lightgbm', 'random_forest')
            
        Returns:
            Dictionary with best parameters and score
        """
        logger.info(f"Starting optimization for {model_type}...")
        
        # Select objective function
        if model_type == 'xgboost':
            objective = lambda trial: self._objective_xgboost(trial, X, y)
        elif model_type == 'lightgbm':
            objective = lambda trial: self._objective_lightgbm(trial, X, y)
        elif model_type == 'random_forest':
            objective = lambda trial: self._objective_random_forest(trial, X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=f'automl_{model_type}'
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        best_score = self.study.best_value
        
        logger.success(f"Optimization complete. Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Train final model with best parameters
        if model_type == 'xgboost':
            if self.task == 'classification':
                self.best_model = xgb.XGBClassifier(**self.best_params, random_state=42)
            else:
                self.best_model = xgb.XGBRegressor(**self.best_params, random_state=42)
        elif model_type == 'lightgbm':
            if self.task == 'classification':
                self.best_model = lgb.LGBMClassifier(**self.best_params, random_state=42, verbose=-1)
            else:
                self.best_model = lgb.LGBMRegressor(**self.best_params, random_state=42, verbose=-1)
        elif model_type == 'random_forest':
            if self.task == 'classification':
                self.best_model = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.best_model = RandomForestRegressor(**self.best_params, random_state=42, n_jobs=-1)
        
        self.best_model.fit(X, y)
        
        return {
            'best_params': self.best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No optimization has been run yet")
        
        df = self.study.trials_dataframe()
        return df
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No optimization has been run yet")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0])
        
        # Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[1])
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Run AutoML
    automl = AutoMLOptimizer(
        task='classification',
        metric='accuracy',
        n_trials=50
    )
    
    results = automl.optimize(X_train, y_train, model_type='xgboost')
    
    print("\nAutoML Results:")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Parameters: {results['best_params']}")
    
    # Evaluate on test set
    from sklearn.metrics import accuracy_score
    y_pred = automl.best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
