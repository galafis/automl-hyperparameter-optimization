"""
AutoML Hyperparameter Optimization Module

Grid search, random search, Bayesian optimization (Gaussian Process surrogate),
Hyperband, early stopping, parameter space definition, and experiment tracking.

Author: Gabriel Demetrios Lafis
"""

import math
import random
import time
from itertools import product as iter_product
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Parameter Space ───────────────────────────────────────────────────

class ParameterSpace:
    """Define a hyperparameter search space."""

    def __init__(self):
        self._params: Dict[str, Dict[str, Any]] = {}

    def add_float(self, name: str, low: float, high: float, log_scale: bool = False) -> "ParameterSpace":
        self._params[name] = {"type": "float", "low": low, "high": high, "log_scale": log_scale}
        return self

    def add_int(self, name: str, low: int, high: int) -> "ParameterSpace":
        self._params[name] = {"type": "int", "low": low, "high": high}
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> "ParameterSpace":
        self._params[name] = {"type": "categorical", "choices": choices}
        return self

    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random configuration from the space."""
        if rng is None:
            rng = random.Random()
        config: Dict[str, Any] = {}
        for name, spec in self._params.items():
            if spec["type"] == "float":
                if spec.get("log_scale"):
                    log_low = math.log(spec["low"])
                    log_high = math.log(spec["high"])
                    config[name] = math.exp(rng.uniform(log_low, log_high))
                else:
                    config[name] = rng.uniform(spec["low"], spec["high"])
            elif spec["type"] == "int":
                config[name] = rng.randint(spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                config[name] = rng.choice(spec["choices"])
        return config

    def grid(self, n_points: int = 5) -> List[Dict[str, Any]]:
        """Generate a grid of configurations (n_points per continuous param)."""
        axes = {}
        for name, spec in self._params.items():
            if spec["type"] == "float":
                if spec.get("log_scale"):
                    log_low = math.log(spec["low"])
                    log_high = math.log(spec["high"])
                    axes[name] = [math.exp(log_low + i * (log_high - log_low) / (n_points - 1)) for i in range(n_points)]
                else:
                    axes[name] = [spec["low"] + i * (spec["high"] - spec["low"]) / (n_points - 1) for i in range(n_points)]
            elif spec["type"] == "int":
                vals = list(range(spec["low"], spec["high"] + 1))
                step = max(1, len(vals) // n_points)
                axes[name] = vals[::step][:n_points]
            elif spec["type"] == "categorical":
                axes[name] = spec["choices"]

        keys = list(axes.keys())
        return [dict(zip(keys, combo)) for combo in iter_product(*(axes[k] for k in keys))]

    @property
    def param_names(self) -> List[str]:
        return list(self._params.keys())


# ── Experiment Tracker ────────────────────────────────────────────────

class ExperimentTracker:
    """Track optimization trials and results."""

    def __init__(self):
        self._trials: List[Dict[str, Any]] = []

    def log_trial(self, params: Dict[str, Any], score: float, duration: float = 0.0, extra: Dict = None) -> int:
        trial_id = len(self._trials)
        entry = {
            "trial_id": trial_id,
            "params": dict(params),
            "score": score,
            "duration": duration,
            "extra": extra or {},
        }
        self._trials.append(entry)
        return trial_id

    @property
    def best_trial(self) -> Optional[Dict[str, Any]]:
        if not self._trials:
            return None
        return max(self._trials, key=lambda t: t["score"])

    @property
    def trials(self) -> List[Dict[str, Any]]:
        return list(self._trials)

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    def scores(self) -> List[float]:
        return [t["score"] for t in self._trials]

    def best_score_over_time(self) -> List[float]:
        """Running maximum of scores."""
        bests = []
        current_best = float("-inf")
        for t in self._trials:
            current_best = max(current_best, t["score"])
            bests.append(current_best)
        return bests


# ── Early Stopping ────────────────────────────────────────────────────

class EarlyStopping:
    """Stop optimization when no improvement is seen for *patience* trials."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("-inf")
        self._counter = 0

    def should_stop(self, score: float) -> bool:
        if score > self._best + self.min_delta:
            self._best = score
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience

    def reset(self):
        self._best = float("-inf")
        self._counter = 0


# ── Grid Search ───────────────────────────────────────────────────────

class GridSearchOptimizer:
    """Exhaustive grid search over parameter space."""

    def __init__(self, space: ParameterSpace, n_points: int = 5):
        self.space = space
        self.n_points = n_points
        self.tracker = ExperimentTracker()

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        grid = self.space.grid(self.n_points)
        for config in grid:
            t0 = time.time()
            score = objective(config)
            self.tracker.log_trial(config, score, time.time() - t0)
        return self.tracker.best_trial


# ── Random Search ─────────────────────────────────────────────────────

class RandomSearchOptimizer:
    """Random sampling search."""

    def __init__(self, space: ParameterSpace, n_trials: int = 100, seed: int = 42):
        self.space = space
        self.n_trials = n_trials
        self.rng = random.Random(seed)
        self.tracker = ExperimentTracker()
        self.early_stopping: Optional[EarlyStopping] = None

    def set_early_stopping(self, patience: int = 10) -> None:
        self.early_stopping = EarlyStopping(patience=patience)

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        for _ in range(self.n_trials):
            config = self.space.sample(self.rng)
            t0 = time.time()
            score = objective(config)
            self.tracker.log_trial(config, score, time.time() - t0)
            if self.early_stopping and self.early_stopping.should_stop(score):
                break
        return self.tracker.best_trial


# ── Bayesian Optimization (GP surrogate) ──────────────────────────────

class GaussianProcessSurrogate:
    """Minimal GP surrogate using RBF kernel for Bayesian optimization."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self._X: List[List[float]] = []
        self._y: List[float] = []

    def _rbf_kernel(self, x1: List[float], x2: List[float]) -> float:
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-sq_dist / (2 * self.length_scale ** 2))

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self._X = [list(x) for x in X]
        self._y = list(y)

    def predict(self, x: List[float]) -> Tuple[float, float]:
        """Return (mean, std) prediction at point x."""
        if not self._X:
            return 0.0, 1.0
        n = len(self._X)
        k_star = [self._rbf_kernel(x, xi) for xi in self._X]
        K = [[self._rbf_kernel(self._X[i], self._X[j]) + (self.noise if i == j else 0)
              for j in range(n)] for i in range(n)]
        # Simple approximation: weighted average
        k_sum = sum(k_star) + 1e-10
        mean = sum(k * y for k, y in zip(k_star, self._y)) / k_sum
        var = max(0, 1 - sum(k ** 2 for k in k_star) / (k_sum + 1e-10))
        return mean, math.sqrt(var)


class BayesianOptimizer:
    """Bayesian optimization with GP surrogate and Expected Improvement."""

    def __init__(self, space: ParameterSpace, n_trials: int = 50, n_initial: int = 5, seed: int = 42):
        self.space = space
        self.n_trials = n_trials
        self.n_initial = n_initial
        self.rng = random.Random(seed)
        self.tracker = ExperimentTracker()
        self.gp = GaussianProcessSurrogate()
        self.early_stopping: Optional[EarlyStopping] = None

    def set_early_stopping(self, patience: int = 10) -> None:
        self.early_stopping = EarlyStopping(patience=patience)

    def _config_to_vector(self, config: Dict[str, Any]) -> List[float]:
        return [float(config.get(n, 0)) for n in self.space.param_names]

    def _expected_improvement(self, mean: float, std: float, best: float) -> float:
        if std <= 0:
            return 0.0
        z = (mean - best) / std
        # Approximate CDF/PDF for EI
        ei = std * (z * (0.5 + 0.5 * math.erf(z / math.sqrt(2))) +
                     math.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi))
        return ei

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        # Initial random exploration
        for _ in range(self.n_initial):
            config = self.space.sample(self.rng)
            t0 = time.time()
            score = objective(config)
            self.tracker.log_trial(config, score, time.time() - t0)

        # Bayesian iterations
        for _ in range(self.n_trials - self.n_initial):
            X = [self._config_to_vector(t["params"]) for t in self.tracker.trials]
            y = self.tracker.scores()
            self.gp.fit(X, y)
            best_score = max(y)

            # Sample candidates and pick best EI
            best_ei = -1
            best_config = None
            for _ in range(100):
                candidate = self.space.sample(self.rng)
                vec = self._config_to_vector(candidate)
                mean, std = self.gp.predict(vec)
                ei = self._expected_improvement(mean, std, best_score)
                if ei > best_ei:
                    best_ei = ei
                    best_config = candidate

            t0 = time.time()
            score = objective(best_config)
            self.tracker.log_trial(best_config, score, time.time() - t0)
            if self.early_stopping and self.early_stopping.should_stop(score):
                break

        return self.tracker.best_trial


# ── Hyperband ─────────────────────────────────────────────────────────

class HyperbandOptimizer:
    """Hyperband optimizer for efficient hyperparameter search."""

    def __init__(self, space: ParameterSpace, max_budget: int = 81, eta: int = 3, seed: int = 42):
        self.space = space
        self.max_budget = max_budget
        self.eta = eta
        self.rng = random.Random(seed)
        self.tracker = ExperimentTracker()

    def optimize(self, objective: Callable[[Dict[str, Any], int], float]) -> Dict[str, Any]:
        """Run Hyperband. *objective(config, budget)* should return a score."""
        s_max = int(math.log(self.max_budget) / math.log(self.eta))

        for s in range(s_max, -1, -1):
            n = int(math.ceil((s_max + 1) / (s + 1) * (self.eta ** s)))
            r = self.max_budget * (self.eta ** (-s))

            configs = [self.space.sample(self.rng) for _ in range(n)]

            for i in range(s + 1):
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)

                scores = []
                for config in configs:
                    t0 = time.time()
                    score = objective(config, r_i)
                    self.tracker.log_trial(config, score, time.time() - t0, {"budget": r_i})
                    scores.append((score, config))

                scores.sort(key=lambda x: x[0], reverse=True)
                top_k = max(1, int(n_i / self.eta))
                configs = [c for _, c in scores[:top_k]]

        return self.tracker.best_trial


# ── Unified AutoML interface ─────────────────────────────────────────

class AutoMLOptimizer:
    """Unified interface for hyperparameter optimization."""

    METHODS = ("grid", "random", "bayesian", "hyperband")

    def __init__(self, n_trials: int = 100, method: str = "bayesian", seed: int = 42):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.n_trials = n_trials
        self.method = method
        self.seed = seed
        self.best_params: Optional[Dict] = None
        self.best_score: Optional[float] = None
        self._tracker: Optional[ExperimentTracker] = None

    def optimize(self, space: ParameterSpace, objective: Callable, **kwargs) -> Dict[str, Any]:
        if self.method == "grid":
            opt = GridSearchOptimizer(space, n_points=kwargs.get("n_points", 5))
        elif self.method == "random":
            opt = RandomSearchOptimizer(space, self.n_trials, self.seed)
        elif self.method == "bayesian":
            opt = BayesianOptimizer(space, self.n_trials, seed=self.seed)
        elif self.method == "hyperband":
            opt = HyperbandOptimizer(space, seed=self.seed)

        best = opt.optimize(objective)
        self._tracker = opt.tracker
        self.best_params = best["params"]
        self.best_score = best["score"]
        return best

    @property
    def tracker(self) -> Optional[ExperimentTracker]:
        return self._tracker

    def evaluate(self, X: Any = None, y: Any = None) -> Dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_params": self.best_params,
            "n_trials": self._tracker.n_trials if self._tracker else 0,
        }
