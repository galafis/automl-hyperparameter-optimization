"""
Tests for AutoML Hyperparameter Optimization.
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.optimizer import (
    ParameterSpace,
    ExperimentTracker,
    EarlyStopping,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    HyperbandOptimizer,
    AutoMLOptimizer,
)


def sphere_objective(params):
    """Simple sphere function to minimize (maximize negative)."""
    return -sum(v ** 2 for v in params.values() if isinstance(v, (int, float)))


def make_space():
    return (
        ParameterSpace()
        .add_float("x", -5, 5)
        .add_float("y", -5, 5)
    )


class TestParameterSpace:
    def test_add_float(self):
        space = ParameterSpace().add_float("lr", 0.001, 0.1)
        assert "lr" in space.param_names

    def test_add_int(self):
        space = ParameterSpace().add_int("depth", 3, 10)
        sample = space.sample()
        assert 3 <= sample["depth"] <= 10

    def test_add_categorical(self):
        space = ParameterSpace().add_categorical("algo", ["rf", "xgb"])
        sample = space.sample()
        assert sample["algo"] in ["rf", "xgb"]

    def test_sample(self):
        space = make_space()
        sample = space.sample()
        assert -5 <= sample["x"] <= 5
        assert -5 <= sample["y"] <= 5

    def test_grid(self):
        space = ParameterSpace().add_float("a", 0, 1).add_float("b", 0, 1)
        grid = space.grid(n_points=3)
        assert len(grid) == 9  # 3 x 3

    def test_log_scale(self):
        space = ParameterSpace().add_float("lr", 0.001, 1.0, log_scale=True)
        sample = space.sample()
        assert 0.001 <= sample["lr"] <= 1.0


class TestExperimentTracker:
    def test_log_and_retrieve(self):
        tracker = ExperimentTracker()
        tracker.log_trial({"x": 1}, 0.8)
        tracker.log_trial({"x": 2}, 0.9)
        assert tracker.n_trials == 2
        assert tracker.best_trial["score"] == 0.9

    def test_best_score_over_time(self):
        tracker = ExperimentTracker()
        tracker.log_trial({}, 0.5)
        tracker.log_trial({}, 0.3)
        tracker.log_trial({}, 0.7)
        assert tracker.best_score_over_time() == [0.5, 0.5, 0.7]

    def test_empty_tracker(self):
        tracker = ExperimentTracker()
        assert tracker.best_trial is None


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        assert not es.should_stop(0.5)
        assert not es.should_stop(0.6)
        assert not es.should_stop(0.7)

    def test_stop_when_stagnant(self):
        es = EarlyStopping(patience=3)
        es.should_stop(0.9)
        es.should_stop(0.8)
        es.should_stop(0.8)
        assert es.should_stop(0.8)

    def test_reset(self):
        es = EarlyStopping(patience=2)
        es.should_stop(0.9)
        es.should_stop(0.5)
        es.should_stop(0.5)
        es.reset()
        assert not es.should_stop(0.1)


class TestGridSearchOptimizer:
    def test_exhaustive_search(self):
        space = ParameterSpace().add_float("x", -2, 2)
        opt = GridSearchOptimizer(space, n_points=5)
        best = opt.optimize(sphere_objective)
        assert best is not None
        assert best["score"] <= 0

    def test_tracks_all_trials(self):
        space = ParameterSpace().add_float("x", 0, 1)
        opt = GridSearchOptimizer(space, n_points=3)
        opt.optimize(sphere_objective)
        assert opt.tracker.n_trials == 3


class TestRandomSearchOptimizer:
    def test_basic_search(self):
        space = make_space()
        opt = RandomSearchOptimizer(space, n_trials=20, seed=42)
        best = opt.optimize(sphere_objective)
        assert best is not None
        assert best["score"] <= 0

    def test_early_stopping_integration(self):
        space = make_space()
        opt = RandomSearchOptimizer(space, n_trials=100, seed=42)
        opt.set_early_stopping(patience=5)
        opt.optimize(sphere_objective)
        assert opt.tracker.n_trials <= 100


class TestBayesianOptimizer:
    def test_basic_optimization(self):
        space = make_space()
        opt = BayesianOptimizer(space, n_trials=15, n_initial=5, seed=42)
        best = opt.optimize(sphere_objective)
        assert best is not None
        assert best["score"] <= 0

    def test_with_early_stopping(self):
        space = make_space()
        opt = BayesianOptimizer(space, n_trials=50, seed=42)
        opt.set_early_stopping(patience=5)
        opt.optimize(sphere_objective)
        assert opt.tracker.n_trials <= 50


class TestHyperbandOptimizer:
    def test_basic_hyperband(self):
        space = make_space()

        def objective_with_budget(params, budget):
            base = sphere_objective(params)
            return base + 0.01 * budget

        opt = HyperbandOptimizer(space, max_budget=27, eta=3, seed=42)
        best = opt.optimize(objective_with_budget)
        assert best is not None


class TestAutoMLOptimizer:
    def test_random_method(self):
        space = make_space()
        automl = AutoMLOptimizer(n_trials=10, method="random")
        best = automl.optimize(space, sphere_objective)
        assert automl.best_params is not None
        assert automl.best_score is not None

    def test_bayesian_method(self):
        space = make_space()
        automl = AutoMLOptimizer(n_trials=10, method="bayesian")
        best = automl.optimize(space, sphere_objective)
        assert automl.best_score <= 0

    def test_evaluate(self):
        space = make_space()
        automl = AutoMLOptimizer(n_trials=5, method="random")
        automl.optimize(space, sphere_objective)
        result = automl.evaluate()
        assert "best_score" in result
        assert "n_trials" in result
        assert result["n_trials"] == 5

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            AutoMLOptimizer(method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
