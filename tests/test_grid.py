"""Tests for grid search axis generation + full run."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.grid import GridSearch, grid_points
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


def test_bool_axis_is_always_two_points():
    spec = ParamSpec(name="flag", dtype="bool", neutral=False)
    assert grid_points(spec, n=5) == [False, True]
    assert grid_points(spec, n=2) == [False, True]


def test_int_axis_small_range_uses_full():
    spec = ParamSpec(name="n", dtype="int", low=1, high=3, neutral=2)
    assert grid_points(spec, n=5) == [1, 2, 3]


def test_int_axis_large_range_uses_linspace():
    spec = ParamSpec(name="n", dtype="int", low=0, high=100, neutral=50)
    pts = grid_points(spec, n=5)
    assert pts[0] == 0
    assert pts[-1] == 100
    assert len(pts) == 5


def test_continuous_axis_linspace():
    spec = ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)
    pts = grid_points(spec, n=5)
    assert pts == [0.0, 2.5, 5.0, 7.5, 10.0]


class QuadraticTarget:
    """f(x, y) = -(x - 3)^2 - (y - 7)^2. Optimum at (3, 7)."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="y", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = params["x"], params["y"]
        f = -((x - 3.0) ** 2 + (y - 7.0) ** 2)
        return EvalResult(fitness=f, n_trials=1)


def test_grid_search_finds_near_optimum():
    target = QuadraticTarget()
    gs = GridSearch(target=target, unlocked=["x", "y"], grid_points_per_axis=5, verbose=False)
    pts = gs.run(base_params={"x": 5.0, "y": 5.0})
    assert len(pts) == 25
    best = max(pts, key=lambda p: p.result.fitness)
    # Grid: [0, 2.5, 5, 7.5, 10] — closest to (3, 7) is (2.5, 7.5), f = -0.5
    assert best.unlocked["x"] == pytest.approx(2.5, abs=0.01)
    assert best.unlocked["y"] == pytest.approx(7.5, abs=0.01)
    assert best.result.fitness == pytest.approx(-0.5, abs=1e-6)


def test_grid_search_respects_locked_params():
    """Locked param values in base_params must pass through unchanged."""
    target = QuadraticTarget()
    gs = GridSearch(target=target, unlocked=["x"], grid_points_per_axis=3, verbose=False)
    pts = gs.run(base_params={"x": 0.0, "y": 7.0})
    # Only x varies; y is locked to 7.0 (the optimum for y)
    assert len(pts) == 3
    for p in pts:
        assert p.params["y"] == 7.0
    best = max(pts, key=lambda p: p.result.fitness)
    # best x ∈ {0, 5, 10}, closest to 3 is 5 → fitness = -(5-3)^2 - 0 = -4
    assert best.unlocked["x"] == pytest.approx(5.0, abs=0.01)


def test_grid_rejects_unknown_param():
    target = QuadraticTarget()
    gs = GridSearch(target=target, unlocked=["z"], grid_points_per_axis=3, verbose=False)
    with pytest.raises(KeyError):
        gs.axes()
