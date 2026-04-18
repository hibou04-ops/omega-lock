"""Tests for ZoomingGridSearch (fractal-vise refinement)."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.grid import GridSearch, ZoomingGridSearch, grid_points_in
from omega_lock.target import EvalResult, ParamSpec


class AsymmetricQuadratic:
    """Optimum at (x=0.7, y=0.49). Asymmetric scales so both axes get non-zero stress."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=-2.0, high=2.0, neutral=0.0),
            ParamSpec(name="y", dtype="float", low=-2.0, high=2.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        f = -(10.0 * (params["x"] - 0.7) ** 2 + 1.0 * (params["y"] - 0.49) ** 2)
        return EvalResult(fitness=f, n_trials=1)


def test_grid_points_in_respects_subrange():
    spec = ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)
    pts = grid_points_in(spec, lo=2.0, hi=4.0, n=5)
    assert pts == [2.0, 2.5, 3.0, 3.5, 4.0]


def test_grid_points_in_clips_outside_spec():
    spec = ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)
    pts = grid_points_in(spec, lo=-5.0, hi=15.0, n=3)
    assert pts == [0.0, 5.0, 10.0]


def test_grid_points_in_int_small_range_uses_full():
    spec = ParamSpec(name="n", dtype="int", low=0, high=100, neutral=50)
    pts = grid_points_in(spec, lo=3, hi=6, n=5)
    assert pts == [3, 4, 5, 6]


def test_grid_points_in_bool_always_both():
    spec = ParamSpec(name="flag", dtype="bool", neutral=False)
    assert grid_points_in(spec, lo=False, hi=True, n=5) == [False, True]


def test_zooming_rejects_bad_factor():
    target = AsymmetricQuadratic()
    with pytest.raises(ValueError):
        ZoomingGridSearch(target=target, unlocked=["x"], zoom_rounds=3, zoom_factor=1.5)
    with pytest.raises(ValueError):
        ZoomingGridSearch(target=target, unlocked=["x"], zoom_rounds=3, zoom_factor=0.0)


def test_zooming_rejects_zero_rounds():
    target = AsymmetricQuadratic()
    with pytest.raises(ValueError):
        ZoomingGridSearch(target=target, unlocked=["x"], zoom_rounds=0)


def test_zooming_rounds_1_matches_plain_gridsearch():
    """zoom_rounds=1 should return exactly the same points as GridSearch."""
    target = AsymmetricQuadratic()
    base = {"x": 0.0, "y": 0.0}

    plain = GridSearch(target=target, unlocked=["x", "y"], grid_points_per_axis=5, verbose=False)
    plain_pts = plain.run(base_params=base)

    zoom1 = ZoomingGridSearch(
        target=target, unlocked=["x", "y"], grid_points_per_axis=5,
        zoom_rounds=1, verbose=False,
    )
    zoom_pts = zoom1.run(base_params=base)

    assert len(plain_pts) == len(zoom_pts) == 25
    plain_fitnesses = sorted(p.result.fitness for p in plain_pts)
    zoom_fitnesses = sorted(p.result.fitness for p in zoom_pts)
    assert plain_fitnesses == pytest.approx(zoom_fitnesses, abs=1e-9)


def test_zooming_improves_precision_geometrically():
    """Each pair of zoom rounds should shrink the error by ~4x (zoom_factor=0.5)."""
    target = AsymmetricQuadratic()
    base = {"x": 0.0, "y": 0.0}

    def best_error(zoom_rounds: int) -> float:
        gs = ZoomingGridSearch(
            target=target, unlocked=["x", "y"], grid_points_per_axis=5,
            zoom_rounds=zoom_rounds, zoom_factor=0.5, verbose=False,
        )
        pts = gs.run(base_params=base)
        best = max(pts, key=lambda p: p.result.fitness)
        return ((best.unlocked["x"] - 0.7) ** 2 + (best.unlocked["y"] - 0.49) ** 2) ** 0.5

    e1 = best_error(1)
    e4 = best_error(4)
    e8 = best_error(8)
    # Monotone improvement
    assert e4 < e1
    assert e8 < e4
    # At least an order of magnitude improvement over 8 rounds
    assert e8 * 10 < e1
    # Final error should be tight
    assert e8 < 5e-2


def test_zooming_total_point_count():
    """Total evaluations = zoom_rounds * (plain grid size) for continuous axes."""
    target = AsymmetricQuadratic()
    gs = ZoomingGridSearch(
        target=target, unlocked=["x", "y"], grid_points_per_axis=5,
        zoom_rounds=3, verbose=False,
    )
    pts = gs.run(base_params={"x": 0.0, "y": 0.0})
    assert len(pts) == 3 * 25


def test_zooming_stays_within_spec_bounds():
    """Zoom ranges must always clip to original spec.low/high."""
    target = AsymmetricQuadratic()
    gs = ZoomingGridSearch(
        target=target, unlocked=["x", "y"], grid_points_per_axis=5,
        zoom_rounds=6, zoom_factor=0.5, verbose=False,
    )
    pts = gs.run(base_params={"x": 0.0, "y": 0.0})
    for p in pts:
        assert -2.0 <= p.unlocked["x"] <= 2.0
        assert -2.0 <= p.unlocked["y"] <= 2.0


def test_zooming_with_bool_axis_keeps_bool_fixed():
    """Bool axes should be {False, True} in every zoom round (no shrinking)."""
    class Mixed:
        def param_space(self):
            return [
                ParamSpec("x", "float", low=0.0, high=10.0, neutral=5.0),
                ParamSpec("flag", "bool", neutral=False),
            ]
        def evaluate(self, p):
            bonus = 1.0 if p["flag"] else 0.0
            return EvalResult(fitness=-(p["x"] - 3.0) ** 2 + bonus, n_trials=1)

    gs = ZoomingGridSearch(
        target=Mixed(), unlocked=["x", "flag"], grid_points_per_axis=5,
        zoom_rounds=3, verbose=False,
    )
    pts = gs.run(base_params={"x": 5.0, "flag": False})
    # Each round has 5 (x) * 2 (flag) = 10 points → 3 rounds → 30
    assert len(pts) == 30
    # Both bool values appear in every round
    flags_seen = {p.unlocked["flag"] for p in pts}
    assert flags_seen == {False, True}
