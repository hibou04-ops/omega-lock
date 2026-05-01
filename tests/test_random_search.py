"""Tests for RandomSearch baseline (P1 SPEC §4 SC-2)."""
from __future__ import annotations

import math
from typing import Any

import pytest

from omega_lock.grid import GridPoint, GridSearch
from omega_lock.random_search import (
    RandomSearch,
    compare_to_grid,
    top_quartile_fitness,
)
from omega_lock.target import EvalResult, ParamSpec


class QuadraticTargetPositive:
    """f(x, y) = 100 - (x - 3)^2 - (y - 7)^2 on [0, 10]^2.

    Positive-valued form of the usual test target. Keeps top-quartile
    means positive so ratio comparisons stay monotone with 'better'.
    Optimum is at (3, 7) with f = 100.
    """

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="y", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = params["x"], params["y"]
        f = 100.0 - ((x - 3.0) ** 2 + (y - 7.0) ** 2)
        return EvalResult(fitness=f, n_trials=1)


class MixedDtypeTarget:
    """Mix of float + int + bool. Used to verify type-correct sampling."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="alpha", dtype="float", low=-1.0, high=1.0, neutral=0.0),
            ParamSpec(name="n", dtype="int", low=1, high=10, neutral=5),
            ParamSpec(name="flag", dtype="bool", neutral=False),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        bonus = 1.0 if params["flag"] else 0.0
        f = -abs(params["alpha"]) + 0.1 * params["n"] + bonus
        return EvalResult(fitness=f, n_trials=1)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------
def test_random_search_is_deterministic_for_fixed_seed():
    target = QuadraticTargetPositive()
    base = {"x": 5.0, "y": 5.0}

    rs1 = RandomSearch(target=target, unlocked=["x", "y"], n_samples=25,
                       seed=123, verbose=False)
    rs2 = RandomSearch(target=target, unlocked=["x", "y"], n_samples=25,
                       seed=123, verbose=False)
    pts1 = rs1.run(base_params=base)
    pts2 = rs2.run(base_params=base)

    assert len(pts1) == len(pts2) == 25
    for p1, p2 in zip(pts1, pts2):
        assert p1.unlocked == p2.unlocked
        assert p1.result.fitness == pytest.approx(p2.result.fitness, abs=1e-12)


def test_random_search_different_seeds_yield_different_points():
    target = QuadraticTargetPositive()
    base = {"x": 5.0, "y": 5.0}
    pts_a = RandomSearch(target=target, unlocked=["x", "y"], n_samples=25,
                         seed=1, verbose=False).run(base)
    pts_b = RandomSearch(target=target, unlocked=["x", "y"], n_samples=25,
                         seed=2, verbose=False).run(base)
    # Extremely unlikely that 25 (x,y) pairs all match across different seeds
    matches = sum(1 for a, b in zip(pts_a, pts_b) if a.unlocked == b.unlocked)
    assert matches < len(pts_a)


# ---------------------------------------------------------------------------
# 2. Bounds respected
# ---------------------------------------------------------------------------
def test_random_search_respects_spec_bounds_float():
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["x", "y"], n_samples=200,
                      seed=7, verbose=False)
    pts = rs.run(base_params={"x": 5.0, "y": 5.0})
    for p in pts:
        assert 0.0 <= p.unlocked["x"] <= 10.0
        assert 0.0 <= p.unlocked["y"] <= 10.0
        assert p.params["x"] == p.unlocked["x"]
        assert p.params["y"] == p.unlocked["y"]


def test_random_search_respects_spec_bounds_mixed():
    target = MixedDtypeTarget()
    rs = RandomSearch(target=target, unlocked=["alpha", "n", "flag"],
                      n_samples=200, seed=11, verbose=False)
    pts = rs.run(base_params={"alpha": 0.0, "n": 5, "flag": False})
    for p in pts:
        assert -1.0 <= p.unlocked["alpha"] <= 1.0
        assert 1 <= p.unlocked["n"] <= 10
        assert p.unlocked["flag"] in (True, False)
        # params must mirror unlocked for sampled keys
        assert p.params["alpha"] == p.unlocked["alpha"]
        assert p.params["n"] == p.unlocked["n"]
        assert p.params["flag"] == p.unlocked["flag"]


def test_random_search_preserves_locked_params():
    """Params not in `unlocked` must pass through from base_params unchanged."""
    target = MixedDtypeTarget()
    rs = RandomSearch(target=target, unlocked=["alpha"], n_samples=20,
                      seed=3, verbose=False)
    pts = rs.run(base_params={"alpha": 0.0, "n": 7, "flag": True})
    for p in pts:
        assert p.params["n"] == 7
        assert p.params["flag"] is True


# ---------------------------------------------------------------------------
# 3. top_quartile_fitness
# ---------------------------------------------------------------------------
def test_top_quartile_is_at_least_median():
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["x", "y"], n_samples=100,
                      seed=5, verbose=False)
    pts = rs.run(base_params={"x": 5.0, "y": 5.0})
    fitnesses = sorted(p.result.fitness for p in pts)
    median = fitnesses[len(fitnesses) // 2]
    tq = top_quartile_fitness(pts)
    assert tq >= median


def test_top_quartile_single_point_equals_that_point():
    # With n=1 the quartile collapses to just that point.
    target = QuadraticTargetPositive()
    pts = [
        GridPoint(idx=0, unlocked={"x": 3.0, "y": 7.0},
                  params={"x": 3.0, "y": 7.0},
                  result=EvalResult(fitness=42.0, n_trials=1)),
    ]
    assert top_quartile_fitness(pts) == 42.0


def test_top_quartile_on_125_points_uses_top_31():
    # Known fitnesses 1..125 -> top 31 are 95..125, mean = (95+125)/2 = 110.
    pts = [
        GridPoint(idx=i, unlocked={}, params={},
                  result=EvalResult(fitness=float(i + 1), n_trials=0))
        for i in range(125)
    ]
    expected = sum(range(95, 126)) / 31  # 95..125 inclusive = 31 values
    assert top_quartile_fitness(pts) == pytest.approx(expected)


def test_top_quartile_empty_raises():
    with pytest.raises(ValueError):
        top_quartile_fitness([])


# ---------------------------------------------------------------------------
# 4. compare_to_grid ratio on a 2D quadratic
# ---------------------------------------------------------------------------
class NarrowGridQuadratic:
    """Same quadratic but with param_space restricted around the optimum.

    Used to simulate a 'grid given a small range around optimum' case.
    The full-range version is `QuadraticTargetPositive`.
    """

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=2.5, high=3.5, neutral=3.0),
            ParamSpec(name="y", dtype="float", low=6.5, high=7.5, neutral=7.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = params["x"], params["y"]
        f = 100.0 - ((x - 3.0) ** 2 + (y - 7.0) ** 2)
        return EvalResult(fitness=f, n_trials=1)


def test_compare_to_grid_ratio_ge_one_when_grid_is_near_optimum():
    """Grid over a narrow range around optimum should beat random over full range."""
    narrow = NarrowGridQuadratic()
    full = QuadraticTargetPositive()

    grid = GridSearch(target=narrow, unlocked=["x", "y"],
                      grid_points_per_axis=5, verbose=False)
    grid_pts = grid.run(base_params={"x": 3.0, "y": 7.0})

    rs = RandomSearch(target=full, unlocked=["x", "y"], n_samples=25,
                      seed=42, verbose=False)
    rand_pts = rs.run(base_params={"x": 5.0, "y": 5.0})

    report = compare_to_grid(grid_pts, rand_pts)
    assert report["grid_top_quartile"] > report["random_top_quartile"]
    assert report["ratio"] >= 1.0


def test_compare_to_grid_returns_all_keys():
    target = QuadraticTargetPositive()
    grid = GridSearch(target=target, unlocked=["x", "y"],
                      grid_points_per_axis=5, verbose=False)
    grid_pts = grid.run(base_params={"x": 5.0, "y": 5.0})
    rand_pts = RandomSearch(target=target, unlocked=["x", "y"], n_samples=25,
                            seed=42, verbose=False).run({"x": 5.0, "y": 5.0})
    report = compare_to_grid(grid_pts, rand_pts)
    # Required keys (sc2_warning is conditional — only present when
    # negative fitness makes the ratio unstable; not asserted here).
    required_keys = {
        "grid_top_quartile",
        "random_top_quartile",
        "ratio",
        "sc2_pass",
        "sc2_assumption",
    }
    assert required_keys.issubset(set(report.keys()))
    assert report["sc2_pass"] in (0.0, 1.0)


def test_compare_to_grid_handles_zero_random_top_quartile():
    """If random top-quartile is zero, ratio is inf (positive grid) or -inf."""
    zero_pts = [
        GridPoint(idx=i, unlocked={}, params={},
                  result=EvalResult(fitness=0.0, n_trials=0))
        for i in range(4)
    ]
    pos_pts = [
        GridPoint(idx=i, unlocked={}, params={},
                  result=EvalResult(fitness=10.0, n_trials=0))
        for i in range(4)
    ]
    report = compare_to_grid(pos_pts, zero_pts)
    assert math.isinf(report["ratio"]) and report["ratio"] > 0
    assert report["sc2_pass"] == 1.0

    # Both zero -> ratio 0, sc2 fails
    report_zero = compare_to_grid(zero_pts, zero_pts)
    assert report_zero["ratio"] == 0.0
    assert report_zero["sc2_pass"] == 0.0


# ---------------------------------------------------------------------------
# 5. n_samples count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 7, 125, 200])
def test_random_search_produces_exactly_n_samples(n):
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["x", "y"], n_samples=n,
                      seed=1, verbose=False)
    pts = rs.run(base_params={"x": 5.0, "y": 5.0})
    assert len(pts) == n
    # Indices should be 0..n-1 (mirrors GridSearch.run)
    assert [p.idx for p in pts] == list(range(n))


# ---------------------------------------------------------------------------
# 6. Mixed dtypes — types are right
# ---------------------------------------------------------------------------
def test_random_search_mixed_dtypes_produces_correct_types():
    target = MixedDtypeTarget()
    rs = RandomSearch(target=target, unlocked=["alpha", "n", "flag"],
                      n_samples=200, seed=19, verbose=False)
    pts = rs.run(base_params={"alpha": 0.0, "n": 5, "flag": False})

    alpha_vals = [p.unlocked["alpha"] for p in pts]
    n_vals = [p.unlocked["n"] for p in pts]
    flag_vals = [p.unlocked["flag"] for p in pts]

    # Float values are actual Python floats (not numpy) and in range.
    for a in alpha_vals:
        assert isinstance(a, float)
        assert -1.0 <= a <= 1.0

    # Int values are actual Python ints and in [1, 10].
    for nv in n_vals:
        assert isinstance(nv, int)
        assert not isinstance(nv, bool)  # bool is subclass of int -> exclude
        assert 1 <= nv <= 10

    # Bool values are actual Python bools and BOTH True and False appear.
    for fv in flag_vals:
        assert isinstance(fv, bool)
    assert set(flag_vals) == {True, False}

    # All ten integer values should be hit over 200 samples.
    assert set(n_vals) == set(range(1, 11))


# ---------------------------------------------------------------------------
# Misc sanity
# ---------------------------------------------------------------------------
def test_random_search_rejects_unknown_param():
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["z"], n_samples=5,
                      seed=0, verbose=False)
    with pytest.raises(KeyError):
        rs.run(base_params={"x": 5.0, "y": 5.0})


def test_random_search_rejects_negative_n_samples():
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["x"], n_samples=-1,
                      seed=0, verbose=False)
    with pytest.raises(ValueError):
        rs.run(base_params={"x": 5.0, "y": 5.0})


def test_random_search_output_shape_matches_gridsearch():
    """RandomSearch.run must return GridPoints compatible with walk-forward."""
    target = QuadraticTargetPositive()
    rs = RandomSearch(target=target, unlocked=["x", "y"], n_samples=10,
                      seed=0, verbose=False)
    pts = rs.run(base_params={"x": 5.0, "y": 5.0})
    for p in pts:
        assert isinstance(p, GridPoint)
        assert isinstance(p.idx, int)
        assert isinstance(p.unlocked, dict)
        assert isinstance(p.params, dict)
        assert isinstance(p.result, EvalResult)
        assert p.wall_seconds >= 0.0
        # full param dict must include locked-but-present keys
        assert "x" in p.params and "y" in p.params
