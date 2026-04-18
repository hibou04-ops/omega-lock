"""Tests for run_p2_tpe — the Optuna TPE variant of the P1 pipeline.

All tests are skipped gracefully when optuna is not installed (optional dep).
"""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.p2_tpe import (
    P2Config,
    _OPTUNA_AVAILABLE,
    _OPTUNA_INSTALL_HINT,
    run_p2_tpe,
)
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.kill_criteria import KCThresholds


# Loose KC thresholds suitable for toy deterministic targets where n_trials=1
# per eval (trade_count_min=50 would FAIL trivially) and stress differentiation
# is meaningful but not dramatic.
TOY_THRESHOLDS = KCThresholds(
    trade_count_min=1,
    gini_min=0.0,
    top_bot_ratio_min=1.0,
    pearson_min=0.3,
    trade_ratio_min=0.3,
)


class RosenbrockTarget:
    """Shifted Rosenbrock: f(x, y) = -((0.5 - x)^2 + 100*(y - x^2)^2).

    Optimum at (0.5, 0.25). The shift is deliberate: the unshifted Rosenbrock
    has its optimum at (1, 1), which sits exactly on any 5-point uniform grid
    over [-2, 2] (so grid search would trivially "win" for a contrived
    reason). Shifting to (0.5, 0.25) puts the optimum firmly off the grid —
    the best grid point becomes (0, 0) at err=0.559, whereas TPE at 200
    trials converges within ~0.04. Continuous search genuinely wins.
    """

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=-2.0, high=2.0, neutral=0.0),
            ParamSpec(name="y", dtype="float", low=-2.0, high=2.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = float(params["x"]), float(params["y"])
        f = -((0.5 - x) ** 2 + 100.0 * (y - x ** 2) ** 2)
        return EvalResult(fitness=f, n_trials=1)


class ShiftedQuadratic:
    """Simple convex bowl: optimum at (0.7, 0.3), fitness = 0.

    Used to demonstrate that TPE converges tightly (<0.05) on easy landscapes.
    """

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=-2.0, high=2.0, neutral=0.0),
            ParamSpec(name="y", dtype="float", low=-2.0, high=2.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = float(params["x"]), float(params["y"])
        f = -((x - 0.7) ** 2 + (y - 0.3) ** 2)
        return EvalResult(fitness=f, n_trials=10)


class MixedTypeTarget:
    """Exercises all three dtypes (float, int, bool) — checks spec bounds."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="alpha", dtype="float", low=-1.0, high=1.0, neutral=0.0),
            ParamSpec(name="window", dtype="int", low=5, high=50, neutral=20),
            ParamSpec(name="flag", dtype="bool", neutral=False),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        a = float(params["alpha"])
        w = int(params["window"])
        fl = bool(params["flag"])
        bonus = 0.5 if fl else 0.0
        f = -(a - 0.25) ** 2 - 0.001 * (w - 30) ** 2 + bonus
        return EvalResult(fitness=f, n_trials=5)


# ─── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_tpe_rosenbrock_converges_near_optimum():
    """TPE on shifted Rosenbrock should converge to (0.5, 0.25) within 0.05.

    Why 0.05 is a meaningful bar: the best 5-point uniform grid point on
    [-2, 2]^2 for this objective is (0, 0), sitting 0.559 away from the true
    optimum. Converging to 0.05 means TPE beats grid by >10×, proving
    continuous search is the right tool when the grid can't place a cell
    on the optimum. 200 trials matches 5^3 * 1.6 ~ a single-round P2 budget.
    """
    target = RosenbrockTarget()
    cfg = P2Config(
        unlock_k=2,
        n_trials=200,
        seed=42,
        kc_thresholds=TOY_THRESHOLDS,
        stress_verbose=False,
        trial_verbose=False,
    )
    result = run_p2_tpe(train_target=target, config=cfg)

    assert result.tpe_best is not None
    bx = result.tpe_best["unlocked"]["x"]
    by = result.tpe_best["unlocked"]["y"]
    err = ((bx - 0.5) ** 2 + (by - 0.25) ** 2) ** 0.5
    assert err <= 0.05, (
        f"TPE should converge near (0.5, 0.25); got ({bx:.4f}, {by:.4f}) err={err:.4f}"
    )
    # TPE should meaningfully beat what a 5-point grid would find (err=0.559).
    assert err < 0.2


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_tpe_deterministic_by_seed():
    """Same seed + same target -> identical trial history (and therefore
    identical best params). This is the guarantee TPESampler makes when
    driven single-threaded; we exercise it directly.
    """
    target = RosenbrockTarget()
    cfg = P2Config(
        unlock_k=2,
        n_trials=50,
        seed=123,
        kc_thresholds=TOY_THRESHOLDS,
        stress_verbose=False,
        trial_verbose=False,
    )
    r1 = run_p2_tpe(train_target=target, config=cfg)
    r2 = run_p2_tpe(train_target=target, config=cfg)

    # Full per-trial match (catches sampler-state divergence, not just best)
    assert len(r1.trials) == len(r2.trials) == 50
    for t1, t2 in zip(r1.trials, r2.trials):
        assert t1["trial_idx"] == t2["trial_idx"]
        assert t1["params_unlocked"] == t2["params_unlocked"]
        assert t1["fitness"] == t2["fitness"]
    assert r1.tpe_best == r2.tpe_best


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_tpe_passes_kcs_on_easy_target():
    """Simple convex target should pass every KC (status == 'PASS').

    n_trials=10 per evaluate clears trade_count_min=1 comfortably; the
    shifted-quadratic landscape yields strong stress differentiation on
    the two axes, and walk-forward on the same target is trivially
    Pearson-perfect. This is the smoke test for "the pipeline wires up
    end-to-end without KC fails".
    """
    target = ShiftedQuadratic()
    cfg = P2Config(
        unlock_k=2,
        n_trials=60,
        seed=42,
        walk_forward_top_n=5,
        kc_thresholds=KCThresholds(
            trade_count_min=1,
            gini_min=0.0,
            top_bot_ratio_min=1.0,
            pearson_min=0.3,
            trade_ratio_min=0.3,
        ),
        stress_verbose=False,
        trial_verbose=False,
    )
    result = run_p2_tpe(
        train_target=target,
        config=cfg,
        test_target=target,   # self-walk-forward for KC-4
    )
    assert result.status == "PASS", f"expected PASS, got {result.status}: {result.kc_reports}"
    # And it actually converged (quadratic is easy: err << 0.1)
    bx = result.tpe_best["unlocked"]["x"]
    by = result.tpe_best["unlocked"]["y"]
    err = ((bx - 0.7) ** 2 + (by - 0.3) ** 2) ** 0.5
    assert err <= 0.1, f"quadratic should converge tightly; err={err:.4f}"


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_tpe_respects_spec_bounds():
    """Every trial's suggested value must lie inside the spec range, and
    int/bool dtypes must be preserved. Covers the suggest_float/int/
    categorical dispatch plus the defensive clip() in the objective.
    """
    target = MixedTypeTarget()
    cfg = P2Config(
        unlock_k=3,
        n_trials=40,
        seed=7,
        kc_thresholds=TOY_THRESHOLDS,
        stress_verbose=False,
        trial_verbose=False,
    )
    result = run_p2_tpe(train_target=target, config=cfg)

    specs = {s.name: s for s in target.param_space()}
    assert set(result.top_k) <= set(specs.keys())
    for t in result.trials:
        u = t["params_unlocked"]
        for name, val in u.items():
            spec = specs[name]
            if spec.dtype == "float":
                assert spec.low <= val <= spec.high, f"{name}={val} outside [{spec.low},{spec.high}]"
                assert isinstance(val, float)
            elif spec.dtype == "int":
                assert spec.low <= val <= spec.high, f"{name}={val} outside [{spec.low},{spec.high}]"
                assert isinstance(val, int) and not isinstance(val, bool)
            elif spec.dtype == "bool":
                assert isinstance(val, bool)


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_tpe_with_test_target_runs_walk_forward():
    """When test_target is provided, WalkForward results + KC-4 appear in
    the P2Result. The top-N sort is by train fitness; pearson of
    identical-ranking targets is 1.
    """

    class LinearTrain:
        def param_space(self) -> list[ParamSpec]:
            return [
                ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
                ParamSpec(name="b", dtype="float", low=-5.0, high=5.0, neutral=0.0),
            ]

        def evaluate(self, p: dict[str, Any]) -> EvalResult:
            # stress in both axes; a dominates
            return EvalResult(fitness=p["a"] * 2.0 + p["b"] * 0.5, n_trials=10)

    class LinearTest:
        def param_space(self) -> list[ParamSpec]:
            return [
                ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
                ParamSpec(name="b", dtype="float", low=-5.0, high=5.0, neutral=0.0),
            ]

        def evaluate(self, p: dict[str, Any]) -> EvalResult:
            # Same ranking, different scale → pearson = 1
            return EvalResult(fitness=p["a"] * 1.8 + p["b"] * 0.45, n_trials=8)

    cfg = P2Config(
        unlock_k=2,
        n_trials=30,
        seed=42,
        walk_forward_top_n=5,
        kc_thresholds=TOY_THRESHOLDS,
        stress_verbose=False,
        trial_verbose=False,
    )
    result = run_p2_tpe(
        train_target=LinearTrain(),
        test_target=LinearTest(),
        config=cfg,
    )
    assert result.walk_forward is not None
    assert result.walk_forward["top_n"] == 5
    assert len(result.walk_forward["train_fitnesses"]) == 5
    assert len(result.walk_forward["test_fitnesses"]) == 5
    assert result.walk_forward["pearson"] == pytest.approx(1.0, abs=1e-6)

    # KC-4 should appear in the reports
    kc_names = {r["name"] for r in result.kc_reports}
    assert "KC-4" in kc_names


def test_p2_tpe_raises_without_optuna(monkeypatch):
    """If optuna is unavailable, run_p2_tpe raises ImportError with an
    actionable install hint. We monkeypatch the module flag rather than
    ripping optuna out of sys.modules — cleaner and deterministic.
    """
    from omega_lock import p2_tpe as module

    monkeypatch.setattr(module, "_OPTUNA_AVAILABLE", False)

    class Dummy:
        def param_space(self) -> list[ParamSpec]:
            return [ParamSpec(name="x", dtype="float", low=0.0, high=1.0, neutral=0.5)]

        def evaluate(self, p: dict[str, Any]) -> EvalResult:
            return EvalResult(fitness=0.0, n_trials=1)

    with pytest.raises(ImportError) as excinfo:
        module.run_p2_tpe(train_target=Dummy())
    msg = str(excinfo.value)
    assert "optuna" in msg.lower()
    assert "install" in msg.lower()
    # sanity: the exposed hint string itself should be the message
    assert msg == _OPTUNA_INSTALL_HINT
