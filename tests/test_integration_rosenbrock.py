"""End-to-end integration test: Rosenbrock target through full P1 pipeline."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock import (
    EvalResult,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1,
)


class RosenbrockTarget:
    """f(x, y) = -((1-x)^2 + 100*(y - x^2)^2), optimum at (1, 1)."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=-2.0, high=2.0, neutral=0.0),
            ParamSpec(name="y", dtype="float", low=-2.0, high=2.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = float(params["x"]), float(params["y"])
        f = -((1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2)
        return EvalResult(fitness=f, n_trials=1)


def test_rosenbrock_p1_converges_near_optimum():
    # 21-point grid on [-2, 2] gives step=0.2 → (1.0, 1.0) is an exact grid point (optimum).
    target = RosenbrockTarget()
    cfg = P1Config(
        unlock_k=2,
        grid_points_per_axis=21,
        kc_thresholds=KCThresholds(trade_count_min=1),
        stress_verbose=False,
        grid_verbose=False,
    )
    result = run_p1(train_target=target, config=cfg, test_target=None, output_path=None)

    assert result.grid_best is not None
    bx = result.grid_best["unlocked"]["x"]
    by = result.grid_best["unlocked"]["y"]
    assert abs(bx - 1.0) <= 0.25, f"x should be near 1.0, got {bx}"
    assert abs(by - 1.0) <= 0.25, f"y should be near 1.0, got {by}"
    assert result.grid_best["fitness"] >= -1e-6   # optimum = 0


def test_rosenbrock_stress_both_params_nonzero():
    target = RosenbrockTarget()
    cfg = P1Config(unlock_k=2, grid_points_per_axis=3, stress_verbose=False, grid_verbose=False,
                   kc_thresholds=KCThresholds(trade_count_min=1))
    result = run_p1(train_target=target, config=cfg, output_path=None)
    stresses = {r["name"]: r["raw_stress"] for r in result.stress_results}
    assert stresses["x"] > 0.0
    assert stresses["y"] > 0.0


def test_rosenbrock_kc2_passes_differentiated_stress():
    """Rosenbrock is non-symmetric; stress should differ between x and y."""
    target = RosenbrockTarget()
    cfg = P1Config(unlock_k=2, grid_points_per_axis=3, stress_verbose=False, grid_verbose=False,
                   kc_thresholds=KCThresholds(trade_count_min=1, gini_min=0.0, top_bot_ratio_min=1.0))
    result = run_p1(train_target=target, config=cfg, output_path=None)
    kc_by_name = {r["name"]: r for r in result.kc_reports}
    assert kc_by_name["KC-2"]["status"] == "PASS"


def test_run_p1_with_walk_forward_hybrid():
    """Pipeline should accept test_target + validation_target together."""

    class LinearTrain:
        def param_space(self):
            return [ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0)]

        def evaluate(self, p):
            return EvalResult(fitness=p["a"] * 2.0, n_trials=10)

    class LinearTest:
        def param_space(self):
            return [ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0)]

        def evaluate(self, p):
            # Corresponding test fitness: same ranking, different scale
            return EvalResult(fitness=p["a"] * 1.8, n_trials=8)

    class Validator:
        def param_space(self):
            return [ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0)]

        def evaluate(self, p):
            return EvalResult(fitness=p["a"] * 1.5 + 1.0, n_trials=5)

    cfg = P1Config(
        unlock_k=1,
        grid_points_per_axis=5,
        walk_forward_top_n=3,
        kc_thresholds=KCThresholds(
            trade_count_min=1,
            gini_min=0.0,                # 1-param case: gini = 0, disable check
            top_bot_ratio_min=1.0,       # already auto-skipped for 1 param
            pearson_min=0.3,
            trade_ratio_min=0.3,
        ),
        stress_verbose=False,
        grid_verbose=False,
    )
    result = run_p1(
        train_target=LinearTrain(),
        test_target=LinearTest(),
        validation_target=Validator(),
        config=cfg,
    )
    assert result.walk_forward is not None
    assert result.hybrid_top is not None
    assert len(result.hybrid_top) > 0
    # Perfect-rank preservation → pearson = 1
    assert result.walk_forward["pearson"] == pytest.approx(1.0, abs=1e-6)
