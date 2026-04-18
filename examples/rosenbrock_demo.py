"""Rosenbrock 2D demo — sanity-check the full P1 pipeline.

Rosenbrock function: f(x, y) = -((1 - x)**2 + 100 * (y - x**2)**2)
Optimum: (x=1, y=1), f=0.

This demo:
    1. Builds a RosenbrockTarget (implements CalibrableTarget)
    2. Runs stress measurement → expects both params to have non-trivial stress
    3. Runs K=2 unlock (whole space is 2D anyway) + grid search
    4. Skips walk-forward (static function, no train/test split)
    5. Verifies grid_best converges near (1, 1)

Use this as the reference for how to implement a new CalibrableTarget.

Run:
    python examples/rosenbrock_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow `python examples/rosenbrock_demo.py` without pip install
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import (
    CalibrableTarget,
    EvalResult,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1,
)


class RosenbrockTarget:
    """2D Rosenbrock, parameters bounded to [-2, 2]."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=-2.0, high=2.0, neutral=0.0),
            ParamSpec(name="y", dtype="float", low=-2.0, high=2.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x = float(params["x"])
        y = float(params["y"])
        # Negate so the orchestrator's "maximize fitness" matches Rosenbrock minimization.
        value = -((1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2)
        return EvalResult(
            fitness=value,
            n_trials=1,
            metadata={"x": x, "y": y},
        )


def main() -> int:
    target = RosenbrockTarget()

    # Toy thresholds: relax KC-3 (no "trades" concept here) and KC-4 is skipped
    # since we run without test_target.
    thresholds = KCThresholds(
        trade_count_min=1,       # n_trials=1 always; don't fail KC-3
        pearson_min=0.3,         # unused (no test_target)
        trade_ratio_min=0.5,     # unused
    )
    cfg = P1Config(
        unlock_k=2,              # only 2 params exist
        grid_points_per_axis=21, # 21x21 = 441 grid points → (1.0, 1.0) is an exact point (optimum)
        kc_thresholds=thresholds,
        stress_verbose=True,
        grid_verbose=True,
    )

    output_path = HERE.parent / "output" / "rosenbrock_run.json"
    result = run_p1(
        train_target=target,
        config=cfg,
        test_target=None,        # static function, no train/test split
        validation_target=None,
        output_path=output_path,
    )

    print("\n-- Rosenbrock P1 summary --")
    print(f"  status:   {result.status}")
    print(f"  elapsed:  {result.elapsed_seconds:.2f}s")
    print(f"  baseline: fitness={result.baseline_result['fitness']:.4f}")
    if result.grid_best:
        gb = result.grid_best
        print(f"  best:     fitness={gb['fitness']:.4f} at {gb['unlocked']}")
    for kc in result.kc_reports:
        print(f"  {kc['name']}: {kc['status']:4s} - {kc['message']}")
    print(f"\n  output:   {output_path}")

    assert result.grid_best is not None, "grid_best is None — pipeline regression"
    best_x = result.grid_best["unlocked"]["x"]
    best_y = result.grid_best["unlocked"]["y"]
    # 21x21 grid on [-2, 2] step=0.2 → (1.0, 1.0) is an exact grid point, optimum = 0.
    assert abs(best_x - 1.0) <= 0.25, f"x should be near 1.0, got {best_x}"
    assert abs(best_y - 1.0) <= 0.25, f"y should be near 1.0, got {best_y}"
    print("\nRosenbrock demo PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
