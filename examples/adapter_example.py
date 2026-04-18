"""Adapter pattern — wrap an arbitrary external system as CalibrableTarget.

Shows the two idiomatic patterns:
    1. `CallableAdapter` — quickest. Wrap a plain `(params) -> float` function.
    2. Custom class — more control. Implement `param_space()` and
       `evaluate()` directly when you need state (e.g., holding a trained
       model, a simulation engine, a database connection).

This is the template for integrating Omega-Lock with real systems — e.g.,
the HeartCore strategy adapter that inspired the framework.

Run:
    python examples/adapter_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import (
    CallableAdapter, EvalResult, KCThresholds, P1Config, ParamSpec, run_p1,
)


# ── Pattern 1: CallableAdapter (simplest) ──────────────────────────────

def pattern_1_callable() -> None:
    print("── Pattern 1: CallableAdapter ──")

    # Pretend this is an external scoring function from elsewhere
    # (an existing library, a REST call, a simulation result).
    def external_score(params: dict[str, Any]) -> float:
        # Toy: quadratic bowl with optimum at (3, 7), max fitness = 0
        return -((params["a"] - 3.0) ** 2 + (params["b"] - 7.0) ** 2)

    # Wrap in one call — no subclassing.
    target = CallableAdapter(
        fitness_fn=external_score,
        specs=[
            ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ],
    )

    result = run_p1(
        train_target=target,
        config=P1Config(
            unlock_k=2, grid_points_per_axis=5, zoom_rounds=4,
            kc_thresholds=KCThresholds(trade_count_min=1, gini_min=0.0, top_bot_ratio_min=1.0),
            stress_verbose=False, grid_verbose=False,
        ),
    )
    gb = result.grid_best
    print(f"  status: {result.status}")
    print(f"  grid_best: a={gb['unlocked']['a']:.4f} b={gb['unlocked']['b']:.4f} fitness={gb['fitness']:.4f}")
    print(f"  (true optimum a=3.0, b=7.0, fitness=0.0)")


# ── Pattern 2: Custom class (stateful external system) ─────────────────

class ExternalSystemTarget:
    """Template for wrapping a real external system.

    Use this pattern when your target has:
      - Internal state (trained models, pre-loaded datasets, sessions)
      - Non-trivial setup cost
      - Multiple evaluation phases
      - Custom metadata worth surfacing to callers
    """

    def __init__(self, data: list[float], label_threshold: float = 0.5):
        # Expensive state setup happens once, at construction.
        self.data = data
        self.label_threshold = label_threshold
        self.n_calls = 0   # diagnostic — call count

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="threshold", dtype="float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec(name="window",    dtype="int",   low=1,   high=20, neutral=5),
            ParamSpec(name="absolute",  dtype="bool",  neutral=False),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        self.n_calls += 1
        thr = float(params["threshold"])
        win = int(params["window"])
        absol = bool(params["absolute"])

        # Count events where rolling mean crosses threshold
        hits = 0
        rewards = 0.0
        for i in range(win, len(self.data)):
            val = sum(self.data[i - win:i]) / win
            if absol:
                val = abs(val)
            if val > thr:
                hits += 1
                rewards += 1.0 if self.data[i] > self.label_threshold else -1.0

        return EvalResult(
            fitness=rewards,
            n_trials=hits,
            metadata={
                "call_idx": self.n_calls,
                "threshold": thr,
                "window": win,
                "absolute": absol,
            },
        )


def pattern_2_stateful() -> None:
    print("\n── Pattern 2: Stateful class ──")

    import random
    random.seed(42)
    # Mock up an external data stream (in practice: load from disk, DB, API)
    data = [random.uniform(-1, 1) for _ in range(200)]

    target = ExternalSystemTarget(data=data, label_threshold=0.3)
    result = run_p1(
        train_target=target,
        config=P1Config(
            unlock_k=3, grid_points_per_axis=5,
            kc_thresholds=KCThresholds(trade_count_min=1, gini_min=0.0, top_bot_ratio_min=1.0),
            stress_verbose=False, grid_verbose=False,
        ),
    )
    gb = result.grid_best
    print(f"  status: {result.status}")
    print(f"  grid_best: {gb['unlocked']}  fitness={gb['fitness']:.2f}  trials={gb['n_trials']}")
    print(f"  total evaluate() calls: {target.n_calls}")
    if result.status.startswith("FAIL"):
        print(f"  NOTE: {result.status} is expected for this minimal toy data (200 random "
              f"points). The grid found a config that fires only a few times — KC-3 "
              f"correctly REFUSES to certify because <50 trials is statistically insufficient. "
              f"This is Omega-Lock working as designed: for real targets, tune your param space "
              f"so neutrals fire >=50 times, OR relax KCThresholds.trade_count_min for toys.")


def main() -> int:
    pattern_1_callable()
    pattern_2_stateful()
    print("\nAdapter example complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
