# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Walk-forward validation helpers.

Two responsibilities:
    1. Pearson correlation between train and test fitness vectors
    2. A WalkForward harness that takes "train-eval" and "test-eval" targets
       (often the same target with different data slices) and runs KC-4.

The split itself is target-specific (time-series slice, fold index, etc.)
and is NOT part of this module — the caller constructs the two targets.
"""
from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from omega_lock._parallel import _ordered_eval_map
from omega_lock.grid import GridPoint
from omega_lock.target import CalibrableTarget, EvalResult

if TYPE_CHECKING:
    from omega_lock.kill_criteria import KCThresholds


# Single source of truth for the walk-forward top-N (KC-4 grid points).
# Lives here (lowest in the import graph that all three search configs
# already depend on) so P1Config, IterativeConfig (orchestrator.py) and
# P2Config (p2_tpe.py) reference ONE constant — drift is impossible by
# construction. Raised 10 -> 20: a larger walk-forward sample lowers the
# false-PASS rate of the KC-4a Pearson gate (the 2.0.2 perf gains buy the
# affordability of this cheap statistical INPUT — see threshold recalibration
# plan T1/D3). This is an input-size tightening, not a gate-threshold move.
DEFAULT_WALK_FORWARD_TOP_N = 20


def assert_strict_bundle(
    kc_thresholds: "KCThresholds", walk_forward_top_n: int
) -> None:
    """Fail-loud guard for the two-object strict bundle (plan T3/D2).

    ``KCThresholds.strict()`` raises the KC-4a Pearson floor to 0.5. That
    lower false-PASS rate is only *earned* when the walk-forward gate is fed
    enough points — otherwise the tighter floor is applied to a sample too
    small to support it, which is exactly the kind of silent half-applied
    tightening this framework exists to prevent.

    So the strict KCThresholds value (pearson_min >= 0.5) and the strict
    search input (walk_forward_top_n >= DEFAULT_WALK_FORWARD_TOP_N) MUST
    travel together. If only one is present, raise ValueError at the call
    site (run_p1 / run_p2_tpe) before any work is done, rather than leaving
    the mismatch to documentation.

    A non-strict thresholds object (pearson_min < 0.5 — including the bare
    default 0.3) never trips this guard regardless of walk_forward_top_n.
    """
    if kc_thresholds.pearson_min >= 0.5 and walk_forward_top_n < DEFAULT_WALK_FORWARD_TOP_N:
        raise ValueError(
            "strict KC-4a bundle is half-applied: pearson_min="
            f"{kc_thresholds.pearson_min} (>= 0.5, e.g. KCThresholds.strict()) "
            f"requires walk_forward_top_n >= {DEFAULT_WALK_FORWARD_TOP_N} to earn "
            f"the lower false-PASS rate, but got walk_forward_top_n="
            f"{walk_forward_top_n}. Raise walk_forward_top_n to "
            f"{DEFAULT_WALK_FORWARD_TOP_N} (the default) or drop pearson_min "
            "below 0.5. The strict Pearson floor and the strict sample size "
            "must travel together."
        )


PearsonStatus = Literal[
    "OK",
    "EMPTY",
    "LENGTH_MISMATCH",
    "ZERO_VARIANCE_X",
    "ZERO_VARIANCE_Y",
]


@dataclass(frozen=True)
class PearsonResult:
    """Structured Pearson outcome.

    Reviewer P1: ``pearson()`` previously collapsed every failure mode
    to ``0.0`` — a degenerate "could not compute" was indistinguishable
    from a measured "no correlation". Both situations failed KC-4 by
    falling under ``pearson_min``, but the artifact reader couldn't tell
    whether the run produced bad evidence or no evidence.

    ``value`` is None when status != OK; the caller must inspect status
    before comparing against a numeric threshold.
    """

    value: float | None
    status: PearsonStatus

    @property
    def computable(self) -> bool:
        return self.status == "OK"


def pearson_result(xs: list[float], ys: list[float]) -> PearsonResult:
    """Structured Pearson with explicit degeneracy categories."""
    n = len(xs)
    if n == 0:
        return PearsonResult(value=None, status="EMPTY")
    if n != len(ys):
        return PearsonResult(value=None, status="LENGTH_MISMATCH")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if dx == 0:
        return PearsonResult(value=None, status="ZERO_VARIANCE_X")
    if dy == 0:
        return PearsonResult(value=None, status="ZERO_VARIANCE_Y")
    return PearsonResult(value=num / (dx * dy), status="OK")


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input.

    Backward-compatible thin wrapper around ``pearson_result`` — kept so
    existing callers (and downstream artifacts) read the same number
    they did before. Use ``pearson_result`` directly when you need to
    distinguish "uncomputable" from "uncorrelated".
    """
    pr = pearson_result(xs, ys)
    return pr.value if pr.value is not None else 0.0


@dataclass
class WalkForwardResult:
    top_n: int
    train_fitnesses: list[float]
    test_fitnesses: list[float]
    test_n_trials: list[int]
    pearson: float
    pearson_status: PearsonStatus
    pearson_computable: bool
    train_best_trades_mean: float
    test_best_trades: int
    test_best_fitness: float           # cached from the same evaluation as test_best_trades
    test_best_params: dict[str, Any]   # params of the train-best candidate (top[0])
    trade_ratio_scaled: float      # test_best_trades / (train_mean_trades * scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "train_fitnesses": list(self.train_fitnesses),
            "test_fitnesses": list(self.test_fitnesses),
            "test_n_trials": list(self.test_n_trials),
            "pearson": self.pearson,
            "pearson_status": self.pearson_status,
            "pearson_computable": self.pearson_computable,
            "train_best_trades_mean": self.train_best_trades_mean,
            "test_best_trades": self.test_best_trades,
            "test_best_fitness": self.test_best_fitness,
            "test_best_params": dict(self.test_best_params),
            "trade_ratio_scaled": self.trade_ratio_scaled,
        }


@dataclass
class WalkForward:
    """Re-evaluate train-best N grid points on a test target.

    Each top-N candidate is evaluated on ``test_target`` exactly once;
    the train-best candidate's metrics (fitness AND trade count) come
    from the same evaluation, so a stochastic test_target cannot produce
    an internally inconsistent ``WalkForwardResult`` where
    ``test_fitnesses[0]`` and ``test_best_trades`` come from different
    runs of the same params.

    Typical usage:
        # Caller has trained on `train_target`, now verify on `test_target`
        wf = WalkForward(test_target=test_target)
        wf_result = wf.run(train_grid=grid_points_from_train, top_n=10)
        # Then feed wf_result.pearson + wf_result.trade_ratio_scaled to check_kc4()
    """
    test_target: CalibrableTarget
    trade_ratio_scale: float = 1.0       # e.g. len(test) / len(train) for time series

    def run(
        self,
        train_grid: list[GridPoint],
        top_n: int = DEFAULT_WALK_FORWARD_TOP_N,
        *,
        executor: Executor | None = None,
    ) -> WalkForwardResult:
        """Re-evaluate the train-best ``top_n`` candidates on the test target.

        ``executor`` is an optional, default-off
        ``concurrent.futures.Executor`` seam. When provided, the per-candidate
        test evaluations are dispatched through it and reassembled in INPUT
        order — load-bearing because the train/test fitness vectors are
        Pearson-paired positionally. ``None`` (the default) => strictly serial,
        byte-identical to prior behavior.
        """
        if not train_grid:
            raise ValueError("train_grid is empty")
        ranked = sorted(train_grid, key=lambda p: p.result.fitness, reverse=True)
        top = ranked[:top_n]

        # Evaluate each top candidate on the test target ONCE and cache
        # the EvalResult. Pre-fix: the train-best candidate (top[0]) was
        # evaluated twice — once in this loop for the fitness vector and
        # again below for trade-count. Stochastic test_targets produced
        # an artifact where the "test fitness of best" and "test trade
        # count of best" came from different runs.
        test_results: list[EvalResult] = _ordered_eval_map(
            executor, top, lambda gp: self.test_target.evaluate(gp.params)
        )
        train_fs: list[float] = []
        test_fs: list[float] = []
        test_trials: list[int] = []
        for gp, r in zip(top, test_results):
            train_fs.append(gp.result.fitness)
            test_fs.append(r.fitness)
            test_trials.append(r.sample_count)

        best_on_train = top[0]
        test_best = test_results[0]  # SAME evaluation as test_fs[0] / test_trials[0]
        train_trades_mean = sum(gp.result.sample_count for gp in top) / len(top)
        trade_ratio = (
            test_best.sample_count / (train_trades_mean * self.trade_ratio_scale)
            if (train_trades_mean > 0 and self.trade_ratio_scale > 0)
            else 0.0
        )
        pearson_outcome = pearson_result(train_fs, test_fs)
        pearson_value = (
            pearson_outcome.value if pearson_outcome.value is not None else 0.0
        )

        return WalkForwardResult(
            top_n=len(top),
            train_fitnesses=train_fs,
            test_fitnesses=test_fs,
            test_n_trials=test_trials,
            pearson=pearson_value,
            pearson_status=pearson_outcome.status,
            pearson_computable=pearson_outcome.computable,
            train_best_trades_mean=train_trades_mean,
            test_best_trades=test_best.sample_count,
            test_best_fitness=test_best.fitness,
            test_best_params=dict(best_on_train.params),
            trade_ratio_scaled=trade_ratio,
        )
