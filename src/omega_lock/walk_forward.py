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
from math import fsum, isfinite, sqrt
from typing import Any, Literal

from omega_lock._parallel import _ordered_eval_map
from omega_lock.grid import GridPoint
from omega_lock.target import CalibrableTarget, EvalResult


PearsonStatus = Literal[
    "OK",
    "EMPTY",
    "LENGTH_MISMATCH",
    "ZERO_VARIANCE_X",
    "ZERO_VARIANCE_Y",
    "NON_FINITE_X",
    "NON_FINITE_Y",
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


def _center_scaled(values: list[float]) -> list[float]:
    """Translate before scaling to preserve representable small differences."""
    origin = values[0]
    shifted = [value - origin for value in values]
    if not all(isfinite(value) for value in shifted):
        # Opposite-sign finite extremes can overflow during subtraction.
        scale = max(abs(value) for value in values)
        shifted = [value / scale for value in values]
    scale = max(abs(value) for value in shifted)
    if scale == 0:
        return [0.0] * len(values)
    normalized = [value / scale for value in shifted]
    mean = fsum(normalized) / len(values)
    return [value - mean for value in normalized]


def pearson_result(xs: list[float], ys: list[float]) -> PearsonResult:
    """Pearson with finite-input checks and overflow/underflow-safe scaling.

    This measures linear association, not rank correlation or absolute quality.
    Constant inputs remain uncomputable; nonconstant representable differences
    are retained even with large offsets or subnormal magnitudes.
    """
    n = len(xs)
    if n == 0:
        return PearsonResult(value=None, status="EMPTY")
    if n != len(ys):
        return PearsonResult(value=None, status="LENGTH_MISMATCH")
    if not all(isfinite(value) for value in xs):
        return PearsonResult(value=None, status="NON_FINITE_X")
    if not all(isfinite(value) for value in ys):
        return PearsonResult(value=None, status="NON_FINITE_Y")
    centered_x = _center_scaled(xs)
    centered_y = _center_scaled(ys)
    sum_x2 = fsum(x * x for x in centered_x)
    sum_y2 = fsum(y * y for y in centered_y)
    if sum_x2 == 0:
        return PearsonResult(value=None, status="ZERO_VARIANCE_X")
    if sum_y2 == 0:
        return PearsonResult(value=None, status="ZERO_VARIANCE_Y")
    num = fsum(x * y for x, y in zip(centered_x, centered_y))
    # Roundoff can otherwise put a perfect association just outside [-1, 1].
    return PearsonResult(value=max(-1.0, min(1.0, num / sqrt(sum_x2 * sum_y2))), status="OK")


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
        top_n: int = 10,
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
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
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
