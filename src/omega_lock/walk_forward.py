# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Walk-forward validation helpers.

Two responsibilities:
    1. Pearson correlation between train and test fitness vectors
    2. A WalkForward harness that takes "train-eval" and "test-eval" targets
       (often the same target with different data slices) and runs KC-4.

The split itself is target-specific (time-series slice, fold index, etc.)
and is NOT part of this module ??the caller constructs the two targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega_lock.grid import GridPoint
from omega_lock.target import CalibrableTarget, EvalResult


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


@dataclass
class WalkForwardResult:
    top_n: int
    train_fitnesses: list[float]
    test_fitnesses: list[float]
    test_n_trials: list[int]
    pearson: float
    train_best_trades_mean: float
    test_best_trades: int
    trade_ratio_scaled: float      # test_best_trades / (train_mean_trades * scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "train_fitnesses": list(self.train_fitnesses),
            "test_fitnesses": list(self.test_fitnesses),
            "test_n_trials": list(self.test_n_trials),
            "pearson": self.pearson,
            "train_best_trades_mean": self.train_best_trades_mean,
            "test_best_trades": self.test_best_trades,
            "trade_ratio_scaled": self.trade_ratio_scaled,
        }


@dataclass
class WalkForward:
    """Re-evaluate train-best N grid points on a test target.

    Typical usage:
        # Caller has trained on `train_target`, now verify on `test_target`
        wf = WalkForward(test_target=test_target)
        wf_result = wf.run(train_grid=grid_points_from_train, top_n=10)
        # Then feed wf_result.pearson + wf_result.trade_ratio_scaled to check_kc4()
    """
    test_target: CalibrableTarget
    trade_ratio_scale: float = 1.0       # e.g. len(test) / len(train) for time series

    def run(self, train_grid: list[GridPoint], top_n: int = 10) -> WalkForwardResult:
        if not train_grid:
            raise ValueError("train_grid is empty")
        ranked = sorted(train_grid, key=lambda p: p.result.fitness, reverse=True)
        top = ranked[:top_n]

        train_fs: list[float] = []
        test_fs: list[float] = []
        test_trials: list[int] = []
        for gp in top:
            r = self.test_target.evaluate(gp.params)
            train_fs.append(gp.result.fitness)
            test_fs.append(r.fitness)
            test_trials.append(r.n_trials)

        best_on_train = top[0]
        test_best = self.test_target.evaluate(best_on_train.params)
        train_trades_mean = sum(gp.result.n_trials for gp in top) / len(top)
        trade_ratio = (
            test_best.n_trials / (train_trades_mean * self.trade_ratio_scale)
            if (train_trades_mean > 0 and self.trade_ratio_scale > 0)
            else 0.0
        )

        return WalkForwardResult(
            top_n=len(top),
            train_fitnesses=train_fs,
            test_fitnesses=test_fs,
            test_n_trials=test_trials,
            pearson=pearson(train_fs, test_fs),
            train_best_trades_mean=train_trades_mean,
            test_best_trades=test_best.n_trials,
            trade_ratio_scaled=trade_ratio,
        )
