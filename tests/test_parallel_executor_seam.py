# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for the dormant, default-off parallel-execution seam.

Two layers:
  * the internal ``_ordered_eval_map`` primitive (order preservation under
    out-of-order completion, executor==serial determinism, empty/single
    edge cases);
  * the four public call sites threaded with ``executor=`` -- each must
    return results IDENTICAL to its default serial path.

The load-bearing property is INPUT-ORDER reassembly: walk-forward Pearson
pairing and stress z-score normalization both depend on it. We prove it with
a real ``ThreadPoolExecutor`` and descending per-task delays so item 0
finishes LAST yet must land FIRST in the output.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omega_lock._parallel import _ordered_eval_map
from omega_lock.grid import GridPoint, GridSearch, ZoomingGridSearch
from omega_lock.stress import StressOptions, measure_stress
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.walk_forward import WalkForward


# --------------------------------------------------------------------------- #
# _ordered_eval_map: primitive-level
# --------------------------------------------------------------------------- #
def test_serial_none_executor_matches_list_comprehension():
    items = [1, 2, 3, 4]
    assert _ordered_eval_map(None, items, lambda x: x * x) == [1, 4, 9, 16]


def test_empty_input_serial_and_parallel():
    assert _ordered_eval_map(None, [], lambda x: x) == []
    with ThreadPoolExecutor(max_workers=2) as ex:
        assert _ordered_eval_map(ex, [], lambda x: x) == []


def test_single_item_serial_and_parallel():
    assert _ordered_eval_map(None, [42], lambda x: x + 1) == [43]
    with ThreadPoolExecutor(max_workers=2) as ex:
        assert _ordered_eval_map(ex, [42], lambda x: x + 1) == [43]


def test_results_in_input_order_despite_shuffled_completion():
    # Descending delays: item 0 sleeps longest -> completes LAST. A
    # completion-order reassembly (as_completed) would return [.., .., 0];
    # _ordered_eval_map must return strict input order.
    items = [0, 1, 2, 3, 4]
    completion_order: list[int] = []
    lock = threading.Lock()

    def slow(x: int) -> int:
        time.sleep((len(items) - x) * 0.02)
        with lock:
            completion_order.append(x)
        return x * 10

    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        out = _ordered_eval_map(ex, items, slow)

    assert out == [0, 10, 20, 30, 40]            # input order preserved
    assert completion_order != items             # completion really was shuffled
    assert completion_order[0] != 0              # item 0 did NOT finish first


def test_executor_result_equals_serial_result():
    items = list(range(20))

    def fn(x: int) -> int:
        return (x * 7) % 13

    serial = _ordered_eval_map(None, items, fn)
    with ThreadPoolExecutor(max_workers=4) as ex:
        parallel = _ordered_eval_map(ex, items, fn)
    assert parallel == serial


# --------------------------------------------------------------------------- #
# Call-site target
# --------------------------------------------------------------------------- #
class QuadraticTarget:
    """f(x, y) = -(x - 3)^2 - (y - 7)^2."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="y", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x, y = params["x"], params["y"]
        return EvalResult(fitness=-((x - 3.0) ** 2 + (y - 7.0) ** 2), n_trials=1)


def _grid_signature(points: list[GridPoint]) -> list[tuple[int, dict[str, Any], float, int]]:
    return [
        (p.idx, dict(p.unlocked), p.result.fitness, p.result.sample_count)
        for p in points
    ]


# --------------------------------------------------------------------------- #
# GridSearch.run executor seam
# --------------------------------------------------------------------------- #
def test_gridsearch_executor_matches_serial():
    target = QuadraticTarget()
    gs = GridSearch(target=target, unlocked=["x", "y"], grid_points_per_axis=5, verbose=False)
    serial = gs.run(base_params={"x": 5.0, "y": 5.0})
    with ThreadPoolExecutor(max_workers=4) as ex:
        parallel = gs.run(base_params={"x": 5.0, "y": 5.0}, executor=ex)
    assert _grid_signature(parallel) == _grid_signature(serial)
    # idx must be a strict 0..N-1 input-order sequence, not completion order.
    assert [p.idx for p in parallel] == list(range(len(parallel)))


# --------------------------------------------------------------------------- #
# ZoomingGridSearch.run executor seam (within-round parallel; rounds serial)
# --------------------------------------------------------------------------- #
def test_zooming_executor_matches_serial():
    target = QuadraticTarget()
    zs = ZoomingGridSearch(
        target=target, unlocked=["x", "y"], grid_points_per_axis=4,
        zoom_rounds=3, zoom_factor=0.5, verbose=False,
    )
    serial = zs.run(base_params={"x": 5.0, "y": 5.0})
    with ThreadPoolExecutor(max_workers=4) as ex:
        parallel = zs.run(base_params={"x": 5.0, "y": 5.0}, executor=ex)
    assert _grid_signature(parallel) == _grid_signature(serial)
    # Global idx is contiguous across all rounds in input order: parallelizing
    # the within-round combos must not perturb the re-centering of later rounds.
    assert [p.idx for p in parallel] == list(range(len(parallel)))


# --------------------------------------------------------------------------- #
# measure_stress executor seam (z-score normalization is order-dependent)
# --------------------------------------------------------------------------- #
def _stress_signature(results: list[Any]) -> list[tuple[str, float, float]]:
    return [(r.name, r.raw_stress, r.normalized_stress) for r in results]


def test_measure_stress_executor_matches_serial():
    target = QuadraticTarget()
    base = {"x": 5.0, "y": 5.0}
    baseline = target.evaluate(base)
    opts = StressOptions(verbose=False)
    serial = measure_stress(target, base, baseline, options=opts)
    with ThreadPoolExecutor(max_workers=4) as ex:
        parallel = measure_stress(target, base, baseline, options=opts, executor=ex)
    assert _stress_signature(parallel) == _stress_signature(serial)


# --------------------------------------------------------------------------- #
# WalkForward.run executor seam (Pearson pairing depends on input order)
# --------------------------------------------------------------------------- #
def _make_train_grid(target: QuadraticTarget) -> list[GridPoint]:
    gs = GridSearch(target=target, unlocked=["x", "y"], grid_points_per_axis=4, verbose=False)
    return gs.run(base_params={"x": 5.0, "y": 5.0})


def test_walk_forward_executor_matches_serial():
    target = QuadraticTarget()
    grid = _make_train_grid(target)
    wf = WalkForward(test_target=target)
    serial = wf.run(grid, top_n=6).to_dict()
    with ThreadPoolExecutor(max_workers=4) as ex:
        parallel = wf.run(grid, top_n=6, executor=ex).to_dict()
    # Pearson, the paired fitness vectors, trade counts, and every other field
    # must be byte-equal to the serial computation.
    assert parallel["train_fitnesses"] == serial["train_fitnesses"]
    assert parallel["test_fitnesses"] == serial["test_fitnesses"]
    assert parallel["pearson"] == serial["pearson"]
    assert parallel == serial
