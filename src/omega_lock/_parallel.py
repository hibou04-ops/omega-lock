# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Internal, order-preserving evaluation map.

This is the single load-bearing primitive behind the optional, default-off
parallel-execution seam threaded through the grid / stress / walk-forward
evaluation loops. It is deliberately NOT part of the public API
(``omega_lock.__all__``); consumers opt in only by passing a
``concurrent.futures.Executor`` to the public entry points.

Correctness contract (why this exists):
    Several downstream computations depend on results being in INPUT order,
    not completion order -- e.g. the Pearson pairing of train/test fitness
    vectors in walk-forward, and the float-sum order of the stress z-score
    normalization. Reassembling by completion order (``as_completed``) would
    silently reorder those vectors and change results. We therefore index by
    submission order and read ``Future.result()`` back in that same order.

    When ``executor is None`` we run strictly serially -- a plain list
    comprehension -- so the default path is byte-identical to the pre-seam
    behavior (same call order, same accumulation order).
"""
from __future__ import annotations

from concurrent.futures import Executor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _ordered_eval_map(
    executor: Executor | None,
    items: list[T],
    fn: Callable[[T], R],
) -> list[R]:
    """Apply ``fn`` to each item, returning results in INPUT order.

    Args:
        executor: a ``concurrent.futures.Executor`` to dispatch through, or
            ``None`` for serial evaluation (the default everywhere).
        items: input items, evaluated in order.
        fn: the per-item function.

    Returns:
        ``[fn(items[0]), fn(items[1]), ...]`` -- always in input order,
        regardless of the order in which an executor completes the tasks.

    With ``executor=None`` this is exactly ``[fn(x) for x in items]``; the
    serial path is preserved verbatim so default behavior never changes.
    Empty and single-item inputs are handled by both branches without
    special-casing.
    """
    if executor is None:
        return [fn(x) for x in items]
    # Submit in input order; index is implicit in the list position.
    futures = [executor.submit(fn, x) for x in items]
    # Read back in submission order -- NOT as_completed -- so the returned
    # list is input-ordered even when tasks finish out of order.
    return [f.result() for f in futures]
