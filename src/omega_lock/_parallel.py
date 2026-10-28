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

from collections import deque
from concurrent.futures import Executor, Future
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Bound submitted-but-not-consumed work independently of the grid size.
_MAX_PENDING_EVALUATIONS = 64


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

    Parallel mode keeps at most 64 submitted, unconsumed futures. Targets
    must be safe for concurrent calls; shared/global RNG or mutable state
    can make serial and parallel observations differ. Public call sites use
    local closures, so use a ThreadPoolExecutor (or a compatible in-process
    executor), not a process pool requiring picklable callables.
    """
    if executor is None:
        return [fn(x) for x in items]
    pending: deque[Future[R]] = deque()
    results: list[R] = []
    source = iter(items)
    try:
        for _ in range(min(len(items), _MAX_PENDING_EVALUATIONS)):
            pending.append(executor.submit(fn, next(source)))
        # Consume in input order, refilling only after one result is consumed.
        # Results remain O(N), but the live Future queue is at most 64.
        for item in source:
            results.append(pending.popleft().result())
            pending.append(executor.submit(fn, item))
        results.extend(future.result() for future in pending)
        return results
    except BaseException:
        # Do not own/shut down the caller's executor. Cancel queued work on a
        # best-effort basis; tasks already running may still finish.
        for future in pending:
            future.cancel()
        raise
