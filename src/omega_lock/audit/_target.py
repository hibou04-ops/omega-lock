# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""AuditingTarget — transparent CalibrableTarget decorator.

Every call to `evaluate()` is recorded to a trail with full positional
context (phase, role, round_index, call_index). The wrapped target is
indistinguishable from the inner target to any optimizer that only
uses the CalibrableTarget Protocol (param_space + evaluate).

Multi-target audit:
    When run_p1 uses train/test/holdout together, create one wrapper
    per target and pass the same list as `shared_trail` to all of them.
    The trail stays ordered by call_index (globally monotonic across
    wrappers).

Memory:
    EvalResult.artifacts is designed to hold large/binary payloads
    (see target.py). We drop it from the trail by default. Set
    `retain_artifacts=True` to keep them, at your own memory risk.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from typing import Any, Sequence

from omega_lock.audit._types import AuditedRun, Constraint
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


class AuditingTarget:
    """Decorator that records every evaluate() call into an audit trail.

    Impersonates CalibrableTarget via param_space() + evaluate() — so
    GridSearch, RandomSearch, run_p2_tpe, and any CallableAdapter work
    unchanged.

    Args:
        inner: the underlying target; all evaluations delegate to it.
        constraints: hard constraints applied to every call. Violations
            are recorded but never block the call.
        target_role: "train" | "test" | "validation" | "holdout" | "custom".
            Lets a single trail cover multi-target run_p1 calls cleanly.
        retain_artifacts: if True, EvalResult.artifacts is copied into
            AuditedRun.metadata under key "_artifacts". Default False to
            keep the trail memory-safe.
        shared_trail: an externally-owned list to append AuditedRuns into.
            Use this to stitch train/test/holdout wrappers into one
            ordered trail. If None, a fresh list is created.
        shared_counter: an externally-owned itertools.count for call_index.
            Use with shared_trail when multiple wrappers share a trail so
            call_index stays globally monotonic. If None, a fresh counter.
    """

    def __init__(
        self,
        inner: CalibrableTarget,
        constraints: Sequence[Constraint] = (),
        target_role: str = "train",
        retain_artifacts: bool = False,
        shared_trail: list[AuditedRun] | None = None,
        shared_counter: "count[int] | None" = None,
    ) -> None:
        self.inner = inner
        self.constraints: tuple[Constraint, ...] = tuple(constraints)
        self.target_role = target_role
        self.retain_artifacts = retain_artifacts
        self.trail: list[AuditedRun] = shared_trail if shared_trail is not None else []
        self._counter = shared_counter if shared_counter is not None else count(0)
        self._phase = "search"
        self._round_index = 0

    # ── Phase / round control ──────────────────────────────────────────────
    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def set_round(self, round_index: int) -> None:
        self._round_index = round_index

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def round_index(self) -> int:
        return self._round_index

    # ── CalibrableTarget Protocol ──────────────────────────────────────────
    def param_space(self) -> list[ParamSpec]:
        return self.inner.param_space()

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        r = self.inner.evaluate(params)
        passed: list[str] = []
        failed: list[str] = []
        for c in self.constraints:
            try:
                ok = bool(c.fn(params, r))
            except Exception:
                # A raising predicate counts as FAIL — audit stays running.
                ok = False
            (passed if ok else failed).append(c.name)

        metadata = dict(r.metadata)
        if self.retain_artifacts and r.artifacts:
            metadata["_artifacts"] = dict(r.artifacts)

        run = AuditedRun(
            params=dict(params),
            fitness=float(r.fitness),
            n_trials=int(r.n_trials),
            metadata=metadata,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            constraints_passed=tuple(passed),
            constraints_failed=tuple(failed),
            phase=self._phase,
            call_index=next(self._counter),
            target_role=self.target_role,
            round_index=self._round_index,
        )
        self.trail.append(run)
        return r
