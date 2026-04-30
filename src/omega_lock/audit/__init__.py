# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""omega_lock.audit — method-agnostic audit surface for calibration runs.

Public API:
    from omega_lock.audit import (
        AuditingTarget, Constraint, AuditedRun, AuditReport,
        render_scorecard, make_report,
    )

Usage (single target):
    from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard
    from omega_lock import run_p1, P1Config

    constraints = [
        Constraint("fitness_gt_0", lambda p, r: r.fitness > 0.0,
                   "Fitness must be positive"),
    ]
    wrapped = AuditingTarget(my_target, constraints=constraints)
    wrapped.set_phase("search")
    result = run_p1(wrapped, config=P1Config())
    report = make_report(wrapped, method="p1", seed=42)
    print(render_scorecard(report))

Usage (multi-target train/test/holdout):
    from itertools import count
    from omega_lock.audit import AuditingTarget, Constraint, make_report

    trail, counter = [], count(0)
    wtrain   = AuditingTarget(train, constraints, "train",  shared_trail=trail, shared_counter=counter)
    wtest    = AuditingTarget(test,  constraints, "test",   shared_trail=trail, shared_counter=counter)
    wholdout = AuditingTarget(ho,    constraints, "holdout",shared_trail=trail, shared_counter=counter)
    result = run_p1(wtrain, test_target=wtest, holdout_target=wholdout, config=P1Config())
    report = make_report(wtrain, method="p1", seed=42)   # any wrapper works — trail is shared
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from omega_lock.audit._types import (
    AuditedRun,
    AuditReport,
    Constraint,
    ConstraintFn,
)
from omega_lock.audit._target import AuditingTarget
from omega_lock.audit._scorecard import render_scorecard


def make_report(
    source: AuditingTarget,
    method: str,
    *,
    seed: int | None = None,
    started_iso: str | None = None,
    ended_iso: str | None = None,
    stress_ranking: Iterable[tuple[str, float]] | None = None,
) -> AuditReport:
    """Freeze an AuditingTarget's trail into an AuditReport.

    Args:
        source: any AuditingTarget — its `.trail` is used (works for
            shared trails — wrappers share the same list, so passing any
            one of them gives the full trail).
        method: human-readable method tag ("p1", "p2_tpe", "plain_grid", ...).
        seed: optional RNG seed for the method.
        started_iso / ended_iso: if omitted, falls back to the trail's first
            and last timestamps, or `now()` if the trail is empty.
        stress_ranking: optional (name, score) tuples ordered high -> low.
    """
    from omega_lock import __version__ as _version

    runs = tuple(source.trail)
    now_iso = datetime.now(timezone.utc).isoformat()
    if started_iso is None:
        started_iso = runs[0].timestamp_iso if runs else now_iso
    if ended_iso is None:
        ended_iso = runs[-1].timestamp_iso if runs else now_iso

    return AuditReport(
        method=method,
        omega_lock_version=_version,
        seed=seed,
        started_iso=started_iso,
        ended_iso=ended_iso,
        constraints=tuple(source.constraints),
        runs=runs,
        stress_ranking=(tuple(stress_ranking) if stress_ranking is not None else None),
    )


__all__ = [
    "AuditingTarget",
    "Constraint",
    "ConstraintFn",
    "AuditedRun",
    "AuditReport",
    "render_scorecard",
    "make_report",
]
