# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""render_scorecard — text renderer for AuditReport.

Produces a terminal-friendly, markdown-compatible scorecard. Sections:
  1. Identity (method, version, seed, duration)
  2. Totals (runs, feasibility)
  3. Constraints (pass/fail counts)
  4. Best feasible + best any
  5. Stress ranking (if available)
  6. Phase/round breakdown
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from omega_lock.audit._types import AuditReport, AuditedRun


def _fmt_duration(started_iso: str, ended_iso: str) -> str:
    try:
        t0 = datetime.fromisoformat(started_iso)
        t1 = datetime.fromisoformat(ended_iso)
        secs = (t1 - t0).total_seconds()
    except ValueError:
        return "?"
    if secs < 1:
        return f"{secs*1000:.0f}ms"
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(secs, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_params(params: dict, max_items: int = 6) -> str:
    items = list(params.items())
    head = items[:max_items]
    rest = len(items) - len(head)
    body = ", ".join(f"{k}={_fmt_val(v)}" for k, v in head)
    return body + (f", ... (+{rest} more)" if rest > 0 else "")


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return repr(v)


def _phase_role_breakdown(runs: tuple[AuditedRun, ...]) -> list[tuple[str, str, int]]:
    """Group by (phase, role), preserving first-seen order."""
    order: list[tuple[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for r in runs:
        key = (r.phase, r.target_role)
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1
    return [(p, r, counts[(p, r)]) for (p, r) in order]


def render_scorecard(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("=== omega-lock audit report ===")
    lines.append(
        f"Method:       {report.method} (omega-lock {report.omega_lock_version})"
    )
    seed_str = "none" if report.seed is None else str(report.seed)
    lines.append(f"Seed:         {seed_str}")
    lines.append(
        f"Duration:     {_fmt_duration(report.started_iso, report.ended_iso)} "
        f"({report.started_iso} -> {report.ended_iso})"
    )
    lines.append(f"Total runs:   {report.n_total}")
    lines.append(
        f"Feasible:     {report.n_feasible} / {report.n_total} "
        f"({report.feasibility_rate*100:.1f}%)"
    )

    if report.constraints:
        lines.append("")
        lines.append("Constraints:")
        pass_counts = report.constraint_pass_counts()
        for c in report.constraints:
            passed = pass_counts.get(c.name, 0)
            lines.append(
                f"  [{'PASS' if passed == report.n_total else 'MIXED'}] "
                f"{c.name:30s} {passed}/{report.n_total}"
                + (f"  -- {c.description}" if c.description else "")
            )

    bf = report.best_feasible
    ba = report.best_any
    lines.append("")
    if bf is not None:
        lines.append("Best feasible:")
        lines.append(f"  params:   {_fmt_params(bf.params)}")
        lines.append(f"  fitness:  {bf.fitness:.6g}   n_trials: {bf.n_trials}")
        lines.append(f"  phase:    {bf.phase}   role: {bf.target_role}   round: {bf.round_index}")
    else:
        lines.append("Best feasible: (none - all runs violated constraints)")

    if ba is not None and (bf is None or ba.call_index != bf.call_index):
        lines.append("")
        label = "Best (any):"
        feas = "FEASIBLE" if ba.is_feasible else "INFEASIBLE"
        violates = (
            ""
            if ba.is_feasible else
            f" - failed: {', '.join(ba.constraints_failed)}"
        )
        lines.append(label)
        lines.append(f"  fitness:  {ba.fitness:.6g} ({feas}{violates})")
        lines.append(f"  params:   {_fmt_params(ba.params)}")

    if report.stress_ranking:
        lines.append("")
        lines.append("Stress ranking (top 5):")
        for i, (name, score) in enumerate(list(report.stress_ranking)[:5], 1):
            lines.append(f"  {i}. {name:30s} {score:.4g}")

    breakdown = _phase_role_breakdown(report.runs)
    if breakdown:
        lines.append("")
        lines.append("Trail breakdown (phase/role):")
        for phase, role, n in breakdown:
            lines.append(f"  {phase:15s} {role:12s} {n:5d} runs")

    return "\n".join(lines) + "\n"
