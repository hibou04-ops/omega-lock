"""Audit data types — Constraint, AuditedRun, AuditReport.

Design:
    * Constraint: a named predicate over (params, EvalResult) -> bool.
      Hard-only in v1. Returning False = violated.
    * AuditedRun: one evaluate() call, fully positioned with phase/role/round.
    * AuditReport: the full append-only trail + summary + JSON roundtrip.

The AuditReport reuses the EvalResult serialization pattern from
orchestrator.py (_eval_to_dict + _json_fallback) — artifacts are
deliberately dropped to keep the trail small.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

from omega_lock.target import EvalResult, ParamSpec


# Constraint predicate: True = pass, False = violated.
ConstraintFn = Callable[[dict[str, Any], EvalResult], bool]


@dataclass(frozen=True)
class Constraint:
    """A named hard constraint on (params, EvalResult).

    The predicate returns True when the result satisfies the constraint.
    Soft penalties are deliberately out of scope for v1 — mix a
    custom fitness transform into your CalibrableTarget if you need one.
    """
    name: str
    fn: ConstraintFn
    description: str = ""


@dataclass(frozen=True)
class AuditedRun:
    """One evaluate() call, captured with full positional context.

    Fields:
        params: the full param dict passed to evaluate()
        fitness: r.fitness
        n_trials: r.n_trials
        metadata: dict(r.metadata), best-effort JSON-safe
        timestamp_iso: UTC ISO-8601
        constraints_passed / failed: names of constraints evaluated
        phase: "baseline" | "stress" | "search" | "walk_forward" | "holdout" | "custom"
        call_index: monotonic within a single audit session
        target_role: "train" | "test" | "validation" | "holdout" | "custom"
        round_index: 0 for non-iterative; increments for run_p1_iterative
    """
    params: dict[str, Any]
    fitness: float
    n_trials: int
    metadata: dict[str, Any]
    timestamp_iso: str
    constraints_passed: tuple[str, ...]
    constraints_failed: tuple[str, ...]
    phase: str
    call_index: int
    target_role: str
    round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self.params),
            "fitness": self.fitness,
            "n_trials": self.n_trials,
            "metadata": dict(self.metadata),
            "timestamp_iso": self.timestamp_iso,
            "constraints_passed": list(self.constraints_passed),
            "constraints_failed": list(self.constraints_failed),
            "phase": self.phase,
            "call_index": self.call_index,
            "target_role": self.target_role,
            "round_index": self.round_index,
        }

    @property
    def is_feasible(self) -> bool:
        """True if no constraint failed. Vacuously True when no constraints."""
        return len(self.constraints_failed) == 0


def _json_fallback(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


@dataclass
class AuditReport:
    """Full audit trail + summary for a calibration session.

    runs is append-only. best_feasible/best_any are computed on-demand
    (not cached) so the report stays honest under filtering.
    """
    method: str
    omega_lock_version: str
    seed: int | None
    started_iso: str
    ended_iso: str

    constraints: tuple[Constraint, ...]
    runs: tuple[AuditedRun, ...]

    stress_ranking: tuple[tuple[str, float], ...] | None = None

    @property
    def n_total(self) -> int:
        return len(self.runs)

    @property
    def n_feasible(self) -> int:
        return sum(1 for r in self.runs if r.is_feasible)

    @property
    def feasibility_rate(self) -> float:
        return self.n_feasible / self.n_total if self.n_total else 0.0

    @property
    def best_feasible(self) -> AuditedRun | None:
        feas = [r for r in self.runs if r.is_feasible]
        return max(feas, key=lambda r: r.fitness) if feas else None

    @property
    def best_any(self) -> AuditedRun | None:
        return max(self.runs, key=lambda r: r.fitness) if self.runs else None

    def by_phase(self, phase: str) -> tuple[AuditedRun, ...]:
        return tuple(r for r in self.runs if r.phase == phase)

    def by_role(self, role: str) -> tuple[AuditedRun, ...]:
        return tuple(r for r in self.runs if r.target_role == role)

    def by_round(self, round_index: int) -> tuple[AuditedRun, ...]:
        return tuple(r for r in self.runs if r.round_index == round_index)

    def constraint_pass_counts(self) -> dict[str, int]:
        """For each constraint, how many runs passed it."""
        counts = {c.name: 0 for c in self.constraints}
        for r in self.runs:
            for name in r.constraints_passed:
                if name in counts:
                    counts[name] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "omega_lock_version": self.omega_lock_version,
            "seed": self.seed,
            "started_iso": self.started_iso,
            "ended_iso": self.ended_iso,
            "constraints": [
                {"name": c.name, "description": c.description} for c in self.constraints
            ],
            "runs": [r.to_dict() for r in self.runs],
            "stress_ranking": (
                [list(pair) for pair in self.stress_ranking]
                if self.stress_ranking is not None else None
            ),
            "summary": {
                "n_total": self.n_total,
                "n_feasible": self.n_feasible,
                "feasibility_rate": self.feasibility_rate,
                "constraint_pass_counts": self.constraint_pass_counts(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_fallback)

    @classmethod
    def from_json(cls, s: str) -> "AuditReport":
        """Rehydrate a report. Constraint predicates are NOT restored — we keep
        only names + descriptions, since function objects can't round-trip."""
        d = json.loads(s)
        constraints = tuple(
            Constraint(name=c["name"], fn=_unavailable_predicate, description=c.get("description", ""))
            for c in d.get("constraints", [])
        )
        runs = tuple(
            AuditedRun(
                params=r["params"],
                fitness=r["fitness"],
                n_trials=r["n_trials"],
                metadata=r.get("metadata", {}),
                timestamp_iso=r["timestamp_iso"],
                constraints_passed=tuple(r.get("constraints_passed", [])),
                constraints_failed=tuple(r.get("constraints_failed", [])),
                phase=r["phase"],
                call_index=r["call_index"],
                target_role=r["target_role"],
                round_index=r.get("round_index", 0),
            )
            for r in d.get("runs", [])
        )
        sr_raw = d.get("stress_ranking")
        stress_ranking = (
            tuple((pair[0], float(pair[1])) for pair in sr_raw)
            if sr_raw is not None else None
        )
        return cls(
            method=d["method"],
            omega_lock_version=d["omega_lock_version"],
            seed=d.get("seed"),
            started_iso=d["started_iso"],
            ended_iso=d["ended_iso"],
            constraints=constraints,
            runs=runs,
            stress_ranking=stress_ranking,
        )


def _unavailable_predicate(params: dict[str, Any], result: EvalResult) -> bool:
    """Sentinel for rehydrated constraints — calling it raises. Rehydrated
    reports are for inspection only; to re-run constraint checks, the caller
    must re-supply the original predicates."""
    raise RuntimeError(
        "Constraint predicates are not serialized. Re-supply original Constraint "
        "objects if you need to re-evaluate constraints against trail runs."
    )
