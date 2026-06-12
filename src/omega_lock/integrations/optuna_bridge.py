# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Audit an EXISTING Optuna study with the omega-lock walk-forward gate.

`study.best_trial` is the highest score seen on the data the search
consumed — it says nothing about whether that candidate transfers or
respects hard constraints. `audit_optuna_study` bridges a completed
study into the same gates `run_p1` uses, without re-running the search:

    completed trials -> GridPoint list -> WalkForward (re-eval on holdout)
                                       -> KC-4 Pearson gate (check_kc4)
                                       -> best_any vs best_feasible split

The walk-forward and KC-4 math is REUSED from ``omega_lock.walk_forward``
and ``omega_lock.kill_criteria`` — this module deliberately contains no
duplicate gate arithmetic.

Two related surfaces, do not confuse them:
    * This bridge: audit an EXISTING study after the fact.
    * ``run_p2_tpe`` (optional ``[p2]`` extra): run a NEW search where
      Optuna TPE replaces the grid inside the full P1 gate pipeline.

Optuna is an optional dependency. ``import optuna`` happens lazily inside
``audit_optuna_study``; calling it without optuna installed raises a clean
ImportError with an install hint. Importing this module is always safe.

Feasibility inference
---------------------
Constraints are split (`best_any` vs `best_feasible`) only when they are
inferable from the study itself: a boolean ``user_attrs["feasible"]`` flag
on the trials (set it in your objective via
``trial.set_user_attr("feasible", ...)``). When no trial carries the flag,
the report documents ``feasibility_source="absent"`` and ``best_feasible``
is ``None`` — the bridge never guesses constraints it cannot see. Optuna's
sampler-level ``constraints_func`` storage is NOT consulted.

Direction handling
------------------
Minimize-direction studies are supported: ranking and best-selection
respect the study direction, and reported values keep the study's own
orientation. ``holdout_evaluate`` must return scores in the SAME
orientation as the study objective. Multi-objective studies are rejected.
"""
from __future__ import annotations

from importlib import import_module

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast

from omega_lock.grid import GridPoint
from omega_lock.kill_criteria import KCReport, KCThresholds, check_kc4
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.walk_forward import WalkForward

HoldoutEvaluate = Callable[[dict[str, Any]], float]

STUDY_AUDIT_SCHEMA_VERSION = "omega-lock.study-audit.v1"

# The user_attrs key consulted for per-trial feasibility flags.
FEASIBILITY_USER_ATTR = "feasible"

_OPTUNA_BRIDGE_INSTALL_HINT = (
    "audit_optuna_study requires optuna, which is not installed. "
    "Install it via `pip install \"omega-lock[p2]\"` or `pip install \"optuna>=3.0\"`."
)


@dataclass(frozen=True)
class TrialCandidate:
    """One completed trial, in the study's own value orientation.

    ``holdout_value`` is None when ``holdout_evaluate`` was not provided
    (or, for ``best_feasible`` only, could not be computed). ``feasible``
    is None when the study carries no feasibility flag for this trial.
    """

    number: int
    params: dict[str, Any]
    train_value: float
    holdout_value: float | None = None
    feasible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "params": dict(self.params),
            "train_value": self.train_value,
            "holdout_value": self.holdout_value,
            "feasible": self.feasible,
        }


@dataclass
class StudyAuditReport:
    """Gate verdict + candidate split for an audited Optuna study.

    Attributes:
        passed: True unless the KC-4 walk-forward gate FAILed. When no
            ``holdout_evaluate`` was provided the gate is SKIP (no
            transfer evidence either way) and ``passed`` stays True —
            read ``kc_report.status`` to distinguish PASS from SKIP.
        kc_report: the KC-4 report (PASS / FAIL / SKIP).
        best_any: highest-value completed trial, constraints ignored.
        best_feasible: highest-value trial whose ``user_attrs["feasible"]``
            flag is True; None when no feasible trial exists or when
            feasibility is not inferable (``feasibility_source="absent"``).
        gated_best: the candidate the gate is willing to certify — None
            when the gate FAILed, or when feasibility is inferable but no
            trial is feasible; otherwise ``best_feasible`` when available,
            else ``best_any``.
        candidates: the train-best top-N actually consulted by the gate,
            in train-rank order, with holdout values when evaluated.
        pearson: walk-forward Pearson over the top-N (None when the gate
            did not run or the correlation was not computable).
        feasibility_source: "user_attrs" or "absent".
    """

    passed: bool
    kc_report: KCReport
    best_any: TrialCandidate
    best_feasible: TrialCandidate | None
    gated_best: TrialCandidate | None
    candidates: list[TrialCandidate]
    pearson: float | None
    pearson_status: str
    feasibility_source: str
    holdout_evaluated: bool
    n_trials_total: int
    n_trials_completed: int
    top_n: int
    thresholds: KCThresholds = field(default_factory=KCThresholds.pure_objective)
    schema_version: str = STUDY_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "kc_report": {
                "name": self.kc_report.name,
                "status": self.kc_report.status,
                "message": self.kc_report.message,
                "detail": dict(self.kc_report.detail),
            },
            "best_any": self.best_any.to_dict(),
            "best_feasible": (
                self.best_feasible.to_dict() if self.best_feasible is not None else None
            ),
            "gated_best": (
                self.gated_best.to_dict() if self.gated_best is not None else None
            ),
            "candidates": [c.to_dict() for c in self.candidates],
            "pearson": self.pearson,
            "pearson_status": self.pearson_status,
            "feasibility_source": self.feasibility_source,
            "holdout_evaluated": self.holdout_evaluated,
            "n_trials_total": self.n_trials_total,
            "n_trials_completed": self.n_trials_completed,
            "top_n": self.top_n,
            "kc_thresholds": asdict(self.thresholds),
        }


class _HoldoutTarget:
    """Minimal CalibrableTarget over a user-supplied scoring callable.

    ``WalkForward.run`` only calls ``evaluate``; ``param_space`` exists to
    satisfy the protocol. ``sign`` maps the study's direction onto the
    maximize-orientation the gate machinery expects.
    """

    def __init__(self, fn: HoldoutEvaluate, sign: float) -> None:
        self._fn = fn
        self._sign = sign

    def param_space(self) -> list[ParamSpec]:  # pragma: no cover - protocol stub
        return []

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=self._sign * float(self._fn(dict(params))), sample_count=1)


def _trial_feasible_flag(trial: Any) -> bool | None:
    """Read the boolean user_attrs feasibility flag, if any."""
    attrs = getattr(trial, "user_attrs", None)
    if not isinstance(attrs, dict):
        return None
    flag = attrs.get(FEASIBILITY_USER_ATTR)
    return flag if isinstance(flag, bool) else None


def audit_optuna_study(
    study: Any,
    *,
    holdout_evaluate: HoldoutEvaluate | None = None,
    thresholds: KCThresholds | None = None,
    top_n: int = 10,
) -> StudyAuditReport:
    """Gate an existing Optuna study's completed trials.

    Args:
        study: an ``optuna.study.Study`` (single-objective).
        holdout_evaluate: optional ``(params: dict) -> float`` scored on a
            data slice the study never consumed, in the SAME orientation
            as the study objective. When provided, the train-best top-N
            trials are re-evaluated through it and the KC-4 walk-forward
            Pearson gate runs over the (train, holdout) pairs. When
            omitted, the gate reports SKIP — the audit then documents the
            candidate split only and carries no transfer evidence.
        thresholds: KC thresholds; defaults to
            ``KCThresholds.pure_objective()`` because bridged trials carry
            no action counts (each re-evaluation reports ``sample_count=1``,
            which makes the KC-4b action-ratio sub-gate vacuous here).
        top_n: how many train-best trials the gate re-evaluates. Pearson
            over very few points is high-variance; prefer 10-20.

    Returns:
        StudyAuditReport — see the dataclass docstring for field semantics.

    Raises:
        ImportError: optuna is not installed (optional ``[p2]`` extra).
        ValueError: the study is multi-objective, has no completed trials
            with values, or ``top_n`` < 2.
    """
    try:
        # importlib indirection mirrors p2_tpe.py: optuna is an optional
        # extra, and a static ``import optuna`` fails pyright in environments
        # without it (the publish-readiness gate runs without extras).
        optuna = cast(Any, import_module("optuna"))
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules stub
        raise ImportError(_OPTUNA_BRIDGE_INSTALL_HINT) from exc

    if top_n < 2:
        raise ValueError(
            f"top_n must be >= 2 — Pearson needs at least 2 points, got {top_n}"
        )

    directions = list(getattr(study, "directions", []) or [])
    if len(directions) > 1:
        raise ValueError(
            "audit_optuna_study supports single-objective studies only; "
            f"this study has {len(directions)} objectives"
        )
    sign = (
        -1.0
        if study.direction == optuna.study.StudyDirection.MINIMIZE
        else 1.0
    )

    all_trials = list(study.trials)
    completed = [
        t
        for t in all_trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        raise ValueError(
            "study has no completed trials with values — nothing to audit"
        )

    thr = thresholds if thresholds is not None else KCThresholds.pure_objective()

    # Bridge trials into the same GridPoint shape run_p1's gate consumes.
    # Internal fitness is maximize-oriented (sign-folded); reported values
    # keep the study's own orientation.
    points = [
        GridPoint(
            idx=t.number,
            unlocked=dict(t.params),
            params=dict(t.params),
            result=EvalResult(fitness=sign * float(t.value), sample_count=1),
        )
        for t in completed
    ]
    feas_by_number = {t.number: _trial_feasible_flag(t) for t in completed}
    feasibility_source = (
        "user_attrs"
        if any(flag is not None for flag in feas_by_number.values())
        else "absent"
    )

    ranked = sorted(points, key=lambda p: p.result.fitness, reverse=True)
    top = ranked[: min(top_n, len(ranked))]

    holdout_by_number: dict[int, float] = {}
    if holdout_evaluate is not None:
        # Reuse the exact walk-forward + KC-4 machinery (no duplicated math).
        wf = WalkForward(test_target=_HoldoutTarget(holdout_evaluate, sign))
        wf_result = wf.run(train_grid=points, top_n=top_n)
        kc_report = check_kc4(
            train_fitnesses=wf_result.train_fitnesses,
            test_fitnesses=wf_result.test_fitnesses,
            trade_ratio=wf_result.trade_ratio_scaled,
            thresholds=thr,
        )
        pearson: float | None = (
            wf_result.pearson if wf_result.pearson_computable else None
        )
        pearson_status = wf_result.pearson_status
        for gp, test_fitness in zip(top, wf_result.test_fitnesses):
            holdout_by_number[gp.idx] = sign * test_fitness
    else:
        kc_report = KCReport(
            name="KC-4",
            status="SKIP",
            message=(
                "SKIP: no holdout_evaluate provided — walk-forward transfer "
                "was not measured (train-only audit, no out-of-sample evidence)"
            ),
            detail={
                "reason": "holdout_evaluate_absent",
                "n_completed": len(completed),
                "top_n": top_n,
            },
        )
        pearson = None
        pearson_status = "NOT_RUN"

    def _candidate(gp: GridPoint) -> TrialCandidate:
        return TrialCandidate(
            number=gp.idx,
            params=dict(gp.params),
            train_value=sign * gp.result.fitness,
            holdout_value=holdout_by_number.get(gp.idx),
            feasible=feas_by_number.get(gp.idx),
        )

    candidates = [_candidate(gp) for gp in top]
    best_any = _candidate(ranked[0])

    best_feasible: TrialCandidate | None = None
    if feasibility_source == "user_attrs":
        feasible_points = [gp for gp in ranked if feas_by_number.get(gp.idx) is True]
        if feasible_points:
            best_feasible = _candidate(feasible_points[0])
            if best_feasible.holdout_value is None and holdout_evaluate is not None:
                # The feasible best can sit outside the gated top-N; score it
                # once so the best_any/best_feasible table compares like with
                # like. Reported in the study's own orientation.
                best_feasible = TrialCandidate(
                    number=best_feasible.number,
                    params=best_feasible.params,
                    train_value=best_feasible.train_value,
                    holdout_value=float(
                        holdout_evaluate(dict(best_feasible.params))
                    ),
                    feasible=best_feasible.feasible,
                )

    if kc_report.status == "FAIL":
        gated_best: TrialCandidate | None = None
    elif feasibility_source == "user_attrs":
        gated_best = best_feasible  # None when nothing is feasible: refuse.
    else:
        gated_best = best_any

    return StudyAuditReport(
        passed=kc_report.status != "FAIL",
        kc_report=kc_report,
        best_any=best_any,
        best_feasible=best_feasible,
        gated_best=gated_best,
        candidates=candidates,
        pearson=pearson,
        pearson_status=pearson_status,
        feasibility_source=feasibility_source,
        holdout_evaluated=holdout_evaluate is not None,
        n_trials_total=len(all_trials),
        n_trials_completed=len(completed),
        top_n=top_n,
        thresholds=thr,
    )
