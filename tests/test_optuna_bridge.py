# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for audit_optuna_study — gating an existing Optuna study.

The walk-forward + KC-4 math is reused from the core modules, so these
tests focus on the bridge semantics: trial extraction, direction
handling, feasibility inference, gate wiring, and the report shape.
All optuna-dependent tests skip gracefully when optuna is not installed;
the ImportError-path test runs regardless.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from omega_lock.integrations.optuna_bridge import (
    StudyAuditReport,
    TrialCandidate,
    audit_optuna_study,
)
from omega_lock.kill_criteria import KCThresholds
from omega_lock.walk_forward import pearson

try:
    import optuna as _optuna_module

    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    _optuna_module = None
    _HAS_OPTUNA = False

# Typed Any on purpose: every use below is behind the requires_optuna marker,
# and pyright cannot narrow a module-or-None through pytest skip markers.
optuna: Any = _optuna_module

requires_optuna = pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")


def test_import_error_is_clean_when_optuna_missing(monkeypatch: pytest.MonkeyPatch):
    """Simulate a missing optuna even on machines that have it installed."""
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(ImportError, match=r"omega-lock\[p2\]"):
        audit_optuna_study(object())


# ── Study fixtures ──────────────────────────────────────────────────────────


def _add_complete_trial(
    study: Any,
    x: float,
    value: float,
    *,
    feasible: bool | None = None,
) -> None:
    user_attrs: dict[str, Any] = {}
    if feasible is not None:
        user_attrs["feasible"] = feasible
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": x},
            distributions={"x": optuna.distributions.FloatDistribution(-100.0, 100.0)},
            value=value,
            user_attrs=user_attrs or None,
        )
    )


def _study_with_values(
    values: list[float],
    *,
    direction: str = "maximize",
    feasible_flags: list[bool | None] | None = None,
) -> Any:
    study = optuna.create_study(direction=direction)
    for i, v in enumerate(values):
        flag = feasible_flags[i] if feasible_flags is not None else None
        _add_complete_trial(study, x=float(i), value=v, feasible=flag)
    return study


# ── Extraction ──────────────────────────────────────────────────────────────


@requires_optuna
def test_only_completed_trials_with_values_are_bridged():
    study = _study_with_values([1.0, 2.0, 3.0])
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))

    report = audit_optuna_study(study, top_n=2)

    assert report.n_trials_total == 5
    assert report.n_trials_completed == 3
    assert report.best_any.train_value == 3.0


@requires_optuna
def test_no_completed_trials_raises():
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))

    with pytest.raises(ValueError, match="no completed trials"):
        audit_optuna_study(study)


@requires_optuna
def test_top_n_below_two_raises():
    study = _study_with_values([1.0, 2.0])

    with pytest.raises(ValueError, match="top_n must be >= 2"):
        audit_optuna_study(study, top_n=1)


@requires_optuna
def test_multi_objective_study_is_rejected():
    study = optuna.create_study(directions=["maximize", "maximize"])

    with pytest.raises(ValueError, match="single-objective"):
        audit_optuna_study(study)


# ── Gate wiring (reused KC-4 math) ─────────────────────────────────────────


@requires_optuna
def test_gate_passes_on_correlated_holdout():
    study = _study_with_values([1.0, 2.0, 3.0, 4.0, 5.0])

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: float(p["x"]), top_n=5
    )

    assert report.kc_report.status == "PASS"
    assert report.passed is True
    assert report.pearson == pytest.approx(1.0)
    assert report.pearson_status == "OK"
    assert report.holdout_evaluated is True
    # Feasibility is absent, so the certified candidate is the study winner.
    assert report.feasibility_source == "absent"
    assert report.best_feasible is None
    assert report.gated_best is not None
    assert report.gated_best.number == report.best_any.number


@requires_optuna
def test_gate_fails_on_anticorrelated_holdout_and_certifies_nothing():
    study = _study_with_values([1.0, 2.0, 3.0, 4.0, 5.0])

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: -float(p["x"]), top_n=5
    )

    assert report.kc_report.status == "FAIL"
    assert report.passed is False
    assert report.pearson == pytest.approx(-1.0)
    assert report.gated_best is None


@requires_optuna
def test_pearson_matches_reused_walk_forward_math_exactly():
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3]
    study = _study_with_values(values)
    holdout = {float(i): (v * 0.5 + ((-1) ** i) * 0.7) for i, v in enumerate(values)}

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: holdout[p["x"]], top_n=5
    )

    ranked = sorted(enumerate(values), key=lambda iv: iv[1], reverse=True)[:5]
    train_fs = [v for _, v in ranked]
    test_fs = [holdout[float(i)] for i, _ in ranked]
    assert report.pearson == pearson(train_fs, test_fs)


@requires_optuna
def test_no_holdout_reports_skip_not_pass():
    study = _study_with_values([1.0, 2.0, 3.0])

    report = audit_optuna_study(study, top_n=3)

    assert report.kc_report.status == "SKIP"
    assert "no holdout_evaluate" in report.kc_report.message
    assert report.passed is True  # not failed — but carries no transfer evidence
    assert report.pearson is None
    assert report.pearson_status == "NOT_RUN"
    assert report.holdout_evaluated is False
    assert all(c.holdout_value is None for c in report.candidates)


@requires_optuna
def test_custom_thresholds_are_applied_and_recorded():
    study = _study_with_values([1.0, 2.0, 3.0, 4.0, 5.0])
    strict = KCThresholds.pure_objective(pearson_min=0.999)

    report = audit_optuna_study(
        study,
        holdout_evaluate=lambda p: float(p["x"]) + (0.8 if p["x"] == 2.0 else 0.0),
        thresholds=strict,
        top_n=5,
    )

    assert report.thresholds.pearson_min == 0.999
    assert report.pearson is not None and report.pearson < 0.999
    assert report.kc_report.status == "FAIL"


# ── Feasibility inference ──────────────────────────────────────────────────


@requires_optuna
def test_user_attrs_feasibility_splits_best_any_from_best_feasible():
    # Highest value (10.0) is infeasible; best feasible is 4.0.
    study = _study_with_values(
        [1.0, 10.0, 4.0, 3.0],
        feasible_flags=[True, False, True, True],
    )

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: float(p["x"]), top_n=4
    )

    assert report.feasibility_source == "user_attrs"
    assert report.best_any.train_value == 10.0
    assert report.best_any.feasible is False
    assert report.best_feasible is not None
    assert report.best_feasible.train_value == 4.0
    assert report.best_feasible.feasible is True


@requires_optuna
def test_best_feasible_outside_top_n_still_gets_holdout_value():
    # top_n=2 covers the two highest values; the only feasible trial ranks 3rd.
    study = _study_with_values(
        [9.0, 8.0, 2.0],
        feasible_flags=[False, False, True],
    )
    calls: list[float] = []

    def holdout(params: dict[str, Any]) -> float:
        calls.append(params["x"])
        return float(params["x"])

    report = audit_optuna_study(study, holdout_evaluate=holdout, top_n=2)

    assert report.best_feasible is not None
    assert report.best_feasible.train_value == 2.0
    assert report.best_feasible.holdout_value == 2.0  # scored on demand
    assert 2.0 in calls


@requires_optuna
def test_all_infeasible_means_no_certified_candidate():
    study = _study_with_values(
        [1.0, 2.0, 3.0],
        feasible_flags=[False, False, False],
    )

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: float(p["x"]), top_n=3
    )

    assert report.kc_report.status == "PASS"  # ranking transfers fine
    assert report.feasibility_source == "user_attrs"
    assert report.best_feasible is None
    assert report.gated_best is None  # but nothing is feasible to certify


@requires_optuna
def test_non_boolean_feasible_attr_is_ignored():
    study = optuna.create_study()
    _add_complete_trial(study, x=0.0, value=1.0)
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(-100.0, 100.0)},
            value=2.0,
            user_attrs={"feasible": "yes"},  # not a bool — not a flag
        )
    )

    report = audit_optuna_study(study, top_n=2)

    assert report.feasibility_source == "absent"
    assert report.best_feasible is None


# ── Direction handling ─────────────────────────────────────────────────────


@requires_optuna
def test_minimize_direction_ranks_and_reports_in_study_orientation():
    # Lower is better; the best trial is value 1.0 at x=4.
    study = _study_with_values([5.0, 4.0, 3.0, 2.0, 1.0], direction="minimize")

    report = audit_optuna_study(
        study,
        # Holdout in the SAME (minimize) orientation, rank-consistent.
        holdout_evaluate=lambda p: 10.0 - float(p["x"]),
        top_n=5,
    )

    assert report.best_any.train_value == 1.0
    assert report.best_any.number == 4
    assert report.kc_report.status == "PASS"
    assert report.pearson == pytest.approx(1.0)
    # Reported holdout values keep the study's own orientation.
    assert report.best_any.holdout_value == pytest.approx(6.0)


# ── Report shape ───────────────────────────────────────────────────────────


@requires_optuna
def test_report_to_dict_is_json_serializable_and_complete():
    study = _study_with_values(
        [1.0, 2.0, 3.0], feasible_flags=[True, True, False]
    )

    report = audit_optuna_study(
        study, holdout_evaluate=lambda p: float(p["x"]), top_n=3
    )
    payload = report.to_dict()

    serialized = json.dumps(payload)  # must not raise
    assert isinstance(serialized, str)
    assert payload["schema_version"] == "omega-lock.study-audit.v1"
    assert payload["kc_report"]["name"] == "KC-4"
    assert payload["kc_thresholds"]["pearson_min"] == 0.3
    assert len(payload["candidates"]) == 3
    assert isinstance(report, StudyAuditReport)
    assert all(isinstance(c, TrialCandidate) for c in report.candidates)


@requires_optuna
def test_candidates_are_in_train_rank_order_and_capped_at_top_n():
    study = _study_with_values([3.0, 9.0, 1.0, 7.0, 5.0])

    report = audit_optuna_study(study, top_n=3)

    assert [c.train_value for c in report.candidates] == [9.0, 7.0, 5.0]
    assert report.top_n == 3
