# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for render_html — the stdlib-only single-file HTML scorecard.

Golden-ish: assert load-bearing strings (verdict banner, gate rows,
tables, inline SVG with identity line) rather than full byte snapshots,
plus byte-level determinism of repeated renders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omega_lock import (
    EvalResult,
    KCThresholds,
    P1Config,
    ParamSpec,
    gate_scores,
    render_html,
    run_p1,
)
from omega_lock.audit._types import AuditedRun, AuditReport, Constraint
from omega_lock.integrations.optuna_bridge import StudyAuditReport, TrialCandidate
from omega_lock.kill_criteria import KCReport


class _LinearTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", neutral=0.0, low=0.0, high=1.0),
            ParamSpec(name="y", dtype="float", neutral=0.0, low=0.0, high=1.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(
            fitness=10.0 * float(params["x"]) + float(params["y"]),
            sample_count=100,
        )


class _AntiTarget(_LinearTarget):
    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        base = super().evaluate(params)
        return EvalResult(fitness=-base.fitness, sample_count=100)


def _p1_config() -> P1Config:
    return P1Config(
        unlock_k=2,
        grid_points_per_axis=3,
        walk_forward_top_n=4,
        kc_thresholds=KCThresholds(
            gini_min=0.0,
            top_bot_ratio_min=1.0,
            trade_count_min=1,
            pearson_min=0.9,
            trade_ratio_min=0.0,
        ),
        stress_verbose=False,
        grid_verbose=False,
    )


@pytest.fixture(scope="module")
def p1_pass_result():
    return run_p1(
        train_target=_LinearTarget(),
        test_target=_LinearTarget(),
        config=_p1_config(),
    )


@pytest.fixture(scope="module")
def p1_fail_result():
    return run_p1(
        train_target=_LinearTarget(),
        test_target=_AntiTarget(),
        config=_p1_config(),
    )


# ── P1Result rendering ─────────────────────────────────────────────────────


def test_p1_pass_render_contains_all_sections(p1_pass_result):
    html = render_html(p1_pass_result)

    assert html.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in html
    assert '<div class="banner pass">PASS</div>' in html
    assert "KC-4" in html and "KC-2" in html
    assert "best_any" in html
    assert "grid_best (selected)" in html
    assert "Stress ranking" in html
    assert "<svg" in html and 'class="identity"' in html
    assert "test fitness (walk-forward slice)" in html


def test_p1_fail_render_shows_fail_banner(p1_fail_result):
    html = render_html(p1_fail_result)

    assert "FAIL:KC-4" in html
    assert 'class="banner fail"' in html
    assert 'class="status-FAIL"' in html


def test_render_is_deterministic_and_writes_identical_file(
    p1_pass_result, tmp_path: Path
):
    out = tmp_path / "scorecard.html"

    first = render_html(p1_pass_result, out)
    second = render_html(p1_pass_result)

    assert first == second
    assert out.read_text(encoding="utf-8") == first


def test_no_timestamp_unless_generated_at_passed(p1_pass_result):
    plain = render_html(p1_pass_result)
    stamped = render_html(p1_pass_result, generated_at="2026-01-01T00:00:00Z")

    assert "generated at" not in plain
    assert "generated at 2026-01-01T00:00:00Z" in stamped


def test_p1_result_json_artifact_dict_renders_like_the_object(
    p1_pass_result, tmp_path: Path
):
    artifact = tmp_path / "p1_result.json"
    p1_pass_result.save(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert render_html(payload) == render_html(p1_pass_result)


# ── AuditReport rendering ──────────────────────────────────────────────────


def _audit_report() -> AuditReport:
    constraint = Constraint("x_lte_one", lambda p, r: True, "x must be <= 1")

    def _run(idx: int, x: float, fitness: float, failed: tuple[str, ...]) -> AuditedRun:
        return AuditedRun(
            params={"x": x},
            fitness=fitness,
            n_trials=10,
            metadata={},
            timestamp_iso=f"2026-01-01T00:00:{idx:02d}+00:00",
            constraints_passed=() if failed else ("x_lte_one",),
            constraints_failed=failed,
            phase="grid",
            call_index=idx,
            target_role="train",
        )

    return AuditReport(
        method="test",
        omega_lock_version="0.0.0-test",
        seed=7,
        started_iso="2026-01-01T00:00:00+00:00",
        ended_iso="2026-01-01T00:00:02+00:00",
        constraints=(constraint,),
        runs=(
            _run(0, 0.5, 1.0, ()),
            _run(1, 2.0, 9.0, ("x_lte_one",)),
            _run(2, 1.0, 4.0, ()),
        ),
        stress_ranking=(("x", 3.5),),
    )


def test_audit_report_renders_neutral_banner_and_feasible_split():
    report = _audit_report()

    html = render_html(report)

    assert 'class="banner neutral">TRAIL' in html
    assert "best_any" in html and "best_feasible" in html
    assert ">9<" in html  # infeasible top fitness
    assert ">4<" in html  # feasible best fitness
    assert "Stress ranking" in html
    assert "<svg" not in html  # no per-candidate train/holdout pairs


def test_audit_report_dict_payload_renders_identically():
    report = _audit_report()

    assert render_html(report.to_dict()) == render_html(report)


# ── StudyAuditReport rendering (constructed directly; no optuna needed) ───


def _study_report() -> StudyAuditReport:
    cands = [
        TrialCandidate(number=3, params={"x": 3.0}, train_value=9.0, holdout_value=2.0),
        TrialCandidate(number=1, params={"x": 1.0}, train_value=5.0, holdout_value=5.1),
        TrialCandidate(number=2, params={"x": 2.0}, train_value=4.0, holdout_value=4.2),
    ]
    kc = KCReport(name="KC-4", status="FAIL", message="FAIL: pearson=-0.500<0.3", detail={})
    return StudyAuditReport(
        passed=False,
        kc_report=kc,
        best_any=cands[0],
        best_feasible=cands[1],
        gated_best=None,
        candidates=cands,
        pearson=-0.5,
        pearson_status="OK",
        feasibility_source="user_attrs",
        holdout_evaluated=True,
        n_trials_total=10,
        n_trials_completed=8,
        top_n=3,
    )


def test_study_audit_report_renders_gate_and_candidate_split():
    report = _study_report()

    html = render_html(report)

    assert "Optuna study audit" in html
    assert "KC-4 FAIL" in html
    assert "best_any" in html and "best_feasible" in html and "gated_best" in html
    assert "gate refused to certify" in html
    assert html.count('class="pt"') == 3  # one scatter point per candidate


def test_study_audit_report_dict_payload_renders_identically():
    report = _study_report()

    assert render_html(report.to_dict()) == render_html(report)


# ── GateVerdict rendering ──────────────────────────────────────────────────


def test_gate_verdict_render_lists_reasons_and_scatter():
    verdict = gate_scores([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])

    html = render_html(verdict)

    assert "omega-lock score gate" in html
    assert 'class="banner fail">FAIL' in html
    assert "Reasons" in html
    assert "did not survive out-of-sample" in html
    assert html.count('class="pt"') == 4


# ── Error handling ─────────────────────────────────────────────────────────


def test_unknown_mapping_schema_raises_value_error():
    with pytest.raises(ValueError, match="unrecognized mapping"):
        render_html({"schema_version": "something.else.v1"})


def test_unsupported_object_raises_type_error():
    with pytest.raises(TypeError, match="render_html accepts"):
        render_html(42)
