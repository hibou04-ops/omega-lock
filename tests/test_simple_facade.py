# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for omega_lock.simple — gate_scores / audit / render_html facade."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omega_lock import GateVerdict, gate_scores
from omega_lock.kill_criteria import KCThresholds
from omega_lock.orchestrator import P1Result
from omega_lock.simple import audit
from omega_lock.target import ParamSpec


# ── gate_scores: pass / fail / degenerate ──────────────────────────────────


def test_gate_scores_passes_on_transferring_ranking():
    verdict = gate_scores([1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.2, 2.9, 4.2, 5.1])

    assert isinstance(verdict, GateVerdict)
    assert verdict.passed is True
    assert verdict.pearson is not None and verdict.pearson > 0.99
    assert verdict.reasons == ()
    assert verdict.kc_report.name == "KC-4"
    assert verdict.kc_report.status == "PASS"


def test_gate_scores_fails_on_anticorrelated_ranking():
    verdict = gate_scores([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0])

    assert verdict.passed is False
    assert verdict.pearson == pytest.approx(-1.0)
    assert len(verdict.reasons) == 1
    assert "below the gate threshold 0.3" in verdict.reasons[0]


def test_gate_scores_empty_input_fails_with_reason():
    verdict = gate_scores([], [])

    assert verdict.passed is False
    assert verdict.pearson is None
    assert any("EMPTY" in r for r in verdict.reasons)
    assert any("no scores were provided" in r for r in verdict.reasons)


def test_gate_scores_length_mismatch_fails_with_reason():
    verdict = gate_scores([1.0, 2.0, 3.0], [1.0, 2.0])

    assert verdict.passed is False
    assert verdict.pearson is None
    assert any("LENGTH_MISMATCH" in r for r in verdict.reasons)


def test_gate_scores_zero_variance_fails_with_reason():
    flat_train = gate_scores([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    flat_holdout = gate_scores([1.0, 2.0, 3.0], [7.0, 7.0, 7.0])

    assert flat_train.passed is False
    assert any("ZERO_VARIANCE_X" in r for r in flat_train.reasons)
    assert flat_holdout.passed is False
    assert any("ZERO_VARIANCE_Y" in r for r in flat_holdout.reasons)


def test_gate_scores_custom_thresholds_apply():
    scores = ([1.0, 2.0, 3.0, 4.0, 5.0], [1.3, 1.9, 3.4, 3.8, 5.2])

    default = gate_scores(*scores)
    strict = gate_scores(*scores, thresholds=KCThresholds.pure_objective(pearson_min=0.9999))

    assert default.passed is True
    assert strict.passed is False
    assert "0.9999" in strict.reasons[0]


def test_gate_scores_keeps_inputs_for_reporting_and_serializes():
    verdict = gate_scores([1, 2], [2, 1])

    assert verdict.train_scores == (1.0, 2.0)
    assert verdict.holdout_scores == (2.0, 1.0)
    payload = json.dumps(verdict.to_dict())
    assert "KC-4" in payload


# ── audit: thin CallableAdapter + run_p1 wrapper ───────────────────────────


def _score(params: dict[str, Any]) -> float:
    # Deliberately anisotropic so the default KC-2 differentiation gate
    # (Gini + top/bottom ratio) passes: b dominates the sensitivity.
    return -((params["a"] - 3.0) ** 2 + 100.0 * (params["b"] - 7.0) ** 2)


def test_audit_runs_end_to_end_with_dict_specs():
    result = audit(
        _score,
        {"a": (0.0, 10.0), "b": (0.0, 10.0)},
        holdout_fn=_score,
        unlock_k=2,
        grid_points_per_axis=5,
    )

    assert isinstance(result, P1Result)
    assert result.status == "PASS"
    assert result.grid_best is not None
    assert result.walk_forward is not None
    assert result.walk_forward["pearson"] == pytest.approx(1.0)
    # The facade defaults to pure_objective: action gates report SKIP.
    kc3 = next(r for r in result.kc_reports if r["name"] == "KC-3")
    assert kc3["status"] == "SKIP"


def test_audit_without_holdout_fn_skips_walk_forward():
    result = audit(_score, {"a": (0.0, 10.0), "b": (0.0, 10.0)}, unlock_k=2)

    assert result.walk_forward is None
    assert all(r["name"] != "KC-4" for r in result.kc_reports)


def test_audit_accepts_param_spec_list_and_three_tuples():
    specs = [
        ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
    ]
    from_specs = audit(_score, specs, unlock_k=2, grid_points_per_axis=3)
    from_triples = audit(
        _score,
        {"a": (0.0, 10.0, 5.0), "b": (0.0, 10.0, 5.0)},
        unlock_k=2,
        grid_points_per_axis=3,
    )

    assert from_specs.grid_best is not None
    assert from_triples.grid_best is not None
    assert from_specs.grid_best["unlocked"] == from_triples.grid_best["unlocked"]


def test_audit_dict_specs_default_neutral_is_midpoint():
    result = audit(_score, {"a": (2.0, 4.0), "b": (6.0, 8.0)}, unlock_k=2,
                   grid_points_per_axis=3)

    assert result.baseline_result["fitness"] == pytest.approx(_score({"a": 3.0, "b": 7.0}))


def test_audit_writes_output_artifact(tmp_path: Path):
    out = tmp_path / "artifacts" / "facade_result.json"

    audit(_score, {"a": (0.0, 10.0), "b": (0.0, 10.0)}, unlock_k=2,
          grid_points_per_axis=3, output_path=out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"].startswith("omega-lock.p1-result.")


def test_audit_respects_explicit_kc_thresholds():
    # An impossible action floor must fail KC-3 when explicitly requested —
    # proof the pure_objective default is only a default.
    result = audit(
        _score,
        {"a": (0.0, 10.0), "b": (0.0, 10.0)},
        unlock_k=2,
        grid_points_per_axis=3,
        kc_thresholds=KCThresholds(
            gini_min=0.0, top_bot_ratio_min=1.0, trade_count_min=10_000
        ),
    )

    assert "KC-3" in result.status


def test_audit_input_validation():
    with pytest.raises(ValueError, match="empty"):
        audit(_score, {})
    with pytest.raises(ValueError, match="empty"):
        audit(_score, [])
    with pytest.raises(ValueError, match=r"\(low, high\)"):
        audit(_score, {"a": (1.0,)})
    with pytest.raises(TypeError, match="ParamSpec"):
        audit(_score, ["not-a-spec"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        audit(_score, {"a": (0.0, 1.0)}, not_a_config_key=True)


# ── render_html re-export ──────────────────────────────────────────────────


def test_render_html_reexport_is_the_same_object():
    from omega_lock.report_html import render_html as canonical
    from omega_lock.simple import render_html as reexported

    assert reexported is canonical
