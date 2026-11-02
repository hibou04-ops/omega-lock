# SPDX-License-Identifier: Apache-2.0
"""Regression contracts for deployment authorization and score-level degradation."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import pytest

from omega_lock import gate_scores
from omega_lock.cli import main
from omega_lock.kill_criteria import KCThresholds


def test_exact_location_shift_is_blocked_despite_perfect_correlation():
    result = gate_scores([1, 2, 3], [-100, -99, -98])
    assert result.pearson == pytest.approx(1.0)
    assert result.degradation_gap == 101.0
    assert result.passed is False and result.deployment_authorized is False
    assert result.to_dict()["deployment_authorized"] is False
    assert result.kc_report.detail["degradation_ok"] is False
    assert "degradation_gap" in result.reasons[0]


@pytest.mark.parametrize("holdout,limit,authorized", [
    ([0.0, 1.0, 2.0], 1.0, True),
    ([-0.25, 0.75, 1.75], 1.0, False),
    ([-0.25, 0.75, 1.75], 1.25, True),
    ([1.0, 2.0, 3.0], 0.0, True),
])
def test_gap_limit_is_inclusive_and_uses_score_units(holdout, limit, authorized):
    verdict = gate_scores([1, 2, 3], holdout, max_degradation_gap=limit)
    assert verdict.passed is authorized
    assert verdict.deployment_authorized is authorized


def test_minimizing_orientation_maps_score_increases_to_degradation():
    failed = gate_scores([1, 2, 3], [102, 103, 104], direction="minimize")
    passed = gate_scores([1, 2, 3], [0, 1, 2], direction="minimize")
    assert failed.degradation_gap == 101.0 and not failed.deployment_authorized
    assert passed.degradation_gap == -1.0 and passed.deployment_authorized


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0, None, True])
def test_invalid_or_disabled_gap_threshold_rejected(bad: Any):
    with pytest.raises(ValueError, match="max_degradation_gap"):
        gate_scores([1, 2, 3], [1, 2, 3], max_degradation_gap=bad)


@pytest.mark.parametrize("train,holdout", [
    ([], []), ([1, 2, 3], [1, 2]), ([1, 1, 1], [1, 2, 3]),
    ([1, math.nan, 3], [1, 2, 3]), ([1, 2, 3], [1, math.inf, 3]),
])
def test_missing_nonfinite_or_degenerate_evidence_never_authorizes(train, holdout):
    verdict = gate_scores(train, holdout)
    assert verdict.passed is False
    assert verdict.deployment_authorized is False


def test_gap_overflow_is_an_explicit_unmeasured_failure():
    verdict = gate_scores([1e308, 1.1e308, 1.2e308], [-1.2e308, -1.1e308, -1e308])
    assert verdict.pearson == pytest.approx(1.0)
    assert verdict.degradation_gap is None
    assert verdict.kc_report.detail["degradation_status"] == "NON_FINITE_GAP"
    assert not verdict.deployment_authorized


def test_large_finite_means_do_not_overflow_and_small_represented_drop_survives():
    unchanged = gate_scores([1e308, 1.1e308, 1.2e308], [1e308, 1.1e308, 1.2e308])
    assert unchanged.degradation_gap == 0 and unchanged.deployment_authorized
    dropped = gate_scores(
        [1e16, 1e16 + 2, 1e16 + 4], [1e16, 1e16 + 2, 1e16 + 2],
        max_degradation_gap=0.0,
    )
    assert dropped.degradation_gap == pytest.approx(2 / 3)
    assert not dropped.deployment_authorized


def test_underflow_is_not_silently_rounded_into_zero_degradation():
    verdict = gate_scores([0.0, 1.0, 2.0], [-math.ulp(0.0), 1.0, 2.0], max_degradation_gap=0.0)
    assert verdict.pearson == pytest.approx(1.0)
    assert verdict.degradation_gap is None
    assert verdict.kc_report.detail["degradation_status"] == "UNDERFLOW_GAP"
    assert verdict.kc_report.detail["degradation_ok"] is False
    assert not verdict.deployment_authorized


def test_exact_bound_comparison_does_not_approve_a_gap_rounded_down_to_limit():
    verdict = gate_scores(
        [1.0, 2.0**-52], [-1.0, 0.0], max_degradation_gap=1.0,
        thresholds=KCThresholds.pure_objective(pearson_min=-1.0),
    )
    assert verdict.degradation_gap == 1.0  # rounded display of 1 + 2**-53
    assert verdict.kc_report.detail["degradation_ok"] is False
    assert not verdict.passed and not verdict.deployment_authorized


def test_manually_forced_approval_cannot_override_a_failed_verdict():
    failed = gate_scores([1, 2, 3], [-100, -99, -98])
    assert replace(failed, deployment_authorized=True).deployment_authorized is False


def test_cli_gap_threshold_controls_exit_and_exposes_authorization(tmp_path: Path, capsys):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    train.write_text(json.dumps([1, 2, 3]))
    holdout.write_text(json.dumps([-100, -99, -98]))
    args = ["gate", "--train", str(train), "--holdout", str(holdout)]
    assert main(args) == 1
    assert "deployment_authorized: False" in capsys.readouterr().out
    assert main([*args, "--max-degradation-gap", "101"]) == 0
    assert "deployment_authorized: True" in capsys.readouterr().out
    assert main([*args, "--max-degradation-gap", "nan"]) == 2
    assert "max_degradation_gap" in capsys.readouterr().err
