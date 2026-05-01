# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P1: KC-4's Pearson result distinguishes degenerate cases.

Pre-fix: ``pearson()`` collapsed every failure mode (empty input,
length mismatch, zero variance) to ``0.0``. The KC-4 report then
showed ``pearson=0.0, pearson_ok=False`` — indistinguishable from a
genuine zero correlation. An artifact reader had no way to tell:

  - "we measured no correlation"               → real signal
  - "we couldn't measure the correlation"      → degenerate input

Both situations failed KC-4 by missing the threshold, but the reader
needed different responses for each.

Post-fix: ``pearson_result`` returns a ``PearsonResult`` carrying
``status`` ∈ {OK, EMPTY, LENGTH_MISMATCH, ZERO_VARIANCE_X,
ZERO_VARIANCE_Y}. KC-4's detail dict now exposes ``pearson_status``
and ``pearson_computable`` so consumers can branch on it.
"""
from __future__ import annotations

from omega_lock.kill_criteria import KCThresholds, check_kc4
from omega_lock.walk_forward import PearsonResult, pearson, pearson_result


# ---------------------------------------------------------------------------
# pearson_result direct.
# ---------------------------------------------------------------------------


def test_pearson_result_ok_for_correlated_inputs():
    pr = pearson_result([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.0, 4.2])
    assert pr.status == "OK"
    assert pr.computable
    assert pr.value is not None
    assert 0.95 < pr.value <= 1.0


def test_pearson_result_empty():
    pr = pearson_result([], [])
    assert pr.status == "EMPTY"
    assert not pr.computable
    assert pr.value is None


def test_pearson_result_length_mismatch():
    pr = pearson_result([1.0, 2.0], [1.0, 2.0, 3.0])
    assert pr.status == "LENGTH_MISMATCH"
    assert pr.value is None


def test_pearson_result_zero_variance_x():
    pr = pearson_result([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert pr.status == "ZERO_VARIANCE_X"
    assert pr.value is None


def test_pearson_result_zero_variance_y():
    pr = pearson_result([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert pr.status == "ZERO_VARIANCE_Y"
    assert pr.value is None


# ---------------------------------------------------------------------------
# Backward-compat wrapper.
# ---------------------------------------------------------------------------


def test_pearson_legacy_wrapper_returns_zero_on_degenerate():
    """The thin float wrapper is preserved so existing artifacts /
    callers still see a numeric 0.0 in degenerate cases. Only when
    the caller upgrades to pearson_result do they get the status."""
    assert pearson([], []) == 0.0
    assert pearson([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_pearson_legacy_wrapper_returns_correlation_for_normal_input():
    """Normal correlation passes through unchanged."""
    val = pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # Identity correlation; floating-point reduction may land at
    # 0.999999... so allow a tiny tolerance instead of exact 1.0.
    assert abs(val - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# check_kc4 — degenerate inputs report status, not just zero.
# ---------------------------------------------------------------------------


def test_kc4_marks_pearson_status_ok_when_correlated():
    report = check_kc4(
        train_fitnesses=[1.0, 2.0, 3.0],
        test_fitnesses=[1.0, 2.0, 3.0],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    assert report.detail["pearson_status"] == "OK"
    assert report.detail["pearson_computable"] is True
    assert report.status == "PASS"


def test_kc4_zero_variance_train_surfaces_status_in_detail():
    """Pre-fix: pearson=0.0, pearson_ok=False, KC-4 fails — but the
    detail dict gave no hint why. Post-fix: pearson_status='ZERO_VARIANCE_X'."""
    report = check_kc4(
        train_fitnesses=[1.0, 1.0, 1.0],
        test_fitnesses=[0.5, 0.7, 0.3],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    assert report.status == "FAIL"
    assert report.detail["pearson_status"] == "ZERO_VARIANCE_X"
    assert report.detail["pearson_computable"] is False
    assert "uncomputable" in report.message


def test_kc4_zero_variance_test_surfaces_status_in_detail():
    report = check_kc4(
        train_fitnesses=[0.5, 0.7, 0.3],
        test_fitnesses=[1.0, 1.0, 1.0],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    assert report.status == "FAIL"
    assert report.detail["pearson_status"] == "ZERO_VARIANCE_Y"


def test_kc4_empty_inputs_distinguishable_from_zero_correlation():
    empty_report = check_kc4(
        train_fitnesses=[],
        test_fitnesses=[],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    zero_corr_report = check_kc4(
        train_fitnesses=[1.0, 2.0, 3.0],
        # Anti-correlation: pearson is exactly -1.0
        test_fitnesses=[3.0, 2.0, 1.0],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    # Both fail KC-4, but the reason is now distinguishable in detail.
    assert empty_report.status == "FAIL"
    assert empty_report.detail["pearson_status"] == "EMPTY"
    assert zero_corr_report.status == "FAIL"
    assert zero_corr_report.detail["pearson_status"] == "OK"
    # The "anti-correlated" branch DID measure something:
    assert zero_corr_report.detail["pearson_computable"] is True
    # The "empty" branch did not:
    assert empty_report.detail["pearson_computable"] is False


def test_kc4_legacy_pearson_field_still_numeric():
    """detail['pearson'] is still a float for backward compat — None
    isn't injected even on uncomputable input. Existing dashboards
    that key off the float keep working; new ones look at pearson_status."""
    report = check_kc4(
        train_fitnesses=[],
        test_fitnesses=[],
        trade_ratio=1.0,
        thresholds=KCThresholds(),
    )
    assert isinstance(report.detail["pearson"], float)
    assert report.detail["pearson"] == 0.0
