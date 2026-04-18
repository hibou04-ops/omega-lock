"""Tests for KC-1..4 boundary behavior."""
from __future__ import annotations

import pytest

from omega_lock.kill_criteria import (
    KCThresholds,
    check_kc1,
    check_kc2,
    check_kc3,
    check_kc4,
)


# ── KC-1 ──

def test_kc1_pass_when_under_budget():
    r = check_kc1(elapsed_seconds=100.0, thresholds=KCThresholds(time_box_seconds=200.0))
    assert r.status == "PASS"


def test_kc1_fail_when_over_budget():
    r = check_kc1(elapsed_seconds=300.0, thresholds=KCThresholds(time_box_seconds=200.0))
    assert r.status == "FAIL"


def test_kc1_pass_at_exact_boundary():
    r = check_kc1(elapsed_seconds=200.0, thresholds=KCThresholds(time_box_seconds=200.0))
    assert r.status == "PASS"


# ── KC-2 ──

def test_kc2_pass_on_high_differentiation():
    stresses = [100.0, 80.0, 60.0, 5.0, 3.0, 1.0]
    r = check_kc2(stresses, KCThresholds(gini_min=0.2, top_bot_ratio_min=2.0))
    assert r.status == "PASS"


def test_kc2_fail_on_flat_distribution():
    stresses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    r = check_kc2(stresses, KCThresholds())
    assert r.status == "FAIL"


def test_kc2_fail_on_empty():
    r = check_kc2([], KCThresholds())
    assert r.status == "FAIL"


def test_kc2_low_ratio_fails():
    # Gini might pass, but if top3 and bot3 are close, ratio can fail
    stresses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.5]
    r = check_kc2(stresses, KCThresholds(gini_min=0.0, top_bot_ratio_min=2.0))
    # top3 mean = 9, bot3 mean = 6.17, ratio = 1.46 < 2.0 → FAIL
    assert r.status == "FAIL"
    assert "ratio" in r.detail


# ── KC-3 ──

def test_kc3_pass_all_counts_above_floor():
    r = check_kc3({"baseline": 200, "best": 150}, KCThresholds(trade_count_min=50))
    assert r.status == "PASS"


def test_kc3_fail_any_below_floor():
    r = check_kc3({"baseline": 200, "best": 30}, KCThresholds(trade_count_min=50))
    assert r.status == "FAIL"
    assert "best" in r.detail["failures"]


def test_kc3_empty_fails():
    r = check_kc3({}, KCThresholds())
    assert r.status == "FAIL"


# ── KC-4 ──

def test_kc4_pass_on_good_correlation_and_trades():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    test = [1.1, 1.9, 3.0, 4.1, 5.2]
    r = check_kc4(train, test, trade_ratio=0.8, thresholds=KCThresholds(pearson_min=0.3, trade_ratio_min=0.5))
    assert r.status == "PASS"


def test_kc4_fail_on_uncorrelated():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    test = [5.0, 1.0, 4.0, 2.0, 3.0]  # shuffled — low corr
    r = check_kc4(train, test, trade_ratio=0.8, thresholds=KCThresholds(pearson_min=0.5))
    assert r.status == "FAIL"


def test_kc4_fail_on_low_trade_ratio():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    test = [1.1, 2.1, 3.1, 4.1, 5.1]  # perfect corr
    r = check_kc4(train, test, trade_ratio=0.3, thresholds=KCThresholds(pearson_min=0.3, trade_ratio_min=0.5))
    assert r.status == "FAIL"
    assert not r.detail["trade_ratio_ok"]
