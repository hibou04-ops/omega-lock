"""Tests for KC-1..4 boundary behavior."""
from __future__ import annotations


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


# KC-2 advisory boundary

def test_kc2_single_nonzero_stress_visible_but_advisory_by_default():
    r = check_kc2(
        [10.0, 0.0, 0.0, 0.0],
        KCThresholds(gini_min=0.0, top_bot_ratio_min=1.0),
    )

    assert r.status == "PASS"
    assert r.detail["nonzero_stress_count"] == 1
    assert r.detail["min_nonzero_stress_count"] is None
    assert r.detail["nonzero_ok"] is True


def test_kc2_nonzero_stress_floor_is_blocking_when_configured():
    r = check_kc2(
        [10.0, 0.0, 0.0, 0.0],
        KCThresholds(
            gini_min=0.0,
            top_bot_ratio_min=1.0,
            min_nonzero_stress_count=2,
        ),
    )

    assert r.status == "FAIL"
    assert r.detail["nonzero_stress_count"] == 1
    assert r.detail["nonzero_ok"] is False
    assert "nonzero_stress_count" in r.message


# KC-3

def test_kc3_pass_all_counts_above_floor():
    r = check_kc3({"baseline": 200, "best": 150}, KCThresholds(trade_count_min=50))
    assert r.status == "PASS"


def test_kc3_pass_at_exact_trade_floor():
    r = check_kc3({"baseline": 50, "best": 50}, KCThresholds(trade_count_min=50))

    assert r.status == "PASS"
    assert r.detail["failures"] == {}


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
    r = check_kc4(
        train,
        test,
        trade_ratio=0.8,
        thresholds=KCThresholds(pearson_min=0.3, trade_ratio_min=0.5),
    )
    assert r.status == "PASS"


def test_kc4_pass_at_exact_trade_ratio_boundary():
    train = [1.0, 2.0, 3.0, 4.0]
    test = [1.0, 2.0, 3.0, 4.0]
    r = check_kc4(
        train,
        test,
        trade_ratio=0.5,
        thresholds=KCThresholds(pearson_min=0.99, trade_ratio_min=0.5),
    )

    assert r.status == "PASS"
    assert r.detail["pearson_ok"] is True
    assert r.detail["trade_ratio_ok"] is True


def test_kc4_fail_on_uncorrelated():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    test = [5.0, 1.0, 4.0, 2.0, 3.0]  # shuffled — low corr
    r = check_kc4(train, test, trade_ratio=0.8, thresholds=KCThresholds(pearson_min=0.5))
    assert r.status == "FAIL"


def test_kc4_fail_on_low_trade_ratio():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    test = [1.1, 2.1, 3.1, 4.1, 5.1]  # perfect corr
    r = check_kc4(
        train,
        test,
        trade_ratio=0.3,
        thresholds=KCThresholds(pearson_min=0.3, trade_ratio_min=0.5),
    )
    assert r.status == "FAIL"
    assert not r.detail["trade_ratio_ok"]


# ── pure_objective preset (B2: non-action objectives) ──

def test_pure_objective_disables_action_gates():
    t = KCThresholds.pure_objective()
    assert t.trade_count_min is None
    assert t.trade_ratio_min is None
    # domain-neutral gates keep their defaults
    assert t.gini_min == KCThresholds().gini_min
    assert t.pearson_min == KCThresholds().pearson_min
    assert t.time_box_seconds == KCThresholds().time_box_seconds


def test_pure_objective_accepts_overrides():
    t = KCThresholds.pure_objective(pearson_min=0.5, gini_min=0.0)
    assert t.pearson_min == 0.5
    assert t.gini_min == 0.0
    assert t.trade_count_min is None  # preset default preserved


def test_kc3_skips_when_floor_disabled():
    # even a tiny / empty action count must SKIP, not FAIL, in pure mode
    t = KCThresholds.pure_objective()
    r = check_kc3({"train_best": 1}, t)
    assert r.status == "SKIP"
    assert r.detail["floor"] is None
    r_empty = check_kc3({}, t)
    assert r_empty.status == "SKIP"


def test_kc3_still_gates_with_explicit_floor():
    t = KCThresholds(trade_count_min=50)
    assert check_kc3({"train_best": 49}, t).status == "FAIL"
    assert check_kc3({"train_best": 50}, t).status == "PASS"


def test_kc4_skips_ratio_but_keeps_correlation():
    train = [1.0, 2.0, 3.0, 4.0]
    test = [1.1, 2.1, 2.9, 4.2]  # strong positive correlation
    t = KCThresholds.pure_objective(pearson_min=0.3)
    # trade_ratio well below the normal 0.5 floor, but it is skipped now
    r = check_kc4(train, test, trade_ratio=0.0, thresholds=t)
    assert r.status == "PASS"
    assert "skipped" in r.message


def test_kc4_correlation_still_gates_in_pure_mode():
    train = [1.0, 2.0, 3.0, 4.0]
    test = [4.0, 3.0, 2.0, 1.0]  # strong negative correlation
    t = KCThresholds.pure_objective(pearson_min=0.3)
    r = check_kc4(train, test, trade_ratio=0.0, thresholds=t)
    assert r.status == "FAIL"  # KC-4a (correlation) is retained
