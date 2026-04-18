"""Tests for SC-2 advisory gate (run_p1 with run_sc2_baseline=True)."""
from __future__ import annotations

import pytest

from omega_lock import KCThresholds, P1Config, run_p1
from omega_lock.keyholes.phantom import PhantomKeyhole


def test_sc2_advisory_absent_by_default():
    """Default: run_sc2_baseline=False → no SC-2 entry in kc_reports."""
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    names = {k["name"] for k in r.kc_reports}
    assert "SC-2" not in names


def test_sc2_advisory_appears_when_enabled():
    """run_sc2_baseline=True → kc_reports contains an SC-2 entry with status=ADVISORY."""
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        run_sc2_baseline=True,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    sc2 = next((k for k in r.kc_reports if k["name"] == "SC-2"), None)
    assert sc2 is not None
    assert sc2["status"] == "ADVISORY"
    assert "grid_top_q" in sc2["message"]
    assert "ratio" in sc2["message"]


def test_sc2_advisory_does_not_fail_status():
    """Even if SC-2 fails the 1.5x threshold, overall status must not be FAIL."""
    # Run SC-2 with a seed where grid may not beat random by 1.5x (small decoy coupling)
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        run_sc2_baseline=True,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    # Status should only depend on hard KCs (1-4), not SC-2
    sc2 = next(k for k in r.kc_reports if k["name"] == "SC-2")
    assert sc2["status"] != "FAIL"
    # The overall run status should be unaffected by SC-2's advisory verdict
    hard_fails = [k for k in r.kc_reports if k["status"] == "FAIL"]
    if not hard_fails:
        assert r.status == "PASS"
    else:
        assert r.status.startswith("FAIL:")
        # SC-2 must NOT appear in the failure status list
        assert "SC-2" not in r.status


def test_sc2_detail_contains_ratio_and_flag():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        run_sc2_baseline=True,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    sc2 = next(k for k in r.kc_reports if k["name"] == "SC-2")
    d = sc2["detail"]
    assert "grid_top_quartile" in d
    assert "random_top_quartile" in d
    assert "ratio" in d
    assert "sc2_pass" in d


def test_sc2_wired_through_iterative_config():
    """IterativeConfig.run_sc2_baseline must propagate to per-round P1Config."""
    from omega_lock import IterativeConfig, run_p1_iterative

    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        config=IterativeConfig(
            rounds=1, per_round_unlock_k=3, grid_points_per_axis=5,
            run_sc2_baseline=True,
            kc_thresholds=KCThresholds(trade_count_min=50),
            stress_verbose=False, grid_verbose=False,
        ),
    )
    assert len(r.rounds) == 1
    # Round 1's kc_reports must include an SC-2 advisory
    round1 = r.rounds[0]
    sc2 = next((k for k in round1.kc_reports if k["name"] == "SC-2"), None)
    assert sc2 is not None, "SC-2 advisory missing — IterativeConfig not wired"
    assert sc2["status"] == "ADVISORY"
    # And the advisory must not cause any round to FAIL — overall PASS
    assert r.final_status == "PASS"
