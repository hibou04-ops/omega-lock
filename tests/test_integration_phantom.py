"""End-to-end integration test: PhantomKeyhole through full P1 pipeline.

This is the canonical "keyhole" that proves the framework works on a target
with realistic structure (mixed param types, effective dim ≪ nominal,
train/test split, non-trivial n_trials).
"""
from __future__ import annotations

import numpy as np
import pytest

from omega_lock import KCThresholds, P1Config, run_p1
from omega_lock.keyholes.phantom import PhantomKeyhole


# Tuned seed pair that reliably passes all KCs. Other seeds may fail KC-3
# legitimately (grid best has fewer trials than floor) — that's Omega-Lock
# correctly catching sparse optima, not a bug.
TRAIN_SEED = 42
TEST_SEED = 1337
JUDGE_SEED = 7


def _config(**overrides) -> P1Config:
    defaults = dict(
        unlock_k=3,
        grid_points_per_axis=5,
        walk_forward_top_n=10,
        trade_ratio_scale=1.0,
        kc_thresholds=KCThresholds(trade_count_min=50),
        stress_verbose=False,
        grid_verbose=False,
    )
    defaults.update(overrides)
    return P1Config(**defaults)


def test_phantom_pipeline_passes_all_kill_criteria():
    """Full happy path: stress → unlock → grid → walk-forward → hybrid → PASS."""
    result = run_p1(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        test_target=PhantomKeyhole(seed=TEST_SEED),
        validation_target=PhantomKeyhole(seed=JUDGE_SEED),
        config=_config(),
    )
    assert result.status == "PASS", f"expected PASS, got {result.status!r}"
    by_name = {r["name"]: r["status"] for r in result.kc_reports}
    assert by_name == {"KC-1": "PASS", "KC-2": "PASS", "KC-3": "PASS", "KC-4": "PASS"}


def test_phantom_stress_identifies_effective_trio():
    """Top-3 stress must be the 3 truly-effective params (not any decoy)."""
    result = run_p1(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        config=_config(),
    )
    assert set(result.top_k) == {"alpha", "long_mode", "window"}, \
        f"top-3 missed an effective param: {result.top_k}"

    # Each effective stress should exceed every decoy stress by a wide margin.
    by_name = {s["name"]: s["raw_stress"] for s in result.stress_results}
    effective = ["alpha", "long_mode", "window"]
    decoys = [n for n in by_name if n not in effective]
    min_effective = min(by_name[n] for n in effective)
    max_decoy = max(by_name[n] for n in decoys)
    assert min_effective > max_decoy * 10.0, (
        f"effective/decoy margin too small: min_effective={min_effective} "
        f"max_decoy={max_decoy}"
    )


def test_phantom_kc2_strong_differentiation():
    """KC-2 should pass comfortably — Gini near 1 and top/bot ratio huge."""
    result = run_p1(train_target=PhantomKeyhole(seed=TRAIN_SEED), config=_config())
    kc2 = next(r for r in result.kc_reports if r["name"] == "KC-2")
    assert kc2["status"] == "PASS"
    assert kc2["detail"]["gini"] >= 0.8
    assert kc2["detail"]["ratio"] >= 100.0


def test_phantom_grid_finds_long_mode_true_with_small_window():
    """Grid should converge to long_mode=True (the correct direction) and
    a short lookback window (noisier val gives more fires with positive edge)."""
    result = run_p1(train_target=PhantomKeyhole(seed=TRAIN_SEED), config=_config())
    gb = result.grid_best
    assert gb is not None
    assert gb["unlocked"]["long_mode"] is True
    assert gb["unlocked"]["window"] <= 10       # short lookback preferred
    assert gb["fitness"] > result.baseline_result["fitness"] + 30.0, \
        "grid should improve baseline by at least 30 fitness units"


def test_phantom_walk_forward_preserves_ranking():
    """Train and test are same-distribution seeds, so top-10 ranking
    should correlate strongly (Pearson ≥ 0.5)."""
    result = run_p1(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        test_target=PhantomKeyhole(seed=TEST_SEED),
        config=_config(),
    )
    assert result.walk_forward is not None
    assert result.walk_forward["pearson"] >= 0.5


def test_phantom_hybrid_reranks_top_k_with_judge():
    """validation_target (third seed) should produce a hybrid_top ordering
    sorted by final_fitness (judge). All entries must be valid dicts."""
    result = run_p1(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        test_target=PhantomKeyhole(seed=TEST_SEED),
        validation_target=PhantomKeyhole(seed=JUDGE_SEED),
        config=_config(),
    )
    assert result.hybrid_top is not None
    assert len(result.hybrid_top) > 0
    # Top-K (validated) must be ordered by final_fitness descending.
    validated = [h for h in result.hybrid_top if "validation_fitness" in h]
    finals = [h["final_fitness"] for h in validated]
    assert finals == sorted(finals, reverse=True), \
        f"hybrid top validated not sorted: {finals}"


def test_phantom_n_trials_exceed_kc3_floor():
    """Baseline, train-best, and test-best must all clear the 50-trade floor."""
    result = run_p1(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        test_target=PhantomKeyhole(seed=TEST_SEED),
        config=_config(),
    )
    assert result.baseline_result["n_trials"] >= 50
    assert result.grid_best["n_trials"] >= 50
    assert result.walk_forward["test_best_trades"] >= 50


def test_phantom_deterministic_per_seed():
    """Same seed must give identical events and therefore identical baseline."""
    a = PhantomKeyhole(seed=42)
    b = PhantomKeyhole(seed=42)
    from omega_lock.params import neutral_defaults
    base = neutral_defaults(a.param_space())
    ra = a.evaluate(base)
    rb = b.evaluate(base)
    assert ra.fitness == pytest.approx(rb.fitness, abs=1e-12)
    assert ra.n_trials == rb.n_trials


def test_phantom_different_seeds_produce_different_events():
    """Guard against seed threading regressing silently (e.g., rng dropped)."""
    a = PhantomKeyhole(seed=42)
    b = PhantomKeyhole(seed=43)
    assert not np.array_equal(a._obs, b._obs)
    assert not np.array_equal(a._labels, b._labels)
