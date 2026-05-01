# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P2: iterative advisory fields + structured GeneralizationGap.

Two changes pinned here:

1. ``IterativeResult`` now exposes the test-target reuse risk in
   fields the artifact reader can't miss:
     - selection_reused_test_target (bool)
     - test_reuse_rounds (int)
     - holdout_recommended (bool)
     - holdout_present (bool)
     - advisory_messages (list[str])

   The per-round KC-4 numbers look fine on the surface, but their
   evidence weight degrades because the same test_target is consulted
   for selection at every locking step. These flags surface that.

2. ``compute_generalization_gap_status`` returns a ``GeneralizationGap``
   carrying status ∈ {OK, TRAIN_NEAR_ZERO, NO_TEST}. The float-only
   ``compute_generalization_gap`` wrapper is preserved for backward
   compat.
"""
from __future__ import annotations

from typing import Any

from omega_lock import (
    EvalResult,
    IterativeConfig,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1_iterative,
)
from omega_lock.benchmark import (
    GeneralizationGap,
    compute_generalization_gap,
    compute_generalization_gap_status,
)
from omega_lock.keyholes.phantom import PhantomKeyhole


# ---------------------------------------------------------------------------
# IterativeResult advisory fields.
# ---------------------------------------------------------------------------


def _iter_config(rounds: int = 3) -> IterativeConfig:
    # KC thresholds set permissively + stop_on_kc_fail=False so the loop
    # actually runs the full requested rounds for advisory testing.
    return IterativeConfig(
        rounds=rounds,
        per_round_unlock_k=2,
        grid_points_per_axis=3,
        kc_thresholds=KCThresholds(
            trade_count_min=1,
            gini_min=0.0,
            top_bot_ratio_min=0.0,
            pearson_min=-1.0,
            trade_ratio_min=0.0,
        ),
        stop_on_kc_fail=False,
        min_improvement=-1e9,  # never stop on no-improvement
        stress_verbose=False,
        grid_verbose=False,
    )


def test_iterative_advisory_marks_test_reuse_when_test_target_present():
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=_iter_config(),
    )
    assert r.selection_reused_test_target is True
    assert r.test_reuse_rounds == len(r.rounds)


def test_iterative_advisory_recommends_holdout_when_absent():
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=_iter_config(),
    )
    assert r.holdout_present is False
    assert r.holdout_recommended is True
    assert any("holdout" in msg.lower() for msg in r.advisory_messages)


def test_iterative_advisory_holdout_present_when_supplied():
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=_iter_config(),
    )
    assert r.holdout_present is True
    assert r.holdout_recommended is False
    # Should NOT warn about missing holdout when one was provided.
    assert not any(
        "without holdout_target" in msg for msg in r.advisory_messages
    )


def test_iterative_advisory_warns_about_test_reuse_at_multi_round():
    """When >1 round runs, surface the per-round-KC-4-degradation
    advisory regardless of holdout state."""
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=_iter_config(rounds=3),
    )
    assert any(
        "reused" in msg and "KC-4" in msg for msg in r.advisory_messages
    )


# ---------------------------------------------------------------------------
# GeneralizationGap status.
# ---------------------------------------------------------------------------


def test_gap_ok_for_meaningful_train_fitness():
    gap = compute_generalization_gap_status(train_fitness=10.0, test_fitness=8.0)
    assert isinstance(gap, GeneralizationGap)
    assert gap.status == "OK"
    assert abs(gap.value - 0.2) < 1e-9


def test_gap_no_test_when_test_fitness_is_none():
    gap = compute_generalization_gap_status(train_fitness=10.0, test_fitness=None)
    assert gap.status == "NO_TEST"
    assert gap.value == 0.0


def test_gap_train_near_zero_flagged():
    """Train fitness near 0 produces astronomical gaps that look big
    but mean nothing — status surfaces the issue."""
    gap = compute_generalization_gap_status(
        train_fitness=1e-9, test_fitness=0.5,
    )
    assert gap.status == "TRAIN_NEAR_ZERO"


def test_gap_train_exact_zero_flagged():
    gap = compute_generalization_gap_status(train_fitness=0.0, test_fitness=0.5)
    assert gap.status == "TRAIN_NEAR_ZERO"


def test_legacy_compute_generalization_gap_preserves_float_return():
    """Backward-compat: existing benchmark code that reads the float
    keeps working unchanged."""
    val = compute_generalization_gap(train_fitness=10.0, test_fitness=8.0)
    assert isinstance(val, float)
    assert abs(val - 0.2) < 1e-9


def test_legacy_compute_generalization_gap_no_test_returns_zero():
    val = compute_generalization_gap(train_fitness=10.0, test_fitness=None)
    assert val == 0.0
