"""Tests for holdout_target — single-shot out-of-sample verification.

Holdout defense: test_target is used across iterative rounds for lock-in
decisions (implicit selection via KC-4). A third target that is NEVER
consulted during the run provides independent generalization evidence.
The framework must: (a) evaluate holdout exactly once per run, (b) never
use holdout for any KC decision, (c) surface the result in P1Result /
IterativeResult.
"""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock import (
    EvalResult,
    IterativeConfig,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1,
    run_p1_iterative,
)
from omega_lock.keyholes.phantom import PhantomKeyhole


class CountingTarget:
    """Wraps PhantomKeyhole to count how many times evaluate() is called."""
    def __init__(self, seed: int):
        self.inner = PhantomKeyhole(seed=seed)
        self.calls = 0

    def param_space(self) -> list[ParamSpec]:
        return self.inner.param_space()

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        self.calls += 1
        return self.inner.evaluate(params)


# ── run_p1 with holdout ──

def test_run_p1_holdout_evaluated_exactly_once():
    """holdout_target.evaluate must be called exactly once per run_p1."""
    train = PhantomKeyhole(seed=42)
    test = PhantomKeyhole(seed=1337)
    ho = CountingTarget(seed=9)

    run_p1(
        train_target=train,
        test_target=test,
        holdout_target=ho,
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    assert ho.calls == 1, f"holdout should be called exactly once, was {ho.calls}"


def test_run_p1_holdout_result_contains_fitness_and_trials():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    assert r.holdout_result is not None
    assert "fitness" in r.holdout_result
    assert "n_trials" in r.holdout_result
    assert "params" in r.holdout_result
    assert "fitness_vs_train" in r.holdout_result
    assert "fitness_vs_test" in r.holdout_result


def test_run_p1_without_holdout_gives_none():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    assert r.holdout_result is None


def test_run_p1_holdout_does_not_affect_status():
    """A catastrophic holdout failure must not change the reported status."""
    class SabotageHoldout:
        def param_space(self):
            return PhantomKeyhole(seed=42).param_space()
        def evaluate(self, params):
            return EvalResult(fitness=-999999.0, n_trials=0)

    r_with = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=SabotageHoldout(),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    r_without = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    assert r_with.status == r_without.status
    # KC statuses must match (elapsed times will legitimately differ).
    statuses_with = [(k["name"], k["status"]) for k in r_with.kc_reports]
    statuses_without = [(k["name"], k["status"]) for k in r_without.kc_reports]
    assert statuses_with == statuses_without


# ── run_p1_iterative with holdout ──

def test_iterative_holdout_not_touched_during_rounds():
    """holdout.evaluate must not be called inside any round — only once at the end."""
    train = PhantomKeyhole(seed=42)
    test = PhantomKeyhole(seed=1337)
    ho = CountingTarget(seed=9)

    r = run_p1_iterative(
        train_target=train,
        test_target=test,
        holdout_target=ho,
        config=IterativeConfig(rounds=3, per_round_unlock_k=3, min_improvement=0.5,
                                kc_thresholds=KCThresholds(trade_count_min=50)),
    )
    assert ho.calls == 1, (
        f"holdout should be called exactly once (post-rounds), was {ho.calls} "
        f"after {len(r.rounds)} rounds"
    )


def test_iterative_holdout_result_uses_final_baseline():
    """Holdout params must be the final_baseline (all locked values)."""
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=IterativeConfig(rounds=2, per_round_unlock_k=3, min_improvement=0.5,
                                kc_thresholds=KCThresholds(trade_count_min=50)),
    )
    assert r.holdout_result is not None
    assert r.holdout_result["params"] == r.final_baseline


def test_iterative_holdout_absent_when_no_rounds_run():
    """If KC-2 fails on round 1 stress and no rounds complete, holdout should
    still behave sanely (absent or safe)."""
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        holdout_target=PhantomKeyhole(seed=9),
        config=IterativeConfig(
            rounds=1, per_round_unlock_k=3,
            # Absurd KC-2 so round 1 fails
            kc_thresholds=KCThresholds(trade_count_min=50, gini_min=0.9999,
                                         top_bot_ratio_min=10**9),
        ),
    )
    # Either holdout wasn't computed (safer) or was computed once — but not multiple times
    # The key invariant: if rounds is empty, holdout_result should be None
    if not r.rounds:
        assert r.holdout_result is None
