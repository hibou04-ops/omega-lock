"""End-to-end integration test: PhantomKeyholeDeep through iterative P1.

PhantomKeyholeDeep has effective_dim=6 (of 20 params) organized as two
nearly-independent signal channels. With per_round_unlock_k=3, a single
run_p1 only captures one channel's trio — coordinate descent across
rounds is the only way to discover all 6 effectives and push fitness
past the single-round ceiling. These tests prove that pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from omega_lock import (
    IterativeConfig,
    KCThresholds,
    P1Config,
    run_p1,
    run_p1_iterative,
)
from omega_lock.keyholes.phantom_deep import PhantomKeyholeDeep
from omega_lock.params import neutral_defaults


# Ground-truth effectives the calibrator must rediscover.
EFFECTIVES: set[str] = {"alpha", "window", "long_mode", "beta", "horizon", "use_ema"}

# Seed pair that reliably passes all KCs with PhantomKeyholeDeep.
TRAIN_SEED = 42
TEST_SEED = 1337


def _iter_cfg(**overrides) -> IterativeConfig:
    defaults = dict(
        rounds=3,
        per_round_unlock_k=3,
        grid_points_per_axis=5,
        walk_forward_top_n=10,
        trade_ratio_scale=1.0,
        kc_thresholds=KCThresholds(trade_count_min=50),
        stress_verbose=False,
        grid_verbose=False,
    )
    defaults.update(overrides)
    return IterativeConfig(**defaults)


def test_phantom_deep_iterative_finds_most_effectives_across_rounds():
    """Rounds 1+2 together must lock at least 5 of the 6 effective params.

    A single round with unlock_k=3 can only find one channel's trio; the
    second round must surface the *other* channel's effectives after the
    first is locked in. This verifies coordinate descent genuinely
    discovers new axes rather than re-finding the same three.
    """
    r = run_p1_iterative(PhantomKeyholeDeep(seed=TRAIN_SEED), config=_iter_cfg(rounds=2))
    assert len(r.rounds) >= 2, "must complete at least 2 rounds for this test"
    locked_first_two = set(r.locked_in_order[0]) | set(r.locked_in_order[1])
    covered = EFFECTIVES & locked_first_two
    assert len(covered) >= 5, (
        f"iterative must cover >= 5/6 effectives in rounds 1+2, got {len(covered)}: "
        f"covered={covered} missing={EFFECTIVES - covered} locked={r.locked_in_order}"
    )


def test_phantom_deep_round2_passes_kc2():
    """Round 2 stress on the remaining (unlocked) params must still pass KC-2.

    If the remaining effectives really surface after round 1's lock-in
    (as the two-channel design predicts), the Gini on the round-2 stress
    distribution stays high — the three remaining-channel effectives
    dominate the 14 decoys.
    """
    r = run_p1_iterative(
        PhantomKeyholeDeep(seed=TRAIN_SEED),
        test_target=PhantomKeyholeDeep(seed=TEST_SEED),
        config=_iter_cfg(rounds=2),
    )
    assert len(r.rounds) >= 2
    kc2_round2 = next(k for k in r.rounds[1].kc_reports if k["name"] == "KC-2")
    assert kc2_round2["status"] == "PASS", (
        f"KC-2 round 2 must PASS to prove remaining effectives surface: "
        f"{kc2_round2['message']}"
    )
    # A surfaced effective means Gini is clearly differentiated, not marginal.
    assert kc2_round2["detail"]["gini"] >= 0.3, (
        f"round 2 Gini too low for differentiation: {kc2_round2['detail']['gini']}"
    )
    # Walk-forward on the test seed must also hold up in each passing round.
    for idx, rd in enumerate(r.rounds):
        if rd.status != "PASS":
            continue
        assert rd.walk_forward is not None
        assert rd.walk_forward["pearson"] >= 0.3, (
            f"round {idx+1} walk-forward pearson below 0.3: "
            f"{rd.walk_forward['pearson']}"
        )


def test_phantom_deep_iterative_beats_single_round():
    """Two iterative rounds must strictly beat one single run_p1 (unlock_k=3).

    Single run_p1 with unlock_k=3 locks one channel's trio. Iterative
    then unlocks the other channel in round 2 — fitness must increase
    strictly, proving the second round is not a no-op.
    """
    single_cfg = P1Config(
        unlock_k=3,
        grid_points_per_axis=5,
        kc_thresholds=KCThresholds(trade_count_min=50),
        stress_verbose=False,
        grid_verbose=False,
    )
    r_single = run_p1(PhantomKeyholeDeep(seed=TRAIN_SEED), config=single_cfg)
    assert r_single.grid_best is not None

    r_iter = run_p1_iterative(PhantomKeyholeDeep(seed=TRAIN_SEED), config=_iter_cfg(rounds=2))
    assert len(r_iter.rounds) >= 2
    iter_final_fitness = r_iter.round_best_fitness[1]
    single_fitness = r_single.grid_best["fitness"]

    assert iter_final_fitness > single_fitness, (
        f"iterative ({iter_final_fitness:.3f}) must strictly beat single "
        f"run_p1 ({single_fitness:.3f}); coordinate descent is a no-op otherwise"
    )


def test_phantom_deep_deterministic_per_seed():
    """Same seed must produce identical observations and evaluations."""
    a = PhantomKeyholeDeep(seed=42)
    b = PhantomKeyholeDeep(seed=42)
    assert np.array_equal(a._obs_a, b._obs_a)
    assert np.array_equal(a._obs_b, b._obs_b)
    assert np.array_equal(a._labels_a, b._labels_a)
    assert np.array_equal(a._labels_b, b._labels_b)

    base = neutral_defaults(a.param_space())
    ra = a.evaluate(base)
    rb = b.evaluate(base)
    assert ra.fitness == pytest.approx(rb.fitness, abs=1e-12)
    assert ra.n_trials == rb.n_trials


def test_phantom_deep_different_seeds_differ():
    """Different seeds must yield different observations on both channels."""
    a = PhantomKeyholeDeep(seed=42)
    b = PhantomKeyholeDeep(seed=43)
    assert not np.array_equal(a._obs_a, b._obs_a)
    assert not np.array_equal(a._obs_b, b._obs_b)
    assert not np.array_equal(a._labels_a, b._labels_a)
    assert not np.array_equal(a._labels_b, b._labels_b)


def test_phantom_deep_baseline_trials_above_kc3_floor():
    """Neutral defaults must yield n_trials >= 50 (KC-3 floor).

    PhantomKeyholeDeep sums n_trials across both channels, so this is
    easy to clear even with neutrals deliberately placed in poor regions.
    """
    t = PhantomKeyholeDeep(seed=TRAIN_SEED)
    base = neutral_defaults(t.param_space())
    r = t.evaluate(base)
    assert r.n_trials >= 50, (
        f"baseline n_trials={r.n_trials} below KC-3 floor (50); neutrals "
        f"may have been placed in too-restrictive a region"
    )


def test_phantom_deep_iterative_holdout_single_eval_across_multiple_rounds():
    """Holdout must be consulted exactly once even though iteration runs
    multiple lock-in rounds on a target where those rounds actually happen.

    PhantomKeyhole converges in round 1 (effective_dim=3=unlock_k) so its
    holdout test doesn't exercise the multi-round path. PhantomKeyholeDeep
    does — this test is the only coverage where multiple rounds genuinely
    iterate and holdout's once-only contract is checked in that regime.
    """
    class Counter:
        def __init__(self, seed):
            self.inner = PhantomKeyholeDeep(seed=seed)
            self.calls = 0
        def param_space(self):
            return self.inner.param_space()
        def evaluate(self, p):
            self.calls += 1
            return self.inner.evaluate(p)

    ho = Counter(seed=9)
    r = run_p1_iterative(
        train_target=PhantomKeyholeDeep(seed=TRAIN_SEED),
        test_target=PhantomKeyholeDeep(seed=TEST_SEED),
        holdout_target=ho,
        config=_iter_cfg(rounds=2),
    )
    # Must have actually iterated (proving multi-round path was exercised)
    assert len(r.rounds) >= 2, \
        f"test assumes multi-round iteration; got {len(r.rounds)} rounds"
    # Holdout consulted exactly once despite multiple rounds
    assert ho.calls == 1, (
        f"holdout consulted {ho.calls} times after {len(r.rounds)} rounds; "
        f"contract is exactly-once-at-end"
    )
    assert r.holdout_result is not None
    assert r.holdout_result["params"] == r.final_baseline
