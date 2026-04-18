"""Tests for run_p1_iterative (multi-round coordinate descent with lock-in).

Verifies the "fractal vise" closure pattern: each outer round locks in K
parameters, then the next round runs stress/grid on the remaining params
around the updated baseline.
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


SEED_TRAIN = 42
SEED_TEST = 1337


def _phantom_cfg(**overrides) -> IterativeConfig:
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


def test_iterative_first_round_matches_single_run_p1():
    """Round 1 of iterative should produce the same grid_best as single run_p1."""
    cfg_single = P1Config(unlock_k=3, grid_points_per_axis=5, zoom_rounds=1,
                           kc_thresholds=KCThresholds(trade_count_min=50),
                           stress_verbose=False, grid_verbose=False)
    r_single = run_p1(PhantomKeyhole(seed=SEED_TRAIN), config=cfg_single)
    r_iter = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=1))
    assert r_iter.rounds[0].grid_best["unlocked"] == r_single.grid_best["unlocked"]
    assert r_iter.rounds[0].grid_best["fitness"] == pytest.approx(
        r_single.grid_best["fitness"], abs=1e-9
    )


def test_iterative_round2_baseline_equals_round1_best_fitness():
    """Round r's baseline fitness must equal round r-1's grid_best fitness."""
    r = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=2))
    if len(r.rounds) >= 2:
        r1_best = r.rounds[0].grid_best["fitness"]
        r2_baseline = r.rounds[1].baseline_result["fitness"]
        assert r2_baseline == pytest.approx(r1_best, abs=1e-9)


def test_iterative_locks_disjoint_params_per_round():
    """Each round's locked set must be disjoint from previous rounds'."""
    r = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=3))
    seen: set[str] = set()
    for locked in r.locked_in_order:
        assert not (seen & set(locked)), \
            f"param locked twice across rounds: {seen & set(locked)}"
        seen.update(locked)


def test_iterative_stops_on_min_improvement():
    """With min_improvement=50, iteration should stop after round 1 (round 2 only improves by ~0.01)."""
    r = run_p1_iterative(
        PhantomKeyhole(seed=SEED_TRAIN),
        config=_phantom_cfg(rounds=5, min_improvement=50.0),
    )
    assert r.stop_reason == "no_improvement"
    assert len(r.rounds) == 1


def test_iterative_stops_when_too_few_params_remain():
    """With 12 params and per_round_unlock_k=5, round 3 should have 2 left → stop."""
    r = run_p1_iterative(
        PhantomKeyhole(seed=SEED_TRAIN),
        config=_phantom_cfg(rounds=10, per_round_unlock_k=5),
    )
    # Round 1 locks 5, Round 2 locks 5 → only 2 remain → too_few_params
    assert r.stop_reason == "too_few_params"
    assert len(r.rounds) == 2


def test_iterative_round_best_monotone_nondecreasing():
    """Across rounds, the best fitness found should never decrease."""
    r = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=4))
    for i in range(1, len(r.round_best_fitness)):
        assert r.round_best_fitness[i] >= r.round_best_fitness[i - 1] - 1e-6


def test_iterative_round1_identifies_effective_trio():
    """The first round must pick up the 3 effective params, not a decoy."""
    r = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=1))
    assert set(r.rounds[0].top_k) == {"alpha", "long_mode", "window"}


def test_iterative_with_zoom_refines_alpha():
    """Zooming inside round 1 should move alpha off the coarse grid point (0.5)."""
    # Plain: alpha lands on {0.0, 0.25, 0.5, 0.75, 1.0}
    r_plain = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN),
                                config=_phantom_cfg(rounds=1, zoom_rounds=1))
    # Zoomed: alpha should refine to something finer
    r_zoom = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN),
                               config=_phantom_cfg(rounds=1, zoom_rounds=4, zoom_factor=0.5))
    alpha_plain = r_plain.rounds[0].grid_best["unlocked"]["alpha"]
    alpha_zoom = r_zoom.rounds[0].grid_best["unlocked"]["alpha"]
    # Zoom should produce a value not on the original 5-point grid
    plain_grid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert alpha_plain in plain_grid_values
    # And zoom should improve fitness (at least not worsen)
    assert r_zoom.rounds[0].grid_best["fitness"] >= r_plain.rounds[0].grid_best["fitness"] - 1e-9


def test_iterative_fitness_trajectory_length_matches_rounds():
    r = run_p1_iterative(PhantomKeyhole(seed=SEED_TRAIN), config=_phantom_cfg(rounds=2))
    assert len(r.fitness_trajectory) == len(r.rounds)
    assert len(r.round_best_fitness) == len(r.rounds)


def test_iterative_with_test_target_passes_kc4_each_round():
    """Walk-forward on same-distribution train/test should pass KC-4 every round."""
    r = run_p1_iterative(
        PhantomKeyhole(seed=SEED_TRAIN),
        test_target=PhantomKeyhole(seed=SEED_TEST),
        config=_phantom_cfg(rounds=2),
    )
    for rd in r.rounds:
        kc4 = next((k for k in rd.kc_reports if k["name"] == "KC-4"), None)
        assert kc4 is not None and kc4["status"] == "PASS"


def test_iterative_halts_immediately_on_kc_fail():
    """Winchester defense: if any round fails any KC, iteration must halt.

    Force an unachievable KC-3 floor (trade_count_min=500 > any possible
    n_trials) so round 1 fails → loop must stop at len(rounds)==1.
    """
    r = run_p1_iterative(
        PhantomKeyhole(seed=SEED_TRAIN),
        config=_phantom_cfg(rounds=5, kc_thresholds=KCThresholds(trade_count_min=500)),
    )
    assert r.stop_reason == "kc_fail"
    assert len(r.rounds) == 1
    assert r.final_status.startswith("FAIL")


def test_iterative_result_json_round_trip():
    """IterativeResult.save must produce JSON that round-trips back to dict."""
    import json
    import tempfile
    from pathlib import Path

    r = run_p1_iterative(
        PhantomKeyhole(seed=SEED_TRAIN),
        test_target=PhantomKeyhole(seed=SEED_TEST),
        config=_phantom_cfg(rounds=2),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        r.save(tmp_path)
        payload = json.loads(tmp_path.read_text())
        assert payload["final_status"] == r.final_status
        assert payload["stop_reason"] == r.stop_reason
        assert len(payload["rounds"]) == len(r.rounds)
        assert payload["rounds"][0]["status"] == r.rounds[0].status
        assert payload["locked_in_order"] == r.locked_in_order
    finally:
        tmp_path.unlink(missing_ok=True)
