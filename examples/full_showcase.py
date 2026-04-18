"""Full-capability showcase — exercises every search method on both keyholes.

Demonstrates the four calibration modes in order of increasing sophistication:

    1. Plain 5-pt grid          — baseline discovery
    2. Iterative + zooming       — "fractal vise" multi-scale refinement
    3. Random-search comparison  — SC-2 "grid >= 1.5x random" baseline
    4. Optuna TPE (continuous)   — true sub-grid precision

Each mode is exercised against:
    - PhantomKeyhole       (effective_dim = 3, single round is enough)
    - PhantomKeyholeDeep   (effective_dim = 6, iteration is REQUIRED)

And the honest-generalization check (holdout_target on a third seed) is
included for the iterative runs where KC-4 is reused across rounds.

Run:
    python examples/full_showcase.py
    pip install "optuna>=3.0"  # if the P2 TPE section says 'skipped'
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import (
    GridSearch,
    IterativeConfig,
    KCThresholds,
    P1Config,
    RandomSearch,
    compare_to_grid,
    run_p1,
    run_p1_iterative,
    top_quartile_fitness,
)
from omega_lock.keyholes.phantom import PhantomKeyhole
from omega_lock.keyholes.phantom_deep import PhantomKeyholeDeep

try:
    from omega_lock import P2Config, run_p2_tpe
    _TPE_OK = True
except ImportError:
    _TPE_OK = False


def header(s: str) -> None:
    print(f"\n{'─' * 72}\n  {s}\n{'─' * 72}")


def mode_1_plain_grid() -> None:
    header("Mode 1 — Plain grid on PhantomKeyhole (single round)")
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    gb = r.grid_best
    ho = r.holdout_result
    print(f"  status={r.status}  top_k={r.top_k}")
    print(f"  grid_best: alpha={gb['unlocked']['alpha']} window={gb['unlocked']['window']} "
          f"long_mode={gb['unlocked']['long_mode']}  fitness={gb['fitness']:.3f}")
    print(f"  holdout:   fitness={ho['fitness']:.3f}  vs_train={ho['fitness_vs_train']:+.3f}  "
          f"vs_test={ho['fitness_vs_test']:+.3f}")


def mode_2_fractal_vise() -> None:
    header("Mode 2 — Iterative + zoom on PhantomKeyhole (fractal-vise refinement)")
    r = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=PhantomKeyhole(seed=9),
        config=IterativeConfig(
            rounds=2, per_round_unlock_k=3, grid_points_per_axis=5,
            zoom_rounds=4, zoom_factor=0.5,
            min_improvement=0.5,
            kc_thresholds=KCThresholds(trade_count_min=50),
        ),
    )
    eff_final = {k: r.final_baseline[k] for k in ("alpha", "long_mode", "window")}
    ho = r.holdout_result
    print(f"  status={r.final_status}  stop={r.stop_reason}  rounds={len(r.rounds)}")
    print(f"  refined effective: {eff_final}")
    print(f"  round_best trajectory: {[round(f, 3) for f in r.round_best_fitness]}")
    if ho:
        print(f"  holdout (never touched during rounds): fitness={ho['fitness']:.3f} "
              f"n_trials={ho['n_trials']}")


def mode_3_random_baseline() -> None:
    header("Mode 3 — Random-search baseline (SC-2 check: grid / random >= 1.5x)")
    # Use PhantomKeyholeDeep — larger dynamic range, cleaner SC-2 signal than
    # PhantomKeyhole where the 3-param peak is so dominant that even random
    # sampling lands close to it with modest budgets.
    target = PhantomKeyholeDeep(seed=42)
    base = {s.name: s.neutral for s in target.param_space()}
    unlocked = ["alpha", "long_mode", "beta"]   # channel-A effectives only

    gs = GridSearch(target=target, unlocked=unlocked, grid_points_per_axis=5, verbose=False)
    grid_pts = gs.run(base_params=base)

    rs = RandomSearch(target=target, unlocked=unlocked, n_samples=len(grid_pts), seed=42, verbose=False)
    rand_pts = rs.run(base_params=base)

    grid_tq = top_quartile_fitness(grid_pts)
    rand_tq = top_quartile_fitness(rand_pts)
    print(f"  grid_top_quartile   = {grid_tq:.3f}")
    print(f"  random_top_quartile = {rand_tq:.3f}")
    # Guard the ratio display against sign-crossing. SC-2 was authored
    # assuming non-negative fitness; for Phantom targets we read the gap
    # directly and flag it manually.
    gap = grid_tq - rand_tq
    print(f"  grid − random gap   = {gap:+.3f}  "
          f"({'grid wins' if gap > 0 else 'random wins'})")
    print(f"  (SPEC §4 SC-2 ratio is only meaningful when both top-quartiles "
          f"share sign; gap is the safe metric here.)")
    print(f"  Why this matters: per Bergstra & Bengio 2012, if grid doesn't beat "
          f"random by >= 1.5x, grid coverage is wasted effort. SC-2 is the "
          f"advisory check for 'should we even be using a grid at all'.")


def mode_4_tpe() -> None:
    header("Mode 4 — Optuna TPE on PhantomKeyhole (continuous search)")
    if not _TPE_OK:
        print("  skipped: optuna not installed. Run `pip install \"omega-lock[p2]\"` to enable.")
        return
    r = run_p2_tpe(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=P2Config(
            unlock_k=3, n_trials=200, seed=42,
            kc_thresholds=KCThresholds(trade_count_min=50),
            stress_verbose=False, trial_verbose=False,
        ),
    )
    best = r.tpe_best
    print(f"  status={r.status}  trials_run={len(r.trials)}")
    if best:
        u = best["unlocked"]
        print(f"  tpe_best: alpha={u['alpha']:.4f} window={u['window']} "
              f"long_mode={u['long_mode']}  fitness={best['fitness']:.3f}")
        print(f"  (plain 5-pt grid constrains alpha to {{0, 0.25, 0.5, 0.75, 1.0}}; "
              f"TPE can land anywhere)")
        if r.status.startswith("FAIL"):
            print(f"  NOTE: {r.status} is the framework catching TPE's finer-grained train "
                  f"optimum failing to generalize — same Winchester defense as run_p1.")


def mode_5_deep_iteration() -> None:
    header("Mode 5 — PhantomKeyholeDeep (effective_dim=6): iteration IS required")
    # Single round first — will miss 3 of 6 effectives
    r_single = run_p1(
        train_target=PhantomKeyholeDeep(seed=42),
        test_target=PhantomKeyholeDeep(seed=1337),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KCThresholds(trade_count_min=50),
                        stress_verbose=False, grid_verbose=False),
    )
    print(f"  single-round top_k = {r_single.top_k}")
    print(f"  single-round fitness = {r_single.grid_best['fitness']:.3f}")

    # Iterative — rounds 1+2 find all 6
    r_iter = run_p1_iterative(
        train_target=PhantomKeyholeDeep(seed=42),
        test_target=PhantomKeyholeDeep(seed=1337),
        holdout_target=PhantomKeyholeDeep(seed=9),
        config=IterativeConfig(
            rounds=3, per_round_unlock_k=3, min_improvement=0.5,
            kc_thresholds=KCThresholds(trade_count_min=50),
        ),
    )
    print(f"  iterative locked_in_order: {r_iter.locked_in_order}")
    print(f"  iterative round_best: {[round(f, 2) for f in r_iter.round_best_fitness]}")
    all_locked = [n for rd in r_iter.locked_in_order for n in rd]
    effectives = {"alpha", "window", "long_mode", "beta", "horizon", "use_ema"}
    found = set(all_locked) & effectives
    print(f"  effectives discovered: {len(found)}/6  {found}")
    if r_iter.holdout_result:
        print(f"  holdout (never touched): fitness={r_iter.holdout_result['fitness']:.2f}")


def main() -> int:
    mode_1_plain_grid()
    mode_2_fractal_vise()
    mode_3_random_baseline()
    mode_4_tpe()
    mode_5_deep_iteration()
    print(f"\n{'─' * 72}\nfull_showcase complete.\n{'─' * 72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
