"""PhantomKeyhole demo — end-to-end P1 pipeline on a synthetic black box.

Unlike the 2D Rosenbrock toy, PhantomKeyhole exercises the FULL Omega-Lock
pipeline with realistic structure:

    - 12 parameters (float / int / bool mix) — nominal dim
    - 3 effective params, 9 decoys (one `ofi_biased=True`) — effective dim ≪ nominal
    - Deterministic per-seed synthetic series (AR(1) latent + noise)
    - Train and test instances are different seeds — walk-forward & KC-4 apply
    - Hybrid validation against a third seed — exercises HybridFitness path

Expected outcome on the tuned seeds below:
    - KC-1 / KC-2 / KC-3 / KC-4 all PASS
    - stress ranking top-3 = {alpha, long_mode, window}
    - grid best ≈ (alpha=0.5, long_mode=True, window=3), fitness ≈ +12
    - walk-forward Pearson ≈ 0.87

Note on `trade_ratio_scaled ≈ 3.7`: the top-10 train grid is dominated by
sparse-fire configs (~20-30 trials) that hit a high-label-correlation
regime; test-best of train-best then fires ~86 times. Ratio passes the
default 0.5 floor comfortably but would trip a strict `trade_ratio_min > 1.0`.
Not a defect — an artifact of the rare-event fitness structure.

Run:
    python examples/phantom_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import (
    IterativeConfig,
    KCThresholds,
    P1Config,
    run_p1,
    run_p1_iterative,
)
from omega_lock.keyholes.phantom import PhantomKeyhole


TRAIN_SEED = 42
TEST_SEED = 1337
JUDGE_SEED = 7


def main() -> int:
    train = PhantomKeyhole(seed=TRAIN_SEED)
    test = PhantomKeyhole(seed=TEST_SEED)
    judge = PhantomKeyhole(seed=JUDGE_SEED)

    cfg = P1Config(
        unlock_k=3,
        grid_points_per_axis=5,
        walk_forward_top_n=10,
        trade_ratio_scale=1.0,      # train and test share n_events
        kc_thresholds=KCThresholds(
            trade_count_min=50,
            gini_min=0.2,
            top_bot_ratio_min=2.0,
            pearson_min=0.3,
            trade_ratio_min=0.5,
        ),
        stress_verbose=True,
        grid_verbose=True,
    )

    output_path = HERE.parent / "output" / "phantom_run.json"
    result = run_p1(
        train_target=train,
        test_target=test,
        validation_target=judge,
        config=cfg,
        output_path=output_path,
    )

    print("\n-- PhantomKeyhole P1 summary --")
    print(f"  status:   {result.status}")
    print(f"  elapsed:  {result.elapsed_seconds:.2f}s")
    print(f"  baseline: fitness={result.baseline_result['fitness']:.2f} n_trials={result.baseline_result['n_trials']}")
    print(f"  top_k:    {result.top_k}")
    if result.grid_best:
        gb = result.grid_best
        print(f"  grid_best: {gb['unlocked']} fitness={gb['fitness']:.2f} n_trials={gb['n_trials']}")
    if result.walk_forward:
        wf = result.walk_forward
        print(f"  walk_forward: pearson={wf['pearson']:.3f} trade_ratio={wf['trade_ratio_scaled']:.3f}")
    if result.hybrid_top:
        h = result.hybrid_top[0]
        print(f"  hybrid_top[0]: params={h['params']} final_fitness={h['final_fitness']:.2f}")
    print("  KC reports:")
    for kc in result.kc_reports:
        print(f"    [{kc['status']:4s}] {kc['name']}: {kc['message']}")

    print("  stress top-3 vs bottom-3:")
    sorted_stress = sorted(result.stress_results, key=lambda s: -s["raw_stress"])
    for s in sorted_stress[:3]:
        tag = " OFI" if s["ofi_biased"] else ""
        print(f"    {s['name']:15s} raw={s['raw_stress']:10.4f}{tag}")
    print("    ...")
    for s in sorted_stress[-3:]:
        tag = " OFI" if s["ofi_biased"] else ""
        print(f"    {s['name']:15s} raw={s['raw_stress']:10.6f}{tag}")

    print(f"\n  output: {output_path}")

    assert result.status == "PASS", f"expected PASS, got {result.status}"
    assert set(result.top_k) == {"alpha", "long_mode", "window"}, \
        f"effective trio not identified: top_k = {result.top_k}"
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["long_mode"] is True, "grid should prefer long_mode=True"
    assert result.walk_forward is not None
    assert result.walk_forward["pearson"] >= 0.3

    # ── Fractal-vise mode: iterative + zooming ────────────────────────────
    # Same target, but run coordinate-descent with per-round zooming to
    # refine the discrete grid result. Each outer round locks K params;
    # inside each round the ZoomingGridSearch narrows the axes geometrically
    # around the running winner. This produces smoother (non-grid-aligned)
    # parameter values.
    print("\n-- PhantomKeyhole fractal-vise (iterative + zoom) --")
    fractal_cfg = IterativeConfig(
        rounds=2,
        per_round_unlock_k=3,
        grid_points_per_axis=5,
        walk_forward_top_n=10,
        trade_ratio_scale=1.0,
        kc_thresholds=KCThresholds(trade_count_min=50),
        # stop_on_kc_fail stays at its default (True). Multi-round does NOT
        # relax KC thresholds — a failed round halts the loop. This is the
        # Winchester defense ported to coordinate descent.
        min_improvement=0.5,
        zoom_rounds=4,
        zoom_factor=0.5,
        stress_verbose=False,
        grid_verbose=False,
    )
    fractal_output = HERE.parent / "output" / "phantom_fractal_run.json"
    iter_result = run_p1_iterative(
        train_target=PhantomKeyhole(seed=TRAIN_SEED),
        test_target=PhantomKeyhole(seed=TEST_SEED),
        config=fractal_cfg,
        output_path=fractal_output,
    )
    print(f"  final_status:  {iter_result.final_status}")
    print(f"  stop_reason:   {iter_result.stop_reason}")
    print(f"  rounds run:    {len(iter_result.rounds)}")
    print(f"  locked order:  {iter_result.locked_in_order}")
    print(f"  round_best:    {[round(f, 4) for f in iter_result.round_best_fitness]}")
    eff = {k: iter_result.final_baseline[k] for k in ("alpha", "long_mode", "window")}
    print(f"  refined effective: {eff}")
    print(f"  (plain coarse grid picked alpha={result.grid_best['unlocked']['alpha']}, "
          f"fractal picked alpha={eff['alpha']:.4f})")
    print(f"  output: {fractal_output}")

    assert iter_result.rounds[0].grid_best["fitness"] >= result.grid_best["fitness"] - 1e-6, \
        "fractal-vise should never produce worse fitness than plain grid"

    print("\nPhantomKeyhole demo PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
