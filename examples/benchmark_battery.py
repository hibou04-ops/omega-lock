"""Objective RAGAS-style benchmark: run every method on every keyhole × seeds.

Emits a scorecard with confidence-interval-friendly numbers. Every metric
is mechanically computable from run outputs + ground truth (no human
judgment). Track these across commits to detect regressions.

Run:
    python examples/benchmark_battery.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import (
    BenchmarkSpec, CalibrationMethod, IterativeConfig, KCThresholds, P1Config,
    run_benchmark, run_p1, run_p1_iterative,
)
from omega_lock.keyholes.phantom import PhantomKeyhole
from omega_lock.keyholes.phantom_deep import PhantomKeyholeDeep

try:
    from omega_lock import P2Config, run_p2_tpe
    _TPE_OK = True
except ImportError:
    _TPE_OK = False


SEEDS = [42, 7, 100, 314, 55]
KC = KCThresholds(trade_count_min=50)


# ── Method runners (adapt each pipeline to the benchmark contract) ─────

def _wrap_p1(result, walltime: float) -> dict[str, Any]:
    gb = result.grid_best
    return {
        "found_params": dict(gb["unlocked"]) if gb else {},
        "found_fitness": gb["fitness"] if gb else float("-inf"),
        "train_fitness": gb["fitness"] if gb else float("-inf"),
        "test_fitness": (
            result.walk_forward["test_fitnesses"][0]
            if result.walk_forward and result.walk_forward.get("test_fitnesses")
            else None
        ),
        "unlocked": list(result.top_k),
        "stress_ranking": sorted(
            [r["name"] for r in result.stress_results],
            key=lambda n: -next(
                r["raw_stress"] for r in result.stress_results if r["name"] == n
            ),
        ),
        "status": result.status,
        "n_evaluations": (
            # 1 baseline + 2 * (continuous/int stress) + 1 * (bool stress) + grid_size
            len(result.stress_results) * 2 + 1 + len(result.grid_results)
        ),
        "walltime_s": walltime,
    }


def runner_plain_grid(target, seed):
    t0 = time.time()
    r = run_p1(
        train_target=target,
        test_target=type(target)(seed=seed + 1),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KC, stress_verbose=False, grid_verbose=False),
    )
    return _wrap_p1(r, time.time() - t0)


def runner_fractal_vise(target, seed):
    t0 = time.time()
    r = run_p1_iterative(
        train_target=target,
        test_target=type(target)(seed=seed + 1),
        config=IterativeConfig(
            rounds=3, per_round_unlock_k=3, grid_points_per_axis=5,
            zoom_rounds=4, zoom_factor=0.5, min_improvement=0.5,
            kc_thresholds=KC,
        ),
    )
    walltime = time.time() - t0
    # Flatten locked values for benchmark contract
    all_locked = [n for rd in r.locked_in_order for n in rd]
    last = r.rounds[-1] if r.rounds else None
    final_fitness = r.round_best_fitness[-1] if r.round_best_fitness else float("-inf")
    n_eval_total = sum(
        len(rd.stress_results) * 2 + 1 + len(rd.grid_results) for rd in r.rounds
    )
    # Stress ranking from the first round (neutral baseline; most informative)
    stress_ranking_first_round = None
    if r.rounds:
        r1 = r.rounds[0]
        stress_ranking_first_round = sorted(
            [s["name"] for s in r1.stress_results],
            key=lambda n: -next(
                s["raw_stress"] for s in r1.stress_results if s["name"] == n
            ),
        )
    return {
        "found_params": dict(r.final_baseline),
        "found_fitness": final_fitness,
        "train_fitness": final_fitness,
        "test_fitness": (
            last.walk_forward["test_fitnesses"][0]
            if last and last.walk_forward and last.walk_forward.get("test_fitnesses")
            else None
        ),
        "unlocked": all_locked,
        "stress_ranking": stress_ranking_first_round,
        "status": r.final_status,
        "n_evaluations": n_eval_total,
        "walltime_s": walltime,
    }


def runner_tpe(target, seed):
    if not _TPE_OK:
        raise ImportError("optuna not available")
    t0 = time.time()
    r = run_p2_tpe(
        train_target=target,
        test_target=type(target)(seed=seed + 1),
        config=P2Config(
            unlock_k=3, n_trials=200, seed=seed,
            kc_thresholds=KC,
            stress_verbose=False, trial_verbose=False,
        ),
    )
    walltime = time.time() - t0
    best = r.tpe_best
    test_fit = (
        r.walk_forward["test_fitnesses"][0]
        if r.walk_forward and r.walk_forward.get("test_fitnesses") else None
    )
    return {
        "found_params": dict(best["unlocked"]) if best else {},
        "found_fitness": best["fitness"] if best else float("-inf"),
        "train_fitness": best["fitness"] if best else float("-inf"),
        "test_fitness": test_fit,
        "unlocked": list(r.top_k),
        "stress_ranking": sorted(
            [s["name"] for s in r.stress_results],
            key=lambda n: -next(
                s["raw_stress"] for s in r.stress_results if s["name"] == n
            ),
        ),
        "status": r.status,
        # 1 baseline + 2*nstress + n_trials
        "n_evaluations": len(r.stress_results) * 2 + 1 + len(r.trials),
        "walltime_s": walltime,
    }


def main() -> int:
    methods = [
        CalibrationMethod(name="plain_grid",   runner=runner_plain_grid),
        CalibrationMethod(name="fractal_vise", runner=runner_fractal_vise),
    ]
    if _TPE_OK:
        methods.append(CalibrationMethod(name="optuna_tpe", runner=runner_tpe))

    specs = [
        BenchmarkSpec("PhantomKeyhole",     PhantomKeyhole,     seeds=SEEDS),
        BenchmarkSpec("PhantomKeyholeDeep", PhantomKeyholeDeep, seeds=SEEDS),
    ]

    print(f"Running benchmark: {len(specs)} keyholes × {len(methods)} methods × "
          f"{len(SEEDS)} seeds = {len(specs) * len(methods) * len(SEEDS)} runs\n")

    output = HERE.parent / "output" / "benchmark_report.json"
    report = run_benchmark(specs, methods, output_path=output)

    # Scorecards: one per keyhole for clarity
    for keyhole_name in [s.keyhole_name for s in specs]:
        print(f"\n── Scorecard — {keyhole_name} ──")
        filtered = BenchmarkReport(
            rows=[r for r in report.rows if r.keyhole == keyhole_name]
        )
        print(filtered.render_scorecard())

    # Combined scorecard across both keyholes
    print(f"\n── Scorecard — combined (across {len(specs)} keyholes) ──")
    print(report.render_scorecard())

    print(f"\nFull report JSON: {output}")
    return 0


# Re-import required for per-keyhole filtering
from omega_lock import BenchmarkReport

if __name__ == "__main__":
    sys.exit(main())
