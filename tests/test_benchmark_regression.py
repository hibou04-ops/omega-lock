"""Benchmark regression guard — detect silent drift in calibration quality.

This test runs a compact, deterministic subset of the benchmark battery
and compares the resulting metrics against a frozen gold baseline. Any
meaningful change in method behavior (stress ordering, grid convergence,
iterative lock-in choices, etc.) will move one of these numbers outside
tolerance and fail this test.

Regenerating the gold:
    Run `python -m pytest tests/test_benchmark_regression.py
    --update-gold` (see the `UPDATE_GOLD` env var handling below) OR
    manually: `python -c "from tests.test_benchmark_regression import
    _regenerate_gold; _regenerate_gold()"` and commit the updated file.
    Do this ONLY when you intentionally changed a method's behavior and
    the new numbers are the ones you want to lock in.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from omega_lock import (
    BenchmarkSpec, CalibrationMethod, IterativeConfig, KCThresholds,
    P1Config, run_benchmark, run_p1, run_p1_iterative,
)
from omega_lock.keyholes.phantom import PhantomKeyhole
from omega_lock.keyholes.phantom_deep import PhantomKeyholeDeep


GOLD_PATH = Path(__file__).parent / "fixtures" / "benchmark_gold.json"

# Metrics whose values are deterministic given the target + seed + method.
# Walltime and sample_efficiency depend on wall clock, so they're excluded.
DETERMINISTIC_METRIC_KEYS: tuple[str, ...] = (
    "effective_recall",
    "effective_precision",
    "param_L2_error",
    "fitness_gap_pct",
    "generalization_gap",
    "stress_rank_spearman",
    "status_pass",
    "n_evaluations",
    "found_fitness",
)

# Tolerance: float metrics must match within this.
ABSOLUTE_TOL = 1e-6

FIXED_SEEDS = [42, 7]
KC = KCThresholds(trade_count_min=50)


def _runner_plain_grid(target, seed):
    t0 = time.time()
    r = run_p1(
        train_target=target,
        test_target=type(target)(seed=seed + 1),
        config=P1Config(unlock_k=3, grid_points_per_axis=5,
                        kc_thresholds=KC, stress_verbose=False, grid_verbose=False),
    )
    gb = r.grid_best
    return {
        "found_params": dict(gb["unlocked"]) if gb else {},
        "found_fitness": gb["fitness"] if gb else float("-inf"),
        "train_fitness": gb["fitness"] if gb else float("-inf"),
        "test_fitness": (
            r.walk_forward["test_fitnesses"][0]
            if r.walk_forward and r.walk_forward.get("test_fitnesses") else None
        ),
        "unlocked": list(r.top_k),
        "stress_ranking": sorted(
            [s["name"] for s in r.stress_results],
            key=lambda n: -next(s["raw_stress"] for s in r.stress_results if s["name"] == n),
        ),
        "status": r.status,
        "n_evaluations": len(r.stress_results) * 2 + 1 + len(r.grid_results),
        "walltime_s": time.time() - t0,
    }


def _runner_fractal_vise(target, seed):
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
    all_locked = [n for rd in r.locked_in_order for n in rd]
    last = r.rounds[-1] if r.rounds else None
    final_fitness = r.round_best_fitness[-1] if r.round_best_fitness else float("-inf")
    n_eval_total = sum(
        len(rd.stress_results) * 2 + 1 + len(rd.grid_results) for rd in r.rounds
    )
    stress_ranking = None
    if r.rounds:
        r1 = r.rounds[0]
        stress_ranking = sorted(
            [s["name"] for s in r1.stress_results],
            key=lambda n: -next(s["raw_stress"] for s in r1.stress_results if s["name"] == n),
        )
    return {
        "found_params": dict(r.final_baseline),
        "found_fitness": final_fitness,
        "train_fitness": final_fitness,
        "test_fitness": (
            last.walk_forward["test_fitnesses"][0]
            if last and last.walk_forward and last.walk_forward.get("test_fitnesses") else None
        ),
        "unlocked": all_locked,
        "stress_ranking": stress_ranking,
        "status": r.final_status,
        "n_evaluations": n_eval_total,
        "walltime_s": walltime,
    }


def _build_report():
    methods = [
        CalibrationMethod("plain_grid",   _runner_plain_grid),
        CalibrationMethod("fractal_vise", _runner_fractal_vise),
    ]
    specs = [
        BenchmarkSpec("PhantomKeyhole",     PhantomKeyhole,     seeds=FIXED_SEEDS),
        BenchmarkSpec("PhantomKeyholeDeep", PhantomKeyholeDeep, seeds=FIXED_SEEDS),
    ]
    return run_benchmark(specs, methods)


def _extract_deterministic(report) -> dict[str, dict[str, Any]]:
    """Return {row_key: {metric_key: value}} dict for comparison."""
    out: dict[str, dict[str, Any]] = {}
    for row in report.rows:
        key = f"{row.keyhole}|{row.method}|seed={row.seed}"
        out[key] = {k: getattr(row, k) for k in DETERMINISTIC_METRIC_KEYS}
    return out


def _regenerate_gold() -> None:
    """One-shot: rewrite the gold file from the current behavior."""
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report()
    gold = _extract_deterministic(report)
    GOLD_PATH.write_text(json.dumps(gold, indent=2, sort_keys=True))
    print(f"Regenerated gold: {GOLD_PATH} ({len(gold)} rows)")


def test_benchmark_regression_against_gold():
    """Current run must match the frozen gold baseline on deterministic metrics.

    If this test fails, either:
      (a) You intentionally changed a method's behavior → regenerate gold
          via `_regenerate_gold()` and commit the new fixture.
      (b) You accidentally changed something → investigate the drift.
    """
    if os.environ.get("OMEGA_LOCK_UPDATE_GOLD") == "1":
        _regenerate_gold()

    assert GOLD_PATH.exists(), (
        f"gold fixture missing: {GOLD_PATH}. "
        f"Run `OMEGA_LOCK_UPDATE_GOLD=1 pytest tests/test_benchmark_regression.py` "
        f"once to generate it."
    )

    gold = json.loads(GOLD_PATH.read_text())
    report = _build_report()
    current = _extract_deterministic(report)

    # Row set must match exactly (no new/missing (keyhole, method, seed) combos)
    assert set(current.keys()) == set(gold.keys()), (
        f"row set drift — added: {set(current) - set(gold)}, "
        f"removed: {set(gold) - set(current)}"
    )

    diffs: list[str] = []
    for key, gold_metrics in gold.items():
        cur_metrics = current[key]
        for mkey, gold_v in gold_metrics.items():
            cur_v = cur_metrics.get(mkey)
            # None equality (e.g. stress_rank_spearman None)
            if gold_v is None and cur_v is None:
                continue
            if isinstance(gold_v, float) and isinstance(cur_v, float):
                if abs(gold_v - cur_v) > ABSOLUTE_TOL:
                    diffs.append(
                        f"  {key} / {mkey}: gold={gold_v:.9f} vs current={cur_v:.9f} "
                        f"(Δ={cur_v - gold_v:+.2e})"
                    )
            elif gold_v != cur_v:
                diffs.append(f"  {key} / {mkey}: gold={gold_v!r} vs current={cur_v!r}")

    assert not diffs, (
        f"benchmark regression detected ({len(diffs)} drifted metrics):\n"
        + "\n".join(diffs[:20])  # cap at 20 to keep output readable
        + (f"\n... and {len(diffs) - 20} more" if len(diffs) > 20 else "")
    )
