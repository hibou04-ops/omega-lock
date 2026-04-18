"""Objective benchmark suite for calibration methods.

RAGAS-style scorecard: every metric is mechanically computable from the
run outputs + keyhole ground truth. No human judgment. Run a method
across multiple seeds per keyhole to get confidence-interval-friendly
numbers that can be tracked over time and compared across methods.

Usage:
    from omega_lock.benchmark import (
        BenchmarkSpec, CalibrationMethod, run_benchmark,
    )
    from omega_lock.keyholes.phantom import PhantomKeyhole

    methods = [
        CalibrationMethod(name="plain_grid",   runner=lambda t, s: run_p1(t, ...)),
        CalibrationMethod(name="fractal_vise", runner=lambda t, s: run_p1_iterative(...)),
    ]
    spec = BenchmarkSpec(
        keyhole_name="PhantomKeyhole",
        keyhole_factory=PhantomKeyhole,
        seeds=[42, 7, 100, 314, 55],
    )
    report = run_benchmark([spec], methods)
    print(report.render_scorecard())
    report.save(Path("benchmark_report.json"))

Ground truth expectations:
    Any keyhole passed in must expose three static methods:
        true_effective_params() -> set[str]
        true_optimum_params()   -> dict[str, Any]
        true_importance_ranking() -> list[str]   # most → least important
    PhantomKeyhole and PhantomKeyholeDeep both implement these.

Metrics (all in [0, 1] except where noted; higher = better unless marked):
    effective_recall      |found ∩ true| / |true|          — want 1.0
    effective_precision   |found ∩ true| / |found|         — want 1.0
    param_L2_error        LOW-LOWER-IS-BETTER              — want 0.0
    fitness_gap_pct       (opt - found) / |opt|            — want ≤ 0 (found >= optimum)
    generalization_gap    |train - test| / |train|         — want small
    sample_efficiency     fitness_found / n_evaluations    — higher = better
    walltime_s            wall clock                        — lower = better
    stress_rank_spearman  ρ(measured_rank, truth_rank)     — want close to 1.0
    status_pass           1 if status=='PASS' else 0       — binary
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


# Keyhole ground-truth interface (duck-typed; no formal Protocol enforcement
# because Python Protocols don't support static method checks cleanly).
KeyholeFactory = Callable[[int], Any]    # (seed) -> target instance
MethodRunner = Callable[[Any, int], dict[str, Any]]
# Runner contract: (target, seed) -> dict with keys:
#   'found_params'     dict — the final best parameters
#   'found_fitness'    float
#   'train_fitness'    float (same as found_fitness if no test_target)
#   'test_fitness'     float | None
#   'unlocked'         list[str] — param names the method chose to search
#   'stress_ranking'   list[str] | None — measured stress order (desc), or None
#   'status'           str — PASS / FAIL:...
#   'n_evaluations'    int — total target.evaluate() calls
#   'walltime_s'       float


@dataclass
class CalibrationMethod:
    """A named calibration method wrapped into the benchmark's runner contract."""
    name: str
    runner: MethodRunner


@dataclass
class BenchmarkSpec:
    """Describes one keyhole family to benchmark across multiple seeds."""
    keyhole_name: str
    keyhole_factory: KeyholeFactory
    seeds: list[int]


@dataclass
class BenchmarkRow:
    """One run result — one (keyhole, method, seed) triple."""
    keyhole: str
    method: str
    seed: int
    effective_recall: float
    effective_precision: float
    param_L2_error: float
    fitness_gap_pct: float
    generalization_gap: float
    sample_efficiency: float
    walltime_s: float
    stress_rank_spearman: float | None
    status_pass: int
    status_raw: str
    n_evaluations: int
    found_fitness: float
    found_params: dict[str, Any]


@dataclass
class MethodSummary:
    """Aggregated metrics for one method across (keyholes × seeds)."""
    method: str
    n_runs: int
    effective_recall_mean: float
    effective_recall_std: float
    effective_precision_mean: float
    param_L2_error_mean: float
    param_L2_error_std: float
    fitness_gap_pct_mean: float
    fitness_gap_pct_std: float
    generalization_gap_mean: float
    sample_efficiency_mean: float
    walltime_s_mean: float
    stress_rank_spearman_mean: float | None
    pass_rate: float


@dataclass
class BenchmarkReport:
    rows: list[BenchmarkRow]

    def scorecard(self) -> list[MethodSummary]:
        by_method: dict[str, list[BenchmarkRow]] = {}
        for r in self.rows:
            by_method.setdefault(r.method, []).append(r)

        out: list[MethodSummary] = []
        for method, rs in by_method.items():
            out.append(MethodSummary(
                method=method,
                n_runs=len(rs),
                effective_recall_mean=_mean([r.effective_recall for r in rs]),
                effective_recall_std=_stdev([r.effective_recall for r in rs]),
                effective_precision_mean=_mean([r.effective_precision for r in rs]),
                param_L2_error_mean=_mean([r.param_L2_error for r in rs]),
                param_L2_error_std=_stdev([r.param_L2_error for r in rs]),
                fitness_gap_pct_mean=_mean([r.fitness_gap_pct for r in rs]),
                fitness_gap_pct_std=_stdev([r.fitness_gap_pct for r in rs]),
                generalization_gap_mean=_mean([r.generalization_gap for r in rs]),
                sample_efficiency_mean=_mean([r.sample_efficiency for r in rs]),
                walltime_s_mean=_mean([r.walltime_s for r in rs]),
                stress_rank_spearman_mean=_optional_mean(
                    [r.stress_rank_spearman for r in rs if r.stress_rank_spearman is not None]
                ),
                pass_rate=_mean([float(r.status_pass) for r in rs]),
            ))
        return out

    def render_scorecard(self) -> str:
        lines: list[str] = []
        lines.append(f"{'method':<22s} {'n':>3s} {'recall':>10s} {'prec':>8s} "
                     f"{'L2err':>10s} {'fit_gap%':>10s} {'gen_gap':>10s} "
                     f"{'eff':>10s} {'wall_s':>8s} {'stress_ρ':>10s} {'pass%':>7s}")
        lines.append("─" * 120)
        for s in self.scorecard():
            rho_str = f"{s.stress_rank_spearman_mean:+.3f}" if s.stress_rank_spearman_mean is not None else "   n/a"
            lines.append(
                f"{s.method:<22s} {s.n_runs:>3d} "
                f"{s.effective_recall_mean:>6.3f}±{s.effective_recall_std:.2f} "
                f"{s.effective_precision_mean:>8.3f} "
                f"{s.param_L2_error_mean:>6.3f}±{s.param_L2_error_std:.2f} "
                f"{s.fitness_gap_pct_mean:>6.1f}±{s.fitness_gap_pct_std:.1f} "
                f"{s.generalization_gap_mean:>10.3f} "
                f"{s.sample_efficiency_mean:>10.4f} "
                f"{s.walltime_s_mean:>8.3f} "
                f"{rho_str:>10s} "
                f"{s.pass_rate*100:>6.1f}%"
            )
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rows": [asdict(r) for r in self.rows],
            "scorecard": [asdict(s) for s in self.scorecard()],
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_fallback))


# ── Metric computations ─────────────────────────────────────────────────

def compute_effective_recall(found: set[str], true: set[str]) -> float:
    if not true:
        return 1.0
    return len(found & true) / len(true)


def compute_effective_precision(found: set[str], true: set[str]) -> float:
    if not found:
        return 0.0
    return len(found & true) / len(found)


def compute_param_L2_error(
    found_params: dict[str, Any],
    true_params: dict[str, Any],
    param_ranges: dict[str, tuple[float, float]],
) -> float:
    """Normalized L2 over effective params only.

    For each effective param, we normalize by its spec range so each axis
    contributes in [0, 1] regardless of native scale. Bool axes contribute
    1.0 if mismatched, 0.0 if matched.
    """
    if not true_params:
        return 0.0
    squared = 0.0
    for name, true_v in true_params.items():
        found_v = found_params.get(name)
        if found_v is None:
            squared += 1.0
            continue
        if isinstance(true_v, bool):
            squared += 0.0 if bool(found_v) == true_v else 1.0
        else:
            lo, hi = param_ranges.get(name, (0.0, 1.0))
            span = max(float(hi) - float(lo), 1e-9)
            squared += ((float(found_v) - float(true_v)) / span) ** 2
    return squared ** 0.5


def compute_fitness_gap_pct(found_fitness: float, true_optimum_fitness: float) -> float:
    denom = max(abs(true_optimum_fitness), 1e-9)
    return 100.0 * (true_optimum_fitness - found_fitness) / denom


def compute_generalization_gap(train_fitness: float, test_fitness: float | None) -> float:
    if test_fitness is None:
        return 0.0
    denom = max(abs(train_fitness), 1e-9)
    return abs(train_fitness - test_fitness) / denom


def compute_spearman(rank_a: list[str], rank_b: list[str]) -> float | None:
    """Spearman rank correlation between two orderings.

    Only uses the intersection of both rankings (ignores params in one
    but not the other). Returns None if intersection has <2 items.
    """
    common = [n for n in rank_a if n in rank_b]
    if len(common) < 2:
        return None
    ra = {n: i for i, n in enumerate(rank_a) if n in common}
    rb = {n: i for i, n in enumerate(rank_b) if n in common}
    n = len(common)
    d_sq_sum = sum((ra[x] - rb[x]) ** 2 for x in common)
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))


# ── Benchmark runner ────────────────────────────────────────────────────

def run_benchmark(
    specs: list[BenchmarkSpec],
    methods: list[CalibrationMethod],
    output_path: Path | None = None,
) -> BenchmarkReport:
    """Run every (spec × method × seed) combination and collect metrics."""
    rows: list[BenchmarkRow] = []

    for spec in specs:
        for seed in spec.seeds:
            target = spec.keyhole_factory(seed)
            true_effective = target.true_effective_params()
            true_params = target.true_optimum_params()
            true_ranking = target.true_importance_ranking()

            # Reference optimum fitness (evaluated once on this seed's instance)
            true_fit_result = target.evaluate(dict(true_params))
            true_optimum_fitness = true_fit_result.fitness

            # Build a per-param range dict for L2 normalization
            param_ranges: dict[str, tuple[float, float]] = {}
            for ps in target.param_space():
                if ps.dtype == "bool":
                    param_ranges[ps.name] = (0.0, 1.0)
                else:
                    param_ranges[ps.name] = (float(ps.low), float(ps.high))

            for method in methods:
                t0 = time.time()
                try:
                    out = method.runner(target, seed)
                except Exception as e:
                    rows.append(BenchmarkRow(
                        keyhole=spec.keyhole_name, method=method.name, seed=seed,
                        effective_recall=0.0, effective_precision=0.0,
                        param_L2_error=float("inf"), fitness_gap_pct=float("inf"),
                        generalization_gap=0.0, sample_efficiency=0.0,
                        walltime_s=time.time() - t0,
                        stress_rank_spearman=None,
                        status_pass=0, status_raw=f"CRASH:{type(e).__name__}",
                        n_evaluations=0, found_fitness=float("-inf"),
                        found_params={},
                    ))
                    continue

                found_params = out["found_params"]
                found_fitness = float(out["found_fitness"])
                unlocked = list(out.get("unlocked", []))
                stress_ranking = out.get("stress_ranking")
                status = out.get("status", "UNKNOWN")
                n_eval = int(out.get("n_evaluations", 0))
                walltime = float(out.get("walltime_s", time.time() - t0))
                train_fit = float(out.get("train_fitness", found_fitness))
                test_fit = out.get("test_fitness")

                row = BenchmarkRow(
                    keyhole=spec.keyhole_name, method=method.name, seed=seed,
                    effective_recall=compute_effective_recall(set(unlocked), true_effective),
                    effective_precision=compute_effective_precision(set(unlocked), true_effective),
                    param_L2_error=compute_param_L2_error(
                        found_params, true_params, param_ranges
                    ),
                    fitness_gap_pct=compute_fitness_gap_pct(found_fitness, true_optimum_fitness),
                    generalization_gap=compute_generalization_gap(
                        train_fit, float(test_fit) if test_fit is not None else None
                    ),
                    sample_efficiency=(
                        found_fitness / n_eval if n_eval > 0 else 0.0
                    ),
                    walltime_s=walltime,
                    stress_rank_spearman=(
                        compute_spearman(stress_ranking, true_ranking)
                        if stress_ranking else None
                    ),
                    status_pass=1 if status == "PASS" else 0,
                    status_raw=status,
                    n_evaluations=n_eval,
                    found_fitness=found_fitness,
                    found_params=dict(found_params),
                )
                rows.append(row)

    report = BenchmarkReport(rows=rows)
    if output_path is not None:
        report.save(output_path)
    return report


# ── internals ───────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _optional_mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _json_fallback(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
