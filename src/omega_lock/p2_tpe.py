# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""P2 end-to-end orchestrator ??Optuna TPE variant of P1.

Pipeline (same gates as run_p1):
    1. Baseline evaluate on train_target (neutral defaults)
    2. Stress measurement on train_target -> KC-2 gate
    3. Top-K unlock selection
    4. Optuna TPE sampling over K-dim unlocked subspace
    5. If test_target: walk-forward on top-N by train fitness -> KC-4 gate
    6. KC-1 (time box) + KC-3 (trade counts) at the end
    7. Emit P2Result (JSON-serializable)

Motivation: grid search costs grow as g^K (e.g. 5^3 = 125) and cannot refine
below the cell width. TPE fits a posterior over promising regions and can
sample at arbitrary continuous precision within the same trial budget, at
the cost of stochasticity (seed-controlled here) and an optional dep on
optuna. The rest of the gates (KC-1..4) are identical to run_p1 ??this is a
drop-in search-method swap, not a relaxation of the kill criteria.

Optuna is declared as an optional dep (pyproject [project.optional-dependencies].p2).
At import time we flip a module-level flag; run_p2_tpe raises a clear ImportError
with install hint if optuna is not available.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omega_lock.grid import GridPoint
from omega_lock.kill_criteria import (
    KCReport,
    KCThresholds,
    check_kc1,
    check_kc2,
    check_kc3,
    check_kc4,
)
from omega_lock.params import clip, neutral_defaults
from omega_lock.stress import (
    StressOptions,
    StressResult,
    measure_stress,
    select_unlock_top_k,
)
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec
from omega_lock.walk_forward import WalkForward, WalkForwardResult


try:
    import optuna
    # Silence Optuna's per-trial INFO logs and the ExperimentalWarning
    # emitted by TPESampler(multivariate=True) ??stable in practice for years.
    import warnings as _warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _warnings.filterwarnings(
        "ignore",
        category=optuna.exceptions.ExperimentalWarning,
    )
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False


_OPTUNA_INSTALL_HINT = (
    "Optuna is required for run_p2_tpe but is not installed. "
    "Install it via `pip install \"omega-lock[p2]\"` or `pip install \"optuna>=3.0\"`."
)


@dataclass
class P2Config:
    """Configuration for run_p2_tpe.

    n_trials defaults to 125 to match a 5^3 grid's evaluation budget so P2
    and P1 can be compared apples-to-apples at the same cost. seed pins the
    TPE sampler's RNG for reproducible trial histories.
    """
    unlock_k: int = 3
    n_trials: int = 125
    seed: int = 42
    walk_forward_top_n: int = 10
    trade_ratio_scale: float = 1.0
    kc_thresholds: KCThresholds = field(default_factory=KCThresholds)
    exclude_ofi_in_unlock: bool = False
    stress_verbose: bool = False
    trial_verbose: bool = False
    # Multivariate TPE models correlations between unlocked params; much
    # stronger than the default univariate TPE on curved optima (e.g. Rosenbrock).
    multivariate: bool = True


@dataclass
class P2Result:
    status: str                                        # "PASS" or "FAIL:KC-..."
    elapsed_seconds: float
    config: dict[str, Any]

    baseline_result: dict[str, Any]                    # EvalResult as dict
    stress_results: list[dict[str, Any]]
    top_k: list[str]
    top_k_ex_ofi: list[str]

    trials: list[dict[str, Any]]                       # per-trial summaries
    tpe_best: dict[str, Any] | None = None             # {unlocked, fitness, n_trials}

    walk_forward: dict[str, Any] | None = None

    kc_reports: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=_json_fallback))


def _json_fallback(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _eval_to_dict(r: EvalResult) -> dict[str, Any]:
    return {
        "fitness": r.fitness,
        "n_trials": r.n_trials,
        "metadata": dict(r.metadata),
    }


def _suggest(trial: "optuna.trial.Trial", spec: ParamSpec) -> Any:
    """Map a ParamSpec to the corresponding Optuna suggest_* call.

    Float/int bounds are inclusive on both sides (matches ParamSpec + Optuna
    conventions). Bool is expressed as a two-value categorical ??this keeps
    the TPE model on a discrete axis rather than inferring a spurious
    continuous gradient.
    """
    if spec.dtype == "float":
        return trial.suggest_float(spec.name, float(spec.low), float(spec.high))
    if spec.dtype == "int":
        return trial.suggest_int(spec.name, int(spec.low), int(spec.high))
    if spec.dtype == "bool":
        return trial.suggest_categorical(spec.name, [False, True])
    raise ValueError(f"unsupported dtype '{spec.dtype}' for param '{spec.name}'")


def run_p2_tpe(
    train_target: CalibrableTarget,
    config: P2Config | None = None,
    test_target: CalibrableTarget | None = None,
    output_path: Path | None = None,
    base_params: dict[str, Any] | None = None,
    stress_subset: list[str] | None = None,
) -> P2Result:
    """Run the Omega-Lock P2 pipeline (TPE replaces grid search).

    Args:
        train_target: target used for baseline + stress + TPE search
        config: P2Config (defaults from SPEC; n_trials=125 matches 5^3 grid)
        test_target: if provided, runs walk-forward + KC-4
        output_path: if provided, saves P2Result JSON to this path
        base_params: optional explicit baseline; defaults to neutral_defaults.
            Mirrors run_p1 so P2 can be dropped into coordinate-descent loops.
        stress_subset: optional list of param names to measure stress for.

    Raises:
        ImportError: if optuna is not installed (optional dep `p2`).
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(_OPTUNA_INSTALL_HINT)

    cfg = config or P2Config()
    t_start = time.time()

    # 1. Baseline
    specs_list = train_target.param_space()
    specs_by_name = {s.name: s for s in specs_list}
    if base_params is None:
        base_params = neutral_defaults(specs_list)
    baseline = train_target.evaluate(base_params)

    # 2. Stress
    stress_opts = StressOptions(verbose=cfg.stress_verbose)
    stress = measure_stress(
        target=train_target,
        baseline_params=base_params,
        baseline_result=baseline,
        subset=stress_subset,
        options=stress_opts,
    )
    kc2 = check_kc2([r.raw_stress for r in stress], cfg.kc_thresholds)
    kc_reports: list[KCReport] = [kc2]

    if kc2.status == "FAIL":
        return _finalize(
            status="FAIL:KC-2",
            t_start=t_start,
            cfg=cfg,
            baseline=baseline,
            stress=stress,
            top_k=[],
            top_k_ex_ofi=[],
            trial_records=[],
            tpe_best=None,
            wf=None,
            kc_reports=kc_reports,
            output_path=output_path,
        )

    # 3. Unlock top-K
    top_k = select_unlock_top_k(stress, k=cfg.unlock_k, exclude_ofi=cfg.exclude_ofi_in_unlock)
    top_k_ex_ofi = select_unlock_top_k(stress, k=cfg.unlock_k, exclude_ofi=True)

    # 4. TPE search over unlocked subspace
    trial_records, trial_grid_points = _run_tpe(
        train_target=train_target,
        base_params=base_params,
        unlocked=top_k,
        specs_by_name=specs_by_name,
        cfg=cfg,
    )
    tpe_best_gp = (
        max(trial_grid_points, key=lambda gp: gp.result.fitness)
        if trial_grid_points else None
    )
    tpe_best_summary = tpe_best_gp.to_summary() if tpe_best_gp else None

    # 5. Walk-forward (optional)
    wf_result: WalkForwardResult | None = None
    kc4: KCReport | None = None
    if test_target is not None and trial_grid_points:
        wf = WalkForward(test_target=test_target, trade_ratio_scale=cfg.trade_ratio_scale)
        wf_result = wf.run(train_grid=trial_grid_points, top_n=cfg.walk_forward_top_n)
        kc4 = check_kc4(
            train_fitnesses=wf_result.train_fitnesses,
            test_fitnesses=wf_result.test_fitnesses,
            trade_ratio=wf_result.trade_ratio_scaled,
            thresholds=cfg.kc_thresholds,
        )
        kc_reports.append(kc4)

    # 6. KC-1 (time box) + KC-3 (trade counts)
    elapsed = time.time() - t_start
    kc1 = check_kc1(elapsed, cfg.kc_thresholds)
    trade_counts: dict[str, int] = {"baseline": baseline.n_trials}
    if tpe_best_gp is not None:
        trade_counts["train_best"] = tpe_best_gp.result.n_trials
    if wf_result is not None:
        trade_counts["test_best"] = wf_result.test_best_trades
    kc3 = check_kc3(trade_counts, cfg.kc_thresholds)
    kc_reports.append(kc1)
    kc_reports.append(kc3)

    if any(r.status == "FAIL" for r in kc_reports):
        failed = [r.name for r in kc_reports if r.status == "FAIL"]
        status = "FAIL:" + ",".join(failed)
    else:
        status = "PASS"

    return _finalize(
        status=status,
        t_start=t_start,
        cfg=cfg,
        baseline=baseline,
        stress=stress,
        top_k=top_k,
        top_k_ex_ofi=top_k_ex_ofi,
        trial_records=trial_records,
        tpe_best=tpe_best_summary,
        wf=wf_result,
        kc_reports=kc_reports,
        output_path=output_path,
    )


def _run_tpe(
    train_target: CalibrableTarget,
    base_params: dict[str, Any],
    unlocked: list[str],
    specs_by_name: dict[str, ParamSpec],
    cfg: P2Config,
) -> tuple[list[dict[str, Any]], list[GridPoint]]:
    """Drive an Optuna study over the unlocked subspace.

    Returns:
        (trial_records, trial_grid_points)

        trial_records: JSON-safe per-trial summaries.
        trial_grid_points: GridPoint objects, one per trial, reusable by
            WalkForward (which expects list[GridPoint], sorts by fitness).
    """
    assert _OPTUNA_AVAILABLE  # caller guarantees
    for name in unlocked:
        if name not in specs_by_name:
            raise KeyError(f"unlocked param '{name}' not in target.param_space()")

    trial_records: list[dict[str, Any]] = []
    trial_grid_points: list[GridPoint] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        unlocked_vals: dict[str, Any] = {}
        params = dict(base_params)
        for name in unlocked:
            spec = specs_by_name[name]
            raw = _suggest(trial, spec)
            # defensive clip (float out-of-range should not happen, but cheap insurance)
            v = clip(spec, raw)
            unlocked_vals[name] = v
            params[name] = v
        t_eval0 = time.time()
        r = train_target.evaluate(params)
        wall = time.time() - t_eval0

        idx = trial.number
        gp = GridPoint(
            idx=idx,
            unlocked=unlocked_vals,
            params=params,
            result=r,
            wall_seconds=wall,
        )
        trial_grid_points.append(gp)
        trial_records.append({
            "trial_idx": idx,
            "params_unlocked": dict(unlocked_vals),
            "fitness": r.fitness,
            "n_trials": r.n_trials,
            "wall_s": wall,
        })
        if cfg.trial_verbose:
            print(
                f"  trial[{idx+1:4d}/{cfg.n_trials}] "
                f"fitness={r.fitness:.4f} {unlocked_vals}"
            )
        return float(r.fitness)

    sampler = optuna.samplers.TPESampler(
        seed=cfg.seed,
        multivariate=cfg.multivariate,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        show_progress_bar=False,
    )
    return trial_records, trial_grid_points


def _finalize(
    *,
    status: str,
    t_start: float,
    cfg: P2Config,
    baseline: EvalResult,
    stress: list[StressResult],
    top_k: list[str],
    top_k_ex_ofi: list[str],
    trial_records: list[dict[str, Any]],
    tpe_best: dict[str, Any] | None,
    wf: WalkForwardResult | None,
    kc_reports: list[KCReport],
    output_path: Path | None,
) -> P2Result:
    elapsed = time.time() - t_start
    result = P2Result(
        status=status,
        elapsed_seconds=elapsed,
        config={
            "unlock_k": cfg.unlock_k,
            "n_trials": cfg.n_trials,
            "seed": cfg.seed,
            "walk_forward_top_n": cfg.walk_forward_top_n,
            "trade_ratio_scale": cfg.trade_ratio_scale,
            "exclude_ofi_in_unlock": cfg.exclude_ofi_in_unlock,
            "multivariate": cfg.multivariate,
            "kc_thresholds": asdict(cfg.kc_thresholds),
        },
        baseline_result=_eval_to_dict(baseline),
        stress_results=[s.to_dict() for s in stress],
        top_k=top_k,
        top_k_ex_ofi=top_k_ex_ofi,
        trials=trial_records,
        tpe_best=tpe_best,
        walk_forward=wf.to_dict() if wf else None,
        kc_reports=[
            {"name": r.name, "status": r.status, "message": r.message, "detail": r.detail}
            for r in kc_reports
        ],
    )
    if output_path is not None:
        result.save(output_path)
    return result
