# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""P1 end-to-end orchestrator.

Pipeline:
    1. Baseline evaluate on train_target (neutral defaults)
    2. Stress measurement on train_target → KC-2 gate
    3. Top-K unlock selection
    4. Grid search on train_target
    5. If test_target: walk-forward → KC-4 gate
    6. If hybrid: re-rank top-K with validation_target
    7. KC-1 (time box) + KC-3 (trade counts) at the end
    8. Emit P1Result (JSON-serializable)

Aborts soft on KC fails — pipeline still returns P1Result with status flags
so the caller can emit a RESULT.md even on failure.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omega_lock.fitness import HybridFitness, HybridResult
from omega_lock.grid import GridPoint, GridSearch, ZoomingGridSearch
from omega_lock.random_search import RandomSearch, compare_to_grid
from omega_lock.kill_criteria import (
    KCReport,
    KCThresholds,
    check_kc1,
    check_kc2,
    check_kc3,
    check_kc4,
)
from omega_lock.params import neutral_defaults
from omega_lock.stress import (
    StressOptions,
    StressResult,
    measure_stress,
    select_unlock_top_k,
)
from omega_lock.target import CalibrableTarget, EvalResult
from omega_lock.walk_forward import WalkForward, WalkForwardResult


@dataclass
class P1Config:
    unlock_k: int = 3
    grid_points_per_axis: int = 5
    kc_thresholds: KCThresholds = field(default_factory=KCThresholds)
    walk_forward_top_n: int = 10
    trade_ratio_scale: float = 1.0
    exclude_ofi_in_unlock: bool = False
    stress_verbose: bool = True
    grid_verbose: bool = True
    # Zooming grid: if > 1, run ZoomingGridSearch with `zoom_rounds` passes
    # that geometrically narrow around the running winner. Compute scales
    # as zoom_rounds × grid_points_per_axis^K.
    zoom_rounds: int = 1
    zoom_factor: float = 0.5
    # SC-2 advisory check: after grid search, run the same-budget RandomSearch
    # and compute top-quartile ratio. Reported as ADVISORY (never hard-fails
    # the pipeline) — purpose is to surface "grid coverage is wasted" cases
    # (Bergstra & Bengio 2012) so the user can decide whether to switch
    # methods, NOT to invalidate the run.
    run_sc2_baseline: bool = False
    sc2_random_seed: int = 42
    # Constraint-aware grid_best selection. Has effect only when the
    # train_target is wrapped in AuditingTarget with constraints; for a
    # bare CalibrableTarget all candidates count as feasible regardless
    # of policy (no constraints exist to violate).
    #
    #   "record"          — current behavior: grid_best is the raw
    #                       fitness-max, constraint violations live in
    #                       the audit trail only. Backward-compat default.
    #   "prefer_feasible" — pick fitness-max among constraint-satisfying
    #                       candidates. Falls back to raw max when every
    #                       candidate violates (with constraints_violated
    #                       flag set on the result).
    #   "hard_fail"       — same selection as "prefer_feasible", but if
    #                       no candidate is feasible the pipeline emits
    #                       status="FAIL:CONSTRAINTS" instead of falling
    #                       back. Use for ship-blocking audits.
    constraint_policy: str = "record"


@dataclass
class P1Result:
    status: str                              # "PASS", "FAIL:KC-2", "FAIL:KC-4", ...
    elapsed_seconds: float
    config: dict[str, Any]

    baseline_result: dict[str, Any]          # EvalResult as dict
    stress_results: list[dict[str, Any]]
    top_k: list[str]
    top_k_ex_ofi: list[str]

    grid_results: list[dict[str, Any]]       # list of GridPoint summaries
    grid_best: dict[str, Any] | None = None

    walk_forward: dict[str, Any] | None = None
    hybrid_top: list[dict[str, Any]] | None = None

    # Final single-shot evaluation on a held-out target (never used for any
    # KC decision during the run). Present only when `holdout_target` is
    # passed to run_p1(). Use this as independent generalization evidence —
    # especially in iterative mode where test_target is re-consulted across
    # rounds and KC-4 alone becomes weaker per-round evidence.
    holdout_result: dict[str, Any] | None = None

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


def _hybrid_to_dict(h: HybridResult) -> dict[str, Any]:
    d = {
        "params": dict(h.params),
        "search_fitness": h.search_result.fitness,
        "search_n_trials": h.search_result.n_trials,
        "final_fitness": h.final_fitness,
    }
    if h.validation_result is not None:
        d["validation_fitness"] = h.validation_result.fitness
        d["validation_n_trials"] = h.validation_result.n_trials
    return d


def _set_phase(target: Any, phase: str) -> None:
    """Set audit phase on a target if it supports it (AuditingTarget does).

    Bare CalibrableTargets without set_phase() get a no-op so the
    orchestrator can auto-track phases without forcing every caller to
    wrap in AuditingTarget.
    """
    fn = getattr(target, "set_phase", None)
    if callable(fn):
        fn(phase)


def _is_feasible(point: GridPoint) -> bool:
    """A grid point is feasible if it has no recorded constraint failures.

    Looks at `result.metadata['_constraints_failed']` which AuditingTarget
    populates on each evaluate(). Bare CalibrableTargets have no such
    key, so all their candidates are trivially feasible.
    """
    failed = point.result.metadata.get("_constraints_failed", ())
    return not failed


def _select_grid_best(
    grid_points: list[GridPoint], policy: str
) -> tuple[GridPoint, bool]:
    """Pick grid_best per constraint_policy. Returns (best, constraints_violated).

    `constraints_violated` is True only when policy != "record" and every
    candidate violated at least one constraint (forcing a fallback to the
    raw fitness-max).
    """
    if policy == "record":
        return max(grid_points, key=lambda p: p.result.fitness), False
    feasible = [p for p in grid_points if _is_feasible(p)]
    if feasible:
        return max(feasible, key=lambda p: p.result.fitness), False
    # No feasible candidate — fall back to raw max so the pipeline can
    # still emit a result. Caller decides whether to fail status.
    return max(grid_points, key=lambda p: p.result.fitness), True


def run_p1(
    train_target: CalibrableTarget,
    config: P1Config | None = None,
    test_target: CalibrableTarget | None = None,
    validation_target: CalibrableTarget | None = None,
    output_path: Path | None = None,
    base_params: dict[str, Any] | None = None,
    stress_subset: list[str] | None = None,
    holdout_target: CalibrableTarget | None = None,
) -> P1Result:
    """Run the Omega-Lock P1 pipeline.

    Args:
        train_target: target used for baseline + stress + grid search
        config: P1Config (defaults from SPEC)
        test_target: if provided, runs walk-forward + KC-4
        validation_target: if provided alongside grid, re-ranks top-K via HybridFitness
        output_path: if provided, saves P1Result JSON to this path
        base_params: optional explicit baseline; defaults to neutral_defaults.
            Used by `run_p1_iterative` to pass prior-round locked values.
        stress_subset: optional list of param names to measure stress for.
            Used by `run_p1_iterative` to restrict stress to currently-unlocked
            params (already-locked params have their values fixed).
        holdout_target: optional third target evaluated exactly ONCE on the
            final grid_best params. Never consulted for any KC decision —
            purely independent generalization evidence. Use a fresh seed /
            disjoint data slice that has not been seen by train or test.
    """
    cfg = config or P1Config()
    t_start = time.time()

    # 1. Baseline
    _set_phase(train_target, "baseline")
    specs = train_target.param_space()
    if base_params is None:
        base_params = neutral_defaults(specs)
    baseline = train_target.evaluate(base_params)

    # 2. Stress
    _set_phase(train_target, "stress")
    stress_opts = StressOptions(verbose=cfg.stress_verbose)
    stress = measure_stress(
        target=train_target,
        baseline_params=base_params,
        baseline_result=baseline,
        subset=stress_subset,
        options=stress_opts,
    )
    kc2 = check_kc2([r.raw_stress for r in stress], cfg.kc_thresholds)
    kc_reports = [kc2]

    # Early exit on KC-2
    if kc2.status == "FAIL":
        return _finalize(
            status="FAIL:KC-2",
            t_start=t_start,
            cfg=cfg,
            baseline=baseline,
            stress=stress,
            top_k=[], top_k_ex_ofi=[],
            grid=[],
            grid_best=None,
            wf=None,
            hybrid=None,
            kc_reports=kc_reports,
            output_path=output_path,
        )

    # 3. Unlock top-K
    top_k = select_unlock_top_k(stress, k=cfg.unlock_k, exclude_ofi=cfg.exclude_ofi_in_unlock)
    top_k_ex_ofi = select_unlock_top_k(stress, k=cfg.unlock_k, exclude_ofi=True)

    # 4. Grid search on train (plain or zooming)
    _set_phase(train_target, "grid")
    if cfg.zoom_rounds > 1:
        grid_search = ZoomingGridSearch(
            target=train_target,
            unlocked=top_k,
            grid_points_per_axis=cfg.grid_points_per_axis,
            zoom_rounds=cfg.zoom_rounds,
            zoom_factor=cfg.zoom_factor,
            verbose=cfg.grid_verbose,
        )
    else:
        grid_search = GridSearch(
            target=train_target,
            unlocked=top_k,
            grid_points_per_axis=cfg.grid_points_per_axis,
            verbose=cfg.grid_verbose,
        )
    grid_points_list = grid_search.run(base_params=base_params)
    grid_best, constraints_violated = _select_grid_best(
        grid_points_list, cfg.constraint_policy
    )
    if constraints_violated and cfg.constraint_policy == "hard_fail":
        kc_reports.append(KCReport(
            name="CONSTRAINTS",
            status="FAIL",
            message=(
                "every grid candidate violated at least one declared constraint; "
                "selection fell back to raw fitness-max only because "
                "constraint_policy='hard_fail' is set — pipeline status will be FAIL"
            ),
            detail={"n_candidates": len(grid_points_list)},
        ))

    # 5. Walk-forward (optional)
    wf_result: WalkForwardResult | None = None
    kc4: KCReport | None = None
    if test_target is not None:
        _set_phase(test_target, "walk_forward")
        wf = WalkForward(test_target=test_target, trade_ratio_scale=cfg.trade_ratio_scale)
        wf_result = wf.run(train_grid=grid_points_list, top_n=cfg.walk_forward_top_n)
        kc4 = check_kc4(
            train_fitnesses=wf_result.train_fitnesses,
            test_fitnesses=wf_result.test_fitnesses,
            trade_ratio=wf_result.trade_ratio_scaled,
            thresholds=cfg.kc_thresholds,
        )
        kc_reports.append(kc4)

    # 6. Hybrid re-rank (optional)
    hybrid_results: list[HybridResult] | None = None
    if validation_target is not None:
        _set_phase(validation_target, "hybrid")
        hybrid = HybridFitness(
            search_target=train_target,
            validation_target=validation_target,
            validation_top_k=cfg.walk_forward_top_n,
        )
        top_candidates = sorted(grid_points_list, key=lambda p: p.result.fitness, reverse=True)[: cfg.walk_forward_top_n * 2]
        # use pre-computed search results to avoid redundant evaluations
        hybrid_results = []
        for gp in top_candidates:
            hr = HybridResult(params=dict(gp.params), search_result=gp.result)
            hybrid_results.append(hr)
        # validate top N
        for hr in sorted(hybrid_results, key=lambda h: h.search_result.fitness, reverse=True)[: cfg.walk_forward_top_n]:
            hr.validation_result = validation_target.evaluate(hr.params)
        hybrid_results.sort(key=lambda h: h.final_fitness, reverse=True)

    # 7. SC-2 advisory (optional — never hard-fails)
    if cfg.run_sc2_baseline and grid_points_list:
        rs = RandomSearch(
            target=train_target,
            unlocked=top_k,
            n_samples=len(grid_points_list),
            seed=cfg.sc2_random_seed,
            verbose=False,
        )
        rand_pts = rs.run(base_params=base_params)
        sc2 = compare_to_grid(grid_points_list, rand_pts)
        sc2_pass = bool(sc2.get("sc2_pass", False))
        kc_reports.append(KCReport(
            name="SC-2",
            status="ADVISORY",
            message=(
                f"grid_top_q={sc2['grid_top_quartile']:.3f} "
                f"random_top_q={sc2['random_top_quartile']:.3f} "
                f"ratio={sc2['ratio']:.3f} "
                f"({'grid beats random >=1.5x' if sc2_pass else 'grid does NOT beat random >=1.5x — consider switching search method'})"
            ),
            detail=sc2,
        ))

    # 8. KC-1 (time box) + KC-3 (trade counts)
    elapsed = time.time() - t_start
    kc1 = check_kc1(elapsed, cfg.kc_thresholds)
    trade_counts = {"baseline": baseline.n_trials, "train_best": grid_best.result.n_trials}
    if wf_result is not None:
        trade_counts["test_best"] = wf_result.test_best_trades
    kc3 = check_kc3(trade_counts, cfg.kc_thresholds)
    kc_reports.append(kc1)
    kc_reports.append(kc3)

    # Overall status
    if any(r.status == "FAIL" for r in kc_reports):
        failed = [r.name for r in kc_reports if r.status == "FAIL"]
        status = "FAIL:" + ",".join(failed)
    else:
        status = "PASS"

    # 8. Holdout (single-shot, never consulted for KC decisions above)
    holdout_dict: dict[str, Any] | None = None
    if holdout_target is not None:
        _set_phase(holdout_target, "holdout")
        ho = holdout_target.evaluate(grid_best.params)
        train_fit = grid_best.result.fitness
        test_fit = wf_result.test_fitnesses[0] if wf_result and wf_result.test_fitnesses else None
        holdout_dict = {
            "fitness": ho.fitness,
            "n_trials": ho.n_trials,
            "params": dict(grid_best.params),
            "fitness_vs_train": (ho.fitness - train_fit),
            "fitness_vs_test": (ho.fitness - test_fit) if test_fit is not None else None,
            "trade_ratio_vs_train": (
                ho.n_trials / grid_best.result.n_trials
                if grid_best.result.n_trials > 0 else None
            ),
        }

    return _finalize(
        status=status,
        t_start=t_start,
        cfg=cfg,
        baseline=baseline,
        stress=stress,
        top_k=top_k,
        top_k_ex_ofi=top_k_ex_ofi,
        grid=grid_points_list,
        grid_best=grid_best,
        wf=wf_result,
        hybrid=hybrid_results,
        kc_reports=kc_reports,
        output_path=output_path,
        holdout=holdout_dict,
    )


def _finalize(
    *,
    status: str,
    t_start: float,
    cfg: P1Config,
    baseline: EvalResult,
    stress: list[StressResult],
    top_k: list[str],
    top_k_ex_ofi: list[str],
    grid: list[GridPoint],
    grid_best: GridPoint | None,
    wf: WalkForwardResult | None,
    hybrid: list[HybridResult] | None,
    kc_reports: list[KCReport],
    output_path: Path | None,
    holdout: dict[str, Any] | None = None,
) -> P1Result:
    elapsed = time.time() - t_start
    result = P1Result(
        status=status,
        elapsed_seconds=elapsed,
        config={
            "unlock_k": cfg.unlock_k,
            "grid_points_per_axis": cfg.grid_points_per_axis,
            "walk_forward_top_n": cfg.walk_forward_top_n,
            "trade_ratio_scale": cfg.trade_ratio_scale,
            "exclude_ofi_in_unlock": cfg.exclude_ofi_in_unlock,
            "kc_thresholds": asdict(cfg.kc_thresholds),
        },
        baseline_result=_eval_to_dict(baseline),
        stress_results=[s.to_dict() for s in stress],
        top_k=top_k,
        top_k_ex_ofi=top_k_ex_ofi,
        grid_results=[gp.to_summary() for gp in grid],
        grid_best=grid_best.to_summary() if grid_best else None,
        walk_forward=wf.to_dict() if wf else None,
        hybrid_top=[_hybrid_to_dict(h) for h in hybrid] if hybrid else None,
        holdout_result=holdout,
        kc_reports=[
            {"name": r.name, "status": r.status, "message": r.message, "detail": r.detail}
            for r in kc_reports
        ],
    )
    if output_path is not None:
        result.save(output_path)
    return result


# ── Multi-round (coordinate descent) orchestrator ──────────────────────────

@dataclass
class IterativeConfig:
    """Per-round coordinate-descent settings.

    The loop runs up to `rounds` times. In round r:
      - baseline = values locked in rounds 0..r-1 (neutrals for the rest)
      - stress is measured only on currently-unlocked params
      - top-K of the unlocked set is grid-searched
      - winners are locked in for all subsequent rounds

    KC checks run inside each round with the same thresholds — multi-round
    does NOT relax any kill criterion. If `stop_on_kc_fail=True`, the first
    round that fails any KC halts the loop.

    Stopping conditions:
      - hit `rounds` cap
      - fewer than `per_round_unlock_k` params still unlocked
      - any KC fails in the latest round (if stop_on_kc_fail)
      - improvement over previous round below `min_improvement` (diminishing returns)
    """
    rounds: int = 3
    per_round_unlock_k: int = 3
    grid_points_per_axis: int = 5
    walk_forward_top_n: int = 10
    trade_ratio_scale: float = 1.0
    kc_thresholds: KCThresholds = field(default_factory=KCThresholds)
    stop_on_kc_fail: bool = True
    min_improvement: float = 0.0      # round fitness must beat prev by this much
    exclude_ofi_in_unlock: bool = False
    stress_verbose: bool = False
    grid_verbose: bool = False
    # Per-round zooming (fractal-vise refinement inside each outer round)
    zoom_rounds: int = 1
    zoom_factor: float = 0.5
    # Per-round SC-2 advisory — identical semantics to P1Config.run_sc2_baseline
    run_sc2_baseline: bool = False
    sc2_random_seed: int = 42


@dataclass
class IterativeResult:
    """Aggregated result of a coordinate-descent run."""
    rounds: list[P1Result]
    final_baseline: dict[str, Any]           # all values after the last locking step
    locked_in_order: list[list[str]]         # names locked per round
    fitness_trajectory: list[float]          # baseline-fitness at start of each round
    round_best_fitness: list[float]          # grid_best fitness per round
    total_elapsed_seconds: float
    final_status: str                        # "PASS" only if every round passed
    stop_reason: str                         # "max_rounds" | "kc_fail" | "too_few_params" | "no_improvement"
    # Holdout: single-shot evaluation of final_baseline on a target that was
    # NEVER consulted during any round (including KC-4 walk-forward). This
    # is the honest generalization check — per-round KC-4 gets weaker as
    # rounds accumulate because test_target is reused for lock-in decisions.
    holdout_result: dict[str, Any] | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "final_status": self.final_status,
            "stop_reason": self.stop_reason,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "locked_in_order": self.locked_in_order,
            "fitness_trajectory": self.fitness_trajectory,
            "round_best_fitness": self.round_best_fitness,
            "final_baseline": self.final_baseline,
            "rounds": [asdict(r) for r in self.rounds],
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_fallback))


def run_p1_iterative(
    train_target: CalibrableTarget,
    config: IterativeConfig | None = None,
    test_target: CalibrableTarget | None = None,
    validation_target: CalibrableTarget | None = None,
    output_path: Path | None = None,
    holdout_target: CalibrableTarget | None = None,
) -> IterativeResult:
    """Multi-round coordinate-descent version of run_p1.

    After each round's grid search, the winning (alpha, window, ...) values
    are locked in and used as the baseline for the next round. Stress is
    re-measured on the remaining unlocked params only — so a param that
    had low stress at neutrals can surface as load-bearing near the
    emerging optimum.

    Each round runs full KC-1..4 checks at the current-round thresholds.
    This is coordinate descent with per-round certification, NOT "retry
    until pass" — relaxing thresholds is explicitly forbidden (SPEC §3).

    Args:
        holdout_target: optional third target evaluated ONCE on the final
            locked configuration (after all rounds halt). Never used in any
            round's KC decisions. Treat this as the honest out-of-sample
            check — per-round KC-4 on `test_target` becomes weaker evidence
            as rounds accumulate because the test set is consulted repeatedly
            for lock-in choices.
    """
    cfg = config or IterativeConfig()
    specs_list = train_target.param_space()
    all_names = [s.name for s in specs_list]

    base = neutral_defaults(specs_list)
    already_locked: list[str] = []
    rounds: list[P1Result] = []
    locked_in_order: list[list[str]] = []
    fitness_trajectory: list[float] = []
    round_best_fitness: list[float] = []
    stop_reason = "max_rounds"

    t_start = time.time()
    for round_idx in range(cfg.rounds):
        remaining = [n for n in all_names if n not in already_locked]
        if len(remaining) < cfg.per_round_unlock_k:
            stop_reason = "too_few_params"
            break

        per_round_p1_cfg = P1Config(
            unlock_k=cfg.per_round_unlock_k,
            grid_points_per_axis=cfg.grid_points_per_axis,
            walk_forward_top_n=cfg.walk_forward_top_n,
            trade_ratio_scale=cfg.trade_ratio_scale,
            kc_thresholds=cfg.kc_thresholds,
            exclude_ofi_in_unlock=cfg.exclude_ofi_in_unlock,
            stress_verbose=cfg.stress_verbose,
            grid_verbose=cfg.grid_verbose,
            zoom_rounds=cfg.zoom_rounds,
            zoom_factor=cfg.zoom_factor,
            run_sc2_baseline=cfg.run_sc2_baseline,
            sc2_random_seed=cfg.sc2_random_seed,
        )
        r = run_p1(
            train_target=train_target,
            config=per_round_p1_cfg,
            test_target=test_target,
            validation_target=validation_target,
            base_params=dict(base),
            stress_subset=remaining,
        )
        rounds.append(r)
        fitness_trajectory.append(r.baseline_result["fitness"])
        round_best_fitness.append(r.grid_best["fitness"] if r.grid_best else float("-inf"))

        if r.status != "PASS" and cfg.stop_on_kc_fail:
            stop_reason = "kc_fail"
            break
        if r.grid_best is None:
            stop_reason = "no_grid_best"
            break

        improvement = r.grid_best["fitness"] - r.baseline_result["fitness"]
        if improvement < cfg.min_improvement:
            stop_reason = "no_improvement"
            # still lock this round's output before breaking
        # Lock winners
        this_locked: list[str] = []
        for name, val in r.grid_best["unlocked"].items():
            base[name] = val
            already_locked.append(name)
            this_locked.append(name)
        locked_in_order.append(this_locked)
        if improvement < cfg.min_improvement:
            break

    # Holdout evaluation — single shot on final_baseline, outside the loop.
    # The holdout target is never consulted for any lock-in decision; its
    # fitness is an honest out-of-sample number, reported as a companion to
    # per-round KC-4 (which becomes weaker per-round evidence as test is
    # reused across rounds).
    holdout_dict: dict[str, Any] | None = None
    if holdout_target is not None and rounds:
        ho = holdout_target.evaluate(dict(base))
        # Use the latest round's grid-best fitness as the "train reference".
        last = rounds[-1]
        train_ref = last.grid_best["fitness"] if last.grid_best else None
        test_ref = (
            last.walk_forward["test_fitnesses"][0]
            if last.walk_forward and last.walk_forward.get("test_fitnesses")
            else None
        )
        holdout_dict = {
            "fitness": ho.fitness,
            "n_trials": ho.n_trials,
            "params": dict(base),
            "fitness_vs_last_round_train": (
                ho.fitness - train_ref if train_ref is not None else None
            ),
            "fitness_vs_last_round_test": (
                ho.fitness - test_ref if test_ref is not None else None
            ),
        }

    elapsed = time.time() - t_start
    any_fail = any(r.status != "PASS" for r in rounds)
    final_status = "PASS" if rounds and not any_fail else (
        f"FAIL:round{len(rounds)}" if rounds else "FAIL:no_rounds"
    )

    result = IterativeResult(
        rounds=rounds,
        final_baseline=base,
        locked_in_order=locked_in_order,
        fitness_trajectory=fitness_trajectory,
        round_best_fitness=round_best_fitness,
        total_elapsed_seconds=elapsed,
        final_status=final_status,
        stop_reason=stop_reason,
        holdout_result=holdout_dict,
    )
    if output_path is not None:
        result.save(output_path)
    return result
