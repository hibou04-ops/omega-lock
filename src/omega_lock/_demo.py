# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Packaged walk-forward gate case study (backs ``omega-lock demo``).

This module is the single source of the deterministic overfitting case
study that ``examples/walkforward_gate_demo.py`` introduced. It lives
inside the import package (not under ``examples/``) so the installed
console command ``omega-lock demo`` can run it from a wheel, where the
``examples/`` tree does not ship. The example file re-exports everything
from here, so ``python examples/walkforward_gate_demo.py`` and
``omega-lock demo`` print byte-identical narratives.

The case study:

    1. A synthetic target has a real, transferable optimum (fitness ~5.0
       inside a "rated" operating envelope) and a fragile region where a
       slice-dependent noise term can spike far higher (up to ~6 above
       the envelope limit). The noise re-draws on every data slice, so a
       noise spike found on the train slice does NOT transfer.
    2. Naive selection — "take the highest train score" (`best_any`) —
       picks a lucky-noise point in the fragile region.
    3. Omega-Lock's walk-forward gate (KC-4) re-evaluates the train-best
       top-N on a *test* slice. The train ranking does not survive
       (Pearson collapses), so the run is stamped FAIL:KC-4 instead of
       shipping the lucky candidate.
    4. Feasible-best selection (a declared hard constraint: stay inside
       the rated envelope) picks a candidate that holds up on a holdout
       slice that no selection step ever consulted.

Everything is seeded and hash-based: no RNG state, no network, no API
keys. Repeated runs print identical numbers. Runtime is well under 60s.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from omega_lock.audit import AuditingTarget, Constraint, make_report
from omega_lock.grid import GridPoint
from omega_lock.kill_criteria import KCThresholds, check_kc4
from omega_lock.orchestrator import P1Config, run_p1
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.walk_forward import WalkForward


# ── The synthetic system ────────────────────────────────────────────────────
#
# fitness(gain, bias, offset; slice) = signal(gain, bias, offset)
#                                      + noise_amplitude(gain) * u(gain, bias, slice)
#
#   signal     : smooth peak of height 5.0 at (gain=3, bias=0); tiny
#                dependence on `offset` (a decoy axis the stress phase
#                should lock).
#   noise      : deterministic hash noise in [0, amplitude). Inside the
#                rated envelope (gain <= 6) the amplitude is small (0.3).
#                Beyond it the system is fragile: amplitude 6.0 — large
#                enough that some lucky (gain, bias) cell on the train
#                slice will out-score the true optimum.
#   slice seed : changing the slice re-draws every noise value. This is
#                the synthetic stand-in for "a different data window".

SIGNAL_PEAK = 5.0
RATED_GAIN_MAX = 6.0
STABLE_NOISE_AMPLITUDE = 0.3
FRAGILE_NOISE_AMPLITUDE = 6.0

TRAIN_SEED = 2026    # slice used for search
TEST_SEED = 2027     # slice used by the walk-forward gate (KC-4)
HOLDOUT_SEED = 2028  # slice never consulted by any selection step


def _unit_noise(gain: float, bias: float, slice_seed: int) -> float:
    """Deterministic pseudo-noise in [0, 1) — stable across platforms/runs."""
    key = f"{gain:.6f}|{bias:.6f}|{slice_seed}".encode("utf-8")
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


class NoisySliceTarget:
    """CalibrableTarget over one data slice of the synthetic system."""

    def __init__(self, slice_seed: int) -> None:
        self.slice_seed = slice_seed

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="gain", dtype="float", low=0.0, high=10.0, neutral=2.0),
            ParamSpec(name="bias", dtype="float", low=-3.0, high=3.0, neutral=1.5),
            ParamSpec(name="offset", dtype="float", low=-1.0, high=1.0, neutral=0.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        gain = float(params["gain"])
        bias = float(params["bias"])
        offset = float(params["offset"])
        signal = (
            SIGNAL_PEAK
            * math.exp(-((gain - 3.0) ** 2) / 4.0 - (bias ** 2) / 2.0)
            + 0.01 * offset
        )
        amplitude = (
            STABLE_NOISE_AMPLITUDE if gain <= RATED_GAIN_MAX
            else FRAGILE_NOISE_AMPLITUDE
        )
        fitness = signal + amplitude * _unit_noise(gain, bias, self.slice_seed)
        return EvalResult(
            fitness=fitness,
            sample_count=1,
            metadata={"signal": signal, "noise_amplitude": amplitude},
        )


# The declared hard constraint: stay inside the rated operating envelope.
# This is domain knowledge expressed up front — NOT a post-hoc filter
# invented after seeing the holdout numbers.
RATED_ENVELOPE = Constraint(
    "gain_within_rated_envelope",
    lambda params, result: float(params["gain"]) <= RATED_GAIN_MAX,
    f"gain must stay <= {RATED_GAIN_MAX} (system is fragile beyond it)",
)


def _full_params(unlocked: dict[str, Any]) -> dict[str, Any]:
    """Merge grid-unlocked values over the neutral baseline."""
    params = {"gain": 2.0, "bias": 1.5, "offset": 0.0}
    params.update(unlocked)
    return params


def _gap_pct(train: float, holdout: float) -> float:
    return 100.0 * (holdout - train) / abs(train) if train else 0.0


def run_demo() -> int:
    """Run the full case-study narrative; returns a process exit code."""
    train = NoisySliceTarget(TRAIN_SEED)
    test = NoisySliceTarget(TEST_SEED)
    holdout = NoisySliceTarget(HOLDOUT_SEED)

    # Wrap the train target so every evaluation lands in an audit trail
    # with the declared constraint recorded per candidate.
    audited_train = AuditingTarget(train, constraints=[RATED_ENVELOPE], target_role="train")

    print("=" * 72)
    print(" Walk-forward gate demo: the best score is not deployable")
    print("=" * 72)
    print()
    print(f" true optimum   : fitness ~{SIGNAL_PEAK:.1f} at (gain=3.0, bias=0.0), transferable")
    print(f" fragile region : gain > {RATED_GAIN_MAX:.0f} -> slice noise up to "
          f"+{FRAGILE_NOISE_AMPLITUDE:.0f}, NOT transferable")
    print(" slices         : train=search, test=walk-forward gate, holdout=untouched")
    print()

    # ── One P1 pipeline run: stress -> top-K unlock -> grid -> KC gates ──
    cfg = P1Config(
        unlock_k=2,                  # stress should pick gain+bias, lock the decoy
        grid_points_per_axis=21,     # 21x21 = 441 candidates
        walk_forward_top_n=10,
        kc_thresholds=KCThresholds.pure_objective(),  # no "trade count" concept here
        constraint_policy="prefer_feasible",
        stress_verbose=False,
        grid_verbose=False,
    )
    result = run_p1(
        train_target=audited_train,
        config=cfg,
        test_target=test,
        holdout_target=holdout,
    )

    assert result.grid_best is not None, "pipeline regression: no grid_best"
    assert set(result.top_k) == {"gain", "bias"}, (
        f"stress phase should unlock gain+bias and lock the decoy, got {result.top_k}"
    )

    # ── Naive selection: best_any (highest train fitness, no gates) ──────
    best_any = max(result.grid_results, key=lambda g: g["fitness"])
    best_any_params = _full_params(best_any["unlocked"])
    best_any_train = best_any["fitness"]
    best_any_test = test.evaluate(best_any_params).fitness
    best_any_holdout = holdout.evaluate(best_any_params).fitness

    print("[1] Naive selection (best_any = argmax train fitness)")
    print(f"    picked params : {best_any_params}")
    print(f"    train fitness : {best_any_train:.3f}  <- looks like a winner")
    print(f"    test fitness  : {best_any_test:.3f}")
    print(f"    holdout       : {best_any_holdout:.3f}  "
          f"({_gap_pct(best_any_train, best_any_holdout):+.1f}% vs train)")
    assert best_any_params["gain"] > RATED_GAIN_MAX, (
        "expected the naive winner to be a lucky-noise point in the fragile region"
    )
    print()

    # ── The walk-forward gate verdict ─────────────────────────────────────
    assert result.walk_forward is not None
    wf_pearson = result.walk_forward["pearson"]
    print("[2] Omega-Lock walk-forward gate (KC-4) on the train-best top-10")
    print(f"    Pearson(train, test) = {wf_pearson:.3f}  "
          f"(threshold {cfg.kc_thresholds.pearson_min})")
    print(f"    pipeline status      = {result.status}")
    for kc in result.kc_reports:
        print(f"      {kc['name']}: {kc['status']:8s} {kc['message']}")
    assert "KC-4" in result.status, (
        f"expected the walk-forward gate to fail this run, got {result.status}"
    )
    assert wf_pearson < cfg.kc_thresholds.pearson_min, (
        "the lucky-noise ranking should not correlate train->test"
    )
    print("    -> the gate refuses to certify the noise-driven ranking.")
    print()

    # ── Feasible-best selection (constraint declared up front) ───────────
    feasible_best = result.grid_best  # constraint_policy="prefer_feasible"
    feasible_params = _full_params(feasible_best["unlocked"])
    feasible_train = feasible_best["fitness"]
    assert result.holdout_result is not None
    feasible_holdout = result.holdout_result["fitness"]

    print("[3] Gated selection (best_feasible under the declared envelope constraint)")
    print(f"    picked params : {feasible_params}")
    print(f"    train fitness : {feasible_train:.3f}")
    print(f"    holdout       : {feasible_holdout:.3f}  "
          f"({_gap_pct(feasible_train, feasible_holdout):+.1f}% vs train)")
    assert feasible_params["gain"] <= RATED_GAIN_MAX
    print()

    # ── Walk-forward over the feasible candidates only ────────────────────
    # Rebuild GridPoints from the audit trail (grid phase) and gate the
    # feasible subset: the ranking that respects the envelope DOES transfer.
    report = make_report(audited_train, method="run_p1", seed=None)
    grid_runs = report.by_phase("grid")
    feasible_points = [
        GridPoint(
            idx=i,
            unlocked={k: run.params[k] for k in result.top_k},
            params=dict(run.params),
            result=EvalResult(fitness=run.fitness, sample_count=run.n_trials),
        )
        for i, run in enumerate(grid_runs)
        if run.is_feasible
    ]
    wf_feasible = WalkForward(test_target=test).run(
        train_grid=feasible_points, top_n=cfg.walk_forward_top_n
    )
    kc4_feasible = check_kc4(
        train_fitnesses=wf_feasible.train_fitnesses,
        test_fitnesses=wf_feasible.test_fitnesses,
        trade_ratio=wf_feasible.trade_ratio_scaled,
        thresholds=cfg.kc_thresholds,
    )
    print("[4] Same walk-forward gate, feasible candidates only")
    print(f"    feasible candidates  : {len(feasible_points)} / {len(grid_runs)}")
    print(f"    Pearson(train, test) = {wf_feasible.pearson:.3f}")
    print(f"    KC-4 verdict         = {kc4_feasible.status} ({kc4_feasible.message})")
    assert kc4_feasible.status == "PASS", (
        "the feasible ranking should transfer train->test"
    )
    print()

    # ── Scoreboard ─────────────────────────────────────────────────────────
    naive_gap = _gap_pct(best_any_train, best_any_holdout)
    gated_gap = _gap_pct(feasible_train, feasible_holdout)
    print("[5] Held-out scoreboard (holdout slice was never used for selection)")
    print()
    print("    candidate       train    holdout   gap")
    print(f"    best_any        {best_any_train:6.3f}   {best_any_holdout:6.3f}   "
          f"{naive_gap:+7.1f}%   <- collapses out-of-sample")
    print(f"    best_feasible   {feasible_train:6.3f}   {feasible_holdout:6.3f}   "
          f"{gated_gap:+7.1f}%   <- holds up")
    print()
    assert naive_gap < -50.0, f"naive pick should collapse on holdout, gap={naive_gap:.1f}%"
    assert gated_gap > -20.0, f"gated pick should hold on holdout, gap={gated_gap:.1f}%"
    assert feasible_holdout > best_any_holdout, (
        "the gated selection should out-score the naive pick out-of-sample"
    )

    print("Summary: the highest train score was a lucky-noise artifact. The")
    print("walk-forward gate (KC-4) failed it; the constraint-gated feasible-best")
    print("generalized to a slice no selection step ever saw.")
    print()
    print("Walk-forward gate demo PASSED.")
    return 0
