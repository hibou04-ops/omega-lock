# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Optuna bridge demo — gate an EXISTING Optuna study with Omega-Lock.

You already ran an Optuna study. Its `study.best_trial` is the highest
score seen on the training data — which says nothing about whether that
candidate transfers or respects your hard constraints. This demo shows
the bridge in ~15 lines of code:

    Optuna trials -> GridPoint list -> WalkForward gate (KC-4)
                                    -> feasible-best selection

No re-search is needed: the completed trials ARE the candidate set.

Two related surfaces, do not confuse them:
    * This demo: audit an EXISTING study after the fact.
    * `run_p2_tpe` (optional `[p2]` extra): run a NEW search where Optuna
      TPE replaces the grid inside the full P1 gate pipeline
      (stress -> KC-2 -> TPE -> KC-4 -> KC-1/KC-3).

Optuna is an optional dependency. Without it this demo skips gracefully:

    pip install "omega-lock[p2]"     # or: pip install "optuna>=3.0"

Run:
    python examples/optuna_audit_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/optuna_audit_demo.py` without pip install
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import optuna
except ImportError:
    print("optuna is not installed - skipping the Optuna bridge demo.")
    print('Install it with: pip install "omega-lock[p2]"  (or optuna>=3.0)')
    sys.exit(0)

from omega_lock import EvalResult, KCThresholds, WalkForward, check_kc4
from omega_lock.grid import GridPoint

# Reuse the deterministic noisy system from the walk-forward gate demo:
# real optimum ~5.0 inside the rated envelope (gain <= 6), plus a fragile
# region where train-slice noise spikes up to ~6 but never transfers.
from walkforward_gate_demo import (
    HOLDOUT_SEED,
    RATED_ENVELOPE,
    RATED_GAIN_MAX,
    TEST_SEED,
    TRAIN_SEED,
    NoisySliceTarget,
)

N_TRIALS = 120
SAMPLER_SEED = 7
GATE_TOP_N = 20  # Pearson over very few points is high-variance; use 20


def build_existing_study() -> "optuna.study.Study":
    """Stand-in for 'a study you already have' — seeded and offline."""
    train = NoisySliceTarget(TRAIN_SEED)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "gain": trial.suggest_float("gain", 0.0, 10.0),
            "bias": trial.suggest_float("bias", -3.0, 3.0),
            "offset": 0.0,
        }
        return train.evaluate(params).fitness

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study


def main() -> int:
    study = build_existing_study()

    # ── The bridge: Optuna trials -> walk-forward gate + feasible-best ──
    points = [
        GridPoint(
            idx=t.number,
            unlocked=dict(t.params),
            params={**t.params, "offset": 0.0},
            result=EvalResult(fitness=t.value, sample_count=1),
        )
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    thresholds = KCThresholds.pure_objective()      # no action-count gates
    wf = WalkForward(test_target=NoisySliceTarget(TEST_SEED))
    wf_result = wf.run(train_grid=points, top_n=GATE_TOP_N)
    kc4 = check_kc4(
        train_fitnesses=wf_result.train_fitnesses,
        test_fitnesses=wf_result.test_fitnesses,
        trade_ratio=wf_result.trade_ratio_scaled,
        thresholds=thresholds,
    )
    feasible = [gp for gp in points if RATED_ENVELOPE.fn(gp.params, gp.result)]
    best_any = max(points, key=lambda gp: gp.result.fitness)
    best_feasible = max(feasible, key=lambda gp: gp.result.fitness)
    # ── End of bridge ────────────────────────────────────────────────────

    holdout = NoisySliceTarget(HOLDOUT_SEED)
    best_any_holdout = holdout.evaluate(best_any.params).fitness
    best_feasible_holdout = holdout.evaluate(best_feasible.params).fitness

    print("=" * 72)
    print(" Optuna bridge: gate an existing study's trials with Omega-Lock")
    print("=" * 72)
    print()
    print(f" study trials bridged : {len(points)} (completed)")
    print(f" study.best_trial     : fitness={study.best_value:.3f} "
          f"params={ {k: round(v, 3) for k, v in study.best_params.items()} }")
    print()
    print(f"[1] Walk-forward gate (KC-4) over the study's top-{GATE_TOP_N} by train fitness")
    print(f"    Pearson(train, test) = {wf_result.pearson:.3f} "
          f"(threshold {thresholds.pearson_min})")
    print(f"    KC-4 verdict         = {kc4.status} ({kc4.message})")
    print()
    print("[2] Feasible-best selection (declared constraint: "
          f"gain <= {RATED_GAIN_MAX:.0f})")
    print(f"    feasible trials      : {len(feasible)} / {len(points)}")
    print()
    print("[3] Held-out scoreboard (slice never seen by the study)")
    print()
    print("    candidate       train    holdout")
    print(f"    best_any        {best_any.result.fitness:6.3f}   {best_any_holdout:6.3f}"
          "   <- study winner, lucky noise")
    print(f"    best_feasible   {best_feasible.result.fitness:6.3f}   "
          f"{best_feasible_holdout:6.3f}   <- survives out-of-sample")
    print()

    assert kc4.status == "FAIL", (
        f"expected the noise-chasing study ranking to fail KC-4, got {kc4.status}"
    )
    assert best_any.params["gain"] > RATED_GAIN_MAX, (
        "expected the study winner to sit in the fragile region"
    )
    assert best_feasible_holdout > best_any_holdout, (
        "feasible-best should out-score the raw study winner on holdout"
    )

    print("Summary: study.best_trial chased slice noise; the walk-forward gate")
    print("failed it, and the constraint-feasible best held up on holdout.")
    print("For a fresh gated TPE search, see omega_lock.run_p2_tpe.")
    print()
    print("Optuna bridge demo PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
