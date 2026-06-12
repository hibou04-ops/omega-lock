# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Optuna bridge demo — gate an EXISTING Optuna study with Omega-Lock.

You already ran an Optuna study. Its `study.best_trial` is the highest
score seen on the training data — which says nothing about whether that
candidate transfers or respects your hard constraints. Since 0.3.4 the
bridge this demo originally sketched by hand is a real API:

    from omega_lock import audit_optuna_study

    report = audit_optuna_study(study, holdout_evaluate=score_on_holdout)
    report.passed        # KC-4 walk-forward gate verdict
    report.best_any      # study winner, constraints ignored
    report.gated_best    # what the gate is willing to certify (or None)

No re-search is needed: the completed trials ARE the candidate set.
Feasibility is inferred from the per-trial `user_attrs["feasible"]` flag
set inside the objective below.

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
from typing import Any

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

from omega_lock import KCThresholds, audit_optuna_study, render_html

# Reuse the deterministic noisy system from the walk-forward gate demo:
# real optimum ~5.0 inside the rated envelope (gain <= 6), plus a fragile
# region where train-slice noise spikes up to ~6 but never transfers.
from walkforward_gate_demo import (
    HOLDOUT_SEED,
    RATED_GAIN_MAX,
    TEST_SEED,
    TRAIN_SEED,
    NoisySliceTarget,
)

N_TRIALS = 120
SAMPLER_SEED = 7
GATE_TOP_N = 20  # Pearson over very few points is high-variance; use 20

REPORT_HTML = HERE.parent / "output" / "optuna_audit_scorecard.html"


def build_existing_study() -> "optuna.study.Study":
    """Stand-in for 'a study you already have' — seeded and offline.

    The objective records the declared envelope constraint as a per-trial
    `user_attrs["feasible"]` flag — the convention `audit_optuna_study`
    reads to split `best_any` from `best_feasible`.
    """
    train = NoisySliceTarget(TRAIN_SEED)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "gain": trial.suggest_float("gain", 0.0, 10.0),
            "bias": trial.suggest_float("bias", -3.0, 3.0),
            "offset": 0.0,
        }
        trial.set_user_attr("feasible", params["gain"] <= RATED_GAIN_MAX)
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

    # ── The bridge: one call replaces the hand-rolled GridPoint plumbing ──
    test_slice = NoisySliceTarget(TEST_SEED)

    def score_on_test_slice(params: dict[str, Any]) -> float:
        # The study searched (gain, bias) only; the target also needs the
        # locked decoy axis. A holdout_evaluate callable is exactly where
        # that adaptation belongs.
        return test_slice.evaluate({**params, "offset": 0.0}).fitness

    report = audit_optuna_study(
        study,
        holdout_evaluate=score_on_test_slice,
        thresholds=KCThresholds.pure_objective(),  # no action-count gates
        top_n=GATE_TOP_N,
    )
    # ── End of bridge ────────────────────────────────────────────────────

    best_any = report.best_any
    best_feasible = report.best_feasible
    assert best_feasible is not None, "objective sets the feasibility flag"

    holdout = NoisySliceTarget(HOLDOUT_SEED)
    best_any_holdout = holdout.evaluate({**best_any.params, "offset": 0.0}).fitness
    best_feasible_holdout = holdout.evaluate(
        {**best_feasible.params, "offset": 0.0}
    ).fitness

    n_feasible = sum(
        1 for t in study.trials if t.user_attrs.get("feasible") is True
    )

    print("=" * 72)
    print(" Optuna bridge: gate an existing study's trials with Omega-Lock")
    print("=" * 72)
    print()
    print(f" study trials bridged : {report.n_trials_completed} (completed)")
    print(f" study.best_trial     : fitness={study.best_value:.3f} "
          f"params={ {k: round(v, 3) for k, v in study.best_params.items()} }")
    print()
    print(f"[1] Walk-forward gate (KC-4) over the study's top-{report.top_n} by train fitness")
    assert report.pearson is not None
    print(f"    Pearson(train, test) = {report.pearson:.3f} "
          f"(threshold {report.thresholds.pearson_min})")
    print(f"    KC-4 verdict         = {report.kc_report.status} "
          f"({report.kc_report.message})")
    print()
    print("[2] Feasible-best selection (declared constraint: "
          f"gain <= {RATED_GAIN_MAX:.0f}, via trial user_attrs['feasible'])")
    print(f"    feasibility source   : {report.feasibility_source}")
    print(f"    feasible trials      : {n_feasible} / {report.n_trials_completed}")
    print(f"    gated_best           : "
          f"{'(none - gate refused to certify)' if report.gated_best is None else report.gated_best.number}")
    print()
    print("[3] Held-out scoreboard (slice never seen by the study)")
    print()
    print("    candidate       train    holdout")
    print(f"    best_any        {best_any.train_value:6.3f}   {best_any_holdout:6.3f}"
          "   <- study winner, lucky noise")
    print(f"    best_feasible   {best_feasible.train_value:6.3f}   "
          f"{best_feasible_holdout:6.3f}   <- survives out-of-sample")
    print()

    html_path = render_html(report, REPORT_HTML)
    print(f"[4] HTML scorecard written: {REPORT_HTML}")
    assert "<svg" in html_path

    assert report.kc_report.status == "FAIL", (
        f"expected the noise-chasing study ranking to fail KC-4, got {report.kc_report.status}"
    )
    assert not report.passed
    assert report.gated_best is None, "a failed gate certifies nothing"
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
