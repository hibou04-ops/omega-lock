# Omega-Lock Easy Start

> You already have candidate parameters from an optimizer or manual tuning. Omega-Lock asks: **did this candidate actually generalize, and does it survive failure-boundary audits?**

[![Release](https://img.shields.io/badge/release-0.2.4-orange.svg)](https://pypi.org/project/omega-lock/0.2.4/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## What It Does

Omega-Lock is **audit-first, not search-first**. It sits after candidate generation and acts as a structural failure-boundary auditor. It checks:

- **Declared constraints**: Hard invariants evaluated and recorded on every candidate.
- **Walk-forward behavior**: Pre-declared ship gates (like KC-4 Pearson and trade-ratio thresholds) on held-out slices to prevent post-hoc criteria relaxation.
- **Holdout evidence**: Evaluates holdout targets exactly once at the end of runs to provide an honest, independent generalization datapoint.
- **Append-only audit trail**: A reviewable JSON trail that you can easily commit, diff in PRs, or check with an opt-in SHA-256 hash chain.

## Install

```bash
pip install omega-lock==0.2.4
```

For optional Optuna TPE support:

```bash
pip install "omega-lock[p2]==0.2.4"
```

## Start Here

For normal audit and CI usage, wrap your target with an `AuditingTarget` and run search with `constraint_policy="prefer_feasible"` so that the best candidate selected is the highest-fitness one that satisfies all hard constraints:

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint("must_be_feasible",
                   lambda params, result: result.metadata["sharpe"] > 0.5),
    ],
)

result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))
```

## Do I need ground truth?

**Usually no.** Omega-Lock audits structural survival, not answer correctness. A metal fatigue test does not need a "correct answer" for the metal—it needs load conditions, a stress profile, failure criteria, and thresholds. 

Omega-Lock works the same way: you provide constraints, thresholds, stress cases, and train/test/holdout slices. You only need gold labels or semantic judges if your own target's fitness function explicitly requires them.

## Constraint Policy

- `record`: Backward-compatible. Records violations on the audit trail but does not gate best-candidate selection.
- `prefer_feasible` (Recommended): Filters candidates, prioritizing those that satisfy all hard constraints.
- `hard_fail`: Stricter release/CI gate; aborts/fails the run if no candidate is feasible.

## Local Demos

```bash
# Run the sensitivity-driven top-K grid walkthrough replay
python examples/demo_replay.py

# Run a 6T SRAM bitcell multi-corner simulation surrogate with constraints
python examples/demo_sram.py
```

Both demos are deterministic and require no network or API keys.

## What Changed in 0.2.4

- Aligned version references and badges to `0.2.4`.
- Restored `README_KR.md`, `EASY_README.md`, and `EASY_README_KR.md` to perfectly align with the main README's failure-boundary auditor positioning.
- Clarified the "no ground-truth required" model across all documents.
- No runtime behavior changed beyond version metadata.

See [README.md](README.md) for the full technical documentation.
