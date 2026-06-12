# omega-lock

**The best score is lying to you — and your optimizer can't catch it.** omega-lock is the gate that runs *after* your tuner, takes its "winning" candidate, and tells you whether that score is real or just luck — **before** it ships.

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg?cacheSeconds=3600)](https://pypi.org/project/omega-lock/)
[![Python](https://img.shields.io/pypi/pyversions/omega-lock.svg?cacheSeconds=3600)](https://pypi.org/project/omega-lock/)
[![License](https://img.shields.io/pypi/l/omega-lock.svg?cacheSeconds=3600)](https://pypi.org/project/omega-lock/)

```bash
pip install omega-lock
omega-lock demo   # 60s, offline: watch a "winning" score collapse -74% on held-out data
```

> *Keywords: hyperparameter overfitting · eval / prompt regression testing · walk-forward validation · validate an Optuna study · holdout transfer check in CI.*

---

## The 30-second version

You ran a hyperparameter sweep, a prompt search, or a threshold tuner. It came back proud and pointed at the winner — the **highest score on the data you tuned against**.

That is exactly the number you can't trust. When you try hundreds of candidates and keep only the single best one, you don't just keep the most skillful one — you keep the **luckiest** one. And luck doesn't repeat. The moment you test that winner on data it has never seen, the lucky streak is gone:

```
on the data it was picked from   →   5.967   (real skill  +  a lucky streak)
on brand-new, held-out data      →   1.527   (only the real skill that was left)   ▼ -74.4%
```

This is **overfitting from selection**, and no optimizer protects you from it — finding the max is its whole job. omega-lock is your second opinion. It re-tests the winner on a slice the search never touched and returns a flat verdict: **PASS** (ship it) or **FAIL** (block it).

---

## See it fail a lucky winner — 60 seconds, nothing to set up

```bash
omega-lock demo
```

A fully offline case study: a search picks a candidate that looks brilliant in training, then omega-lock re-scores it on a held-out slice.

```
candidate: best-by-score (selected from 125 trials)
  train score    5.967
  holdout score  1.527     ▼ -74.4%
  walk-forward transfer gate ............ FAIL   (train↔holdout correlation 0.179 < 0.3)
  hard-constraint feasibility ........... FAIL   (best_feasible ≠ best_any)

VERDICT: BLOCK — the winning score did not transfer. Selection concentrated luck.
```

The optimizer was thrilled with `5.967`. The reality was `1.527`. omega-lock stamps `FAIL` and your pipeline stops the deploy. That collapse is the whole product in one screen.

---

## Drop it into CI

Point omega-lock at two score files — the scores your optimizer reported on the data it tuned against, and the scores of the *same* candidates re-evaluated on a held-out slice. It exits `0` (ship) or `1` (block):

```bash
omega-lock gate --train train_scores.json --holdout holdout_scores.json
```

```yaml
# .github/workflows/overfit-gate.yml
name: overfit-gate
on: [pull_request]

jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install omega-lock

      # your tuner runs here and writes train_scores.json + holdout_scores.json
      - run: python tune.py

      # the gate: a non-zero exit fails the check and blocks the merge
      - run: omega-lock gate --train train_scores.json --holdout holdout_scores.json
```

When the held-out score doesn't track the tuned score, the step fails red and the PR can't merge. Every run also writes an **append-only audit trail**, so you can prove later exactly what was gated, when, and why.

Prefer Python? The same decision is one call:

```python
from omega_lock.simple import gate_scores

result = gate_scores(train="train_scores.json", holdout="holdout_scores.json")
assert result.passed, result.reason   # fail your test suite on a bad candidate
```

---

## Already have an Optuna study? Gate it in 3 lines

```python
import optuna
from omega_lock import audit_optuna_study

study  = optuna.load_study(study_name="my-sweep", storage="sqlite:///sweep.db")
report = audit_optuna_study(study, holdout_evaluate=score_on_holdout)  # walk-forward + feasibility on study.best_trial
print(report.passed, report.gated_best)   # False, and the candidate it WILL certify (or None)
```

No new study, no rewrite of your objective, no config DSL. It also works on bare lists (Ax, Ray Tune, Hyperopt, `GridSearchCV`, or a hand-rolled sweep) — any leaderboard is enough.

---

## What the gate actually checks

Three independent checks on the candidate your search already chose. Any one can block it.

| Check | Plain English | Blocks when |
|---|---|---|
| **Walk-forward transfer gate** | Does the score earned on the tuned data carry over to a held-out slice it never saw? | The held-out result decorrelates from the tuned ranking — the winner was a fluke. |
| **Hard-constraint feasibility** | Is the highest-scoring candidate also a *valid* one (passes your latency / cost / risk limits), or did you win on a config you can't run? | `best_feasible ≠ best_any` — the top score violates a constraint you declared. |
| **Append-only audit trail** | Can you reconstruct the decision months later? | Never blocks — always records the verdict, inputs, and thresholds, tamper-evident. |

**Core insight:** *the highest score is the most suspicious number you own.* A real edge survives a slice it was never shown. Luck does not.

---

## omega-lock is NOT another optimizer

It does not search, sample, or propose anything. It is the **gate you bolt onto the search you already have** — keep Optuna, keep your sweep, keep your eval loop, and let omega-lock judge the output.

| | Your optimizer (Optuna / Ax / sweep) | omega-lock |
|---|---|---|
| Job | **Finds** the best score | **Tells you if** that score deploys |
| Runs | *during* the search | *after* it, on the result |
| Looks at | the data the search consumed | a held-out slice it never saw |
| Output | a leaderboard + a winner | PASS / FAIL + the certified candidate |

### Where it sits next to the tools you know

| Tool | Its job | Overlap with omega-lock |
|---|---|---|
| **Optuna / Ax / Ray Tune** | search the space, return a winner (constrained optimization) | none — omega-lock **audits their winner** |
| **MLflow / Weights & Biases** | track *what* you ran | none — omega-lock is a **pass/fail gate**, not a tracker |
| **promptfoo / DSPy / your eval harness** | score prompt & model outputs | none — omega-lock catches the prompt that aced the eval but won't generalize |

The empty seat omega-lock fills: an **output-side overfit gate**. The rule of thumb — *if a number was chosen by trying many options and keeping the best, it belongs behind this gate.* For where omega-lock does and does not fit the wider toolbox, see [docs/TOOLKIT_POSITIONING.md](docs/TOOLKIT_POSITIONING.md).

---

## Install

```bash
pip install omega-lock

omega-lock demo                 # 60s offline walkthrough — watch a lucky winner collapse
omega-lock gate --help          # the CI gate (exit 0 = ship, 1 = block)
```

Generate a shareable dark-themed scorecard from any gate run with `render_html` — attach it to a PR or archive it.

---

**READMEs:** [Easy / plain-English README](EASY_README.md) · [한국어 README](README_KR.md) · [쉬운 한국어 README](EASY_README_KR.md) — **Docs:** [How the transfer gate works](docs/HOW_IT_WORKS.md) · [Power API for integrators](docs/API.md) · [Trust & audit model](docs/TRUST_MODEL.md) · [Toolkit positioning](docs/TOOLKIT_POSITIONING.md) · [CHANGELOG](CHANGELOG.md)

<sub>**Badge and download analytics boundaries.** The badges above are static or registry-served links; they do not prove release readiness, correctness, trustworthiness, adoption, or package quality. Downloads or stars may indicate visibility, not skill — stars/downloads must not be used as audit evidence or release approval. No PyPI or GitHub download analytics are asserted here. Only the gate's PASS/FAIL on held-out data is evidence.</sub>

<sub>**Note on terms.** This page uses plain language; the public Python API keeps its established symbols for backward compatibility (other repos import them). In code you may see: `run_p1` / `P1Config` (run the gate + its config), `check_kc4` / `KCThresholds` (the walk-forward transfer check + its pass thresholds, e.g. minimum transfer correlation), `measure_stress` (rank parameters by perturbation sensitivity), `ParamSpec` (a tunable parameter's range), `EvalResult` (one scored candidate). You never need these to use `omega-lock demo`, `omega-lock gate`, or `omega_lock.simple.gate_scores()`. Full reference in [docs/API.md](docs/API.md).</sub>
