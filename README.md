# Omega-Lock

> Audit tuned candidates before they ship: walk-forward validation, declarative
> hard constraints, feasible-best selection, and append-only JSON audit trails.

Omega-Lock runs **after candidate generation**. A search, tuning, or calibration
method proposes a candidate; Omega-Lock decides whether that candidate survives
the declared evidence gates before it is allowed to ship.

[![Version 0.3.3](https://img.shields.io/badge/version-0.3.3-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

**README family:** [Full README](README.md) · [한국어 README](README_KR.md) ·
[Easy README](EASY_README.md) · [쉬운 한국어 README](EASY_README_KR.md)

## The problem: the best score is not deployable

Every optimizer hands back its highest-scoring candidate. That number answers
"what scored highest on the data the search consumed?" — not "does it survive
out-of-sample?" and not "does it respect the hard constraints?". Selection
pressure concentrates luck at the top of the leaderboard: noise spikes,
constraint violations, and slice-specific artifacts. Omega-Lock treats the raw
winner as untrusted until it passes the declared gates — a walk-forward
transfer check (KC-4), hard-constraint feasibility (`best_feasible` vs
`best_any`), and an append-only audit trail a reviewer can replay.

## Quickstart: watch the gate catch an overfit (offline, < 60 s)

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/walkforward_gate_demo.py
```

A seeded synthetic search finds a lucky-noise "winner" that out-scores the
true optimum on the train slice. The demo prints the whole story with numbers:
naive `best_any` collapses out-of-sample (train 5.967 -> holdout 1.527,
-74.4%), the walk-forward gate stamps the run `FAIL:KC-4` (Pearson 0.179,
threshold 0.3), and the constraint-gated `best_feasible` holds up (train 5.233
-> holdout 5.276) on a slice no selection step ever consulted. Deterministic:
your run prints the same numbers.

Already have an Optuna study? Bridge its completed trials through the same
walk-forward gate and feasible-best selection in about 15 lines (the demo
skips cleanly when optuna is not installed):

```bash
pip install "omega-lock[p2]"   # optional Optuna extra
python examples/optuna_audit_demo.py
```

`run_p2_tpe` is the integrated variant: a fresh Optuna TPE search inside the
full gate pipeline (stress -> KC-2 -> TPE -> KC-4 -> KC-1/KC-3).

## Terminology decoder

This codebase uses a compact internal dialect. The table below decodes it:

| Term | Meaning |
| --- | --- |
| `P1` / `run_p1` | The calibration audit pipeline: baseline -> stress -> top-K unlock -> grid search -> walk-forward -> kill-criteria verdict. |
| `P2` / `run_p2_tpe` | The same gates with Optuna TPE search replacing the grid (optional `[p2]` extra). |
| `KC-1` | Kill criterion 1: time box — the run must finish within a declared wall-clock budget. |
| `KC-2` | Kill criterion 2: stress differentiation — per-parameter sensitivities must separate (Gini + top/bottom ratio); a flat profile means the search is noise-mining. |
| `KC-3` | Kill criterion 3: action-count floor — a minimum number of actions (e.g. trades, samples) behind the best config. |
| `KC-4` | Kill criterion 4: the walk-forward gate — Pearson correlation between train and test fitness over the top-N candidates, plus an action-ratio check. |
| `SC-2` | Sanity control 2: a same-budget random-search baseline. Advisory only — flags runs where grid search does not beat random sampling (Bergstra & Bengio 2012). |
| `best_any` | The highest-fitness candidate, constraints ignored. |
| `best_feasible` | The highest-fitness candidate that satisfies every declared hard constraint. |
| stress / unlock / lock | Per-parameter perturbation sensitivity; the top-K most sensitive parameters are searched ("unlocked"), the rest stay fixed ("locked"). |
| `KCThresholds.pure_objective()` | Preset that disables the action-count gates (KC-3 and the KC-4 action-ratio sub-gate) for non-action objectives (math, ML, simulation). |

Release notes: [CHANGELOG.md](CHANGELOG.md) · short per-release summaries
(including 0.3.3) moved to [docs/WHATS_NEW.md](docs/WHATS_NEW.md).

## Use it when

- before shipping a tuned or calibrated candidate
- when the highest-fitness candidate may violate a hard constraint
- when reviewers need `best_any` and `best_feasible` reported separately
- when train/test or holdout transfer needs a walk-forward gate
- when an append-only JSON audit trail is needed for review or CI
- when deterministic, offline release hygiene matters
- when calibrating non-action objectives (math, ML, simulation) — see `KCThresholds.pure_objective()`

## Trust loop

1. generate or receive candidate parameters
2. evaluate them through `AuditingTarget`
3. record hard-constraint outcomes on every candidate
4. select `best_feasible` separately from `best_any`
5. apply walk-forward or holdout gates when configured
6. emit JSON result, audit report, and scorecard
7. optionally serialize with SHA-256 hash-chain evidence
8. verify generated claims and repository consistency offline

## Install

Current local package version: `0.3.3`. This README does not assert PyPI or
GitHub release status. Local version metadata is not proof of registry
publication; registry status requires explicit post-release verification.

```bash
pip install omega-lock==0.3.3
pip install "omega-lock[p2]==0.3.3"
```

Use the PyPI command only after `0.3.3` is visible in the package index you use.
Local version metadata is not proof of registry publication.

From source:

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

## Verification and evidence

Public README claims are tracked in a generated claim ledger. Local checks can
verify the documentation/source alignment; registry publication still requires
explicit post-release verification.

- Claim ledger (source): [docs/claims/public_claims.yml](docs/claims/public_claims.yml)
- Generated claim review: [docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md)
- Repository surface: [docs/REPO_SURFACE.md](docs/REPO_SURFACE.md)
- Trust model: [docs/TRUST_MODEL.md](docs/TRUST_MODEL.md)
- Toolkit positioning: [docs/TOOLKIT_POSITIONING.md](docs/TOOLKIT_POSITIONING.md)
- Release checklist: [RELEASE.md](RELEASE.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Offline quality CI: [.github/workflows/quality-ci.yml](.github/workflows/quality-ci.yml)
- Publish workflow: [.github/workflows/publish.yml](.github/workflows/publish.yml)

Regenerate and check claim artifacts offline:

```bash
python scripts/generate_readme_claims.py
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py --check
```

## Run the deterministic demos (no API, no network)

No API keys and no network access are required.

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

`demo_replay.py` is a paced replay of checked-in `examples/phantom_demo.py`
output — 12-axis sensitivity, top-K unlock, grid search, walk-forward
validation, KC reports, and zoom refinement. Both runs are deterministic and
require no network or API keys.

The 60-second demo video shows the same local flow:

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

## How is this different?

| Capability | omega-lock | Generic optimizer | Ad-hoc grid/random search | Benchmark-only report |
| --- | --- | --- | --- | --- |
| Treats raw winner as untrusted until audited | ✓ | ✗ | ✗ | partial |
| Separates `best_any` from `best_feasible` | ✓ | ✗ | ✗ | ✗ |
| Records declared hard-constraint outcomes per candidate | ✓ | varies | manual | ✗ |
| Supports walk-forward / holdout gate when configured | ✓ | varies | manual | varies |
| Emits reviewable JSON audit artifacts | ✓ | varies | manual | report-only |
| Optional SHA-256 hash-chain tamper evidence | ✓ | ✗ | ✗ | ✗ |
| Generated README claim ledger | ✓ | ✗ | ✗ | ✗ |
| Claims global optimum or domain correctness | ✗ | sometimes | ✗ | ✗ |

Position: Omega-Lock is audit-gate-first, not optimizer-replacement-first.
Optimizers answer "what scored highest?" Omega-Lock answers "what survived the
declared evidence gates?"

## What this is not

- not answer grading or gold-label scoring
- not proof of correctness
- not root-cause proof
- not a production runtime wrapper, dashboard, or web app
- not cryptographic signing or immutable storage
- not a published-registry verifier — registry status requires explicit
  post-release verification
- no installed console command — Omega-Lock does not currently ship a console
  `omega-lock diff` command

## What omega-lock audits

Omega-Lock is an audit-first framework for tuned calibration candidates. It sits
after candidate generation and asks whether a candidate survives declared gates:

- **Walk-forward gate (KC-4)**: walk-forward re-evaluation on test target data,
  using Pearson and trade-ratio checks.
- **Pure-objective preset (0.3.0)**: `KCThresholds.pure_objective()` disables the
  action-count gates (KC-3 and the KC-4 trade-ratio sub-gate) and keeps the
  domain-neutral gates, so non-action objectives are not forced through
  action-count floors.
- **Declarative hard constraints**: constraints are evaluated and recorded on
  every candidate; `constraint_policy="prefer_feasible"` makes selection prefer
  candidates that satisfy all declared constraints.
- **Feasible-best vs absolute-best**: audit reports expose `best_feasible` and
  `best_any`, so reviewers can see when the highest-fitness candidate violated
  a hard constraint.
- **Append-only audit trail**: every evaluated candidate is appended as an
  `AuditedRun` — with phase, role, round, and `call_index` context — to an
  append-only JSON trail.
- **Optional tamper evidence**: audit reports can include an opt-in SHA-256 hash
  chain via `report.to_json(with_hash_chain=True)` and can verify it with
  `AuditReport.verify_hash_chain(...)`.

## Why feasible-best matters

The absolute-best candidate can be the wrong candidate to ship if it violates a
hard constraint. `best_any` answers "what scored highest?" while
`best_feasible` answers "what scored highest while satisfying the declared
constraints?" In audit and CI contexts, the second answer is often the one that
can actually move forward.

Use `constraint_policy="prefer_feasible"` for normal audit runs. Use
`constraint_policy="hard_fail"` when a run with no feasible candidate should
fail immediately. The backward-compatible default, `record`, records constraint
violations but does not gate `grid_best` selection.

## Install and import names

Name boundaries are intentionally distinct:

| Surface | Name |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| Installed console executable | none currently |

Python import:

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard
```

## Minimal audit example

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint(
            "must_be_feasible",
            lambda params, result: result.metadata["sharpe"] > 0.5,
        ),
    ],
)

result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))  # feasible best vs absolute best
```

For tamper-evident audit reports:

```python
signed = report.to_json(with_hash_chain=True)
rehydrated = type(report).from_json(signed)
# Pass the embedded hash_chain from the parsed JSON object to verify_hash_chain.
```

## Benchmark and claim evidence

`run_benchmark` and `examples/benchmark_battery.py` produce an objective
scorecard from mechanically computed metrics such as effective recall,
generalization gap, and `stress_rank_spearman`.

The checked-in benchmark regression fixture tracks deterministic
`stress_rank_spearman` values in the frozen fixture. This is a regression
signal, not a claim that Omega-Lock is superior to other optimizers.

The public claim ledger and its proof links are listed under
[Verification and evidence](#verification-and-evidence) above.

## Badge and download analytics boundaries

Static badges in this README identify local metadata surfaces, supported Python
version, local quality gates, and methodology positioning. They do not prove
release readiness, correctness, trustworthiness, adoption, or package quality.

Downloads or stars may indicate visibility, not correctness, trustworthiness, or
release readiness. Stars/downloads must not be used as audit evidence or release
approval. No PyPI or GitHub download analytics are asserted here.

## Scope

Omega-Lock is a CLI/Python package/CI audit tool. It should remain offline by
default, deterministic where possible, and conservative about public claims.

## License

Apache 2.0. See [LICENSE](LICENSE).
