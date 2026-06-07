# Omega-Lock

> Audit tuned candidates before they ship: walk-forward validation, declarative
> hard constraints, feasible-best selection, and append-only JSON audit trails.

Omega-Lock runs **after candidate generation**. A search, tuning, or calibration
method proposes a candidate; Omega-Lock decides whether that candidate survives
the declared evidence gates before it is allowed to ship.

[![Version 0.3.2](https://img.shields.io/badge/version-0.3.2-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

**README family:** [Full README](README.md) · [한국어 README](README_KR.md) ·
[Easy README](EASY_README.md) · [쉬운 한국어 README](EASY_README_KR.md)

Current local package version: `0.3.2`. This README does not assert PyPI or
GitHub release status. Local version metadata is not proof of registry
publication; registry status requires explicit post-release verification.

## What's new in 0.3.2

Packaging fix plus a dormant, default-off execution seam — no behavior change
for existing users:

- Source-distribution packaging fix: the published sdist now ships `scripts/`,
  `tests/` (including `tests/fixtures/`), and `examples/`, so
  `pip download omega-lock --no-binary :all:` then `pytest --collect-only` on
  the unpacked sdist has zero collection errors. The wheel is unchanged and
  still ships only the import package.
- Optional parallel-execution seam: `GridSearch.run`, `ZoomingGridSearch.run`,
  `measure_stress`, and `WalkForward.run` now accept an optional
  `executor: concurrent.futures.Executor | None = None`. The default (`None`)
  is strictly serial and byte-identical to prior behavior; when an executor is
  supplied, results are reassembled in input order. These are additive optional
  keyword arguments only — no consumed surface changed.
- Golden audit fixtures regenerated only to carry the new version string; the
  default-off seam produces zero additional golden change.

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

```bash
pip install omega-lock==0.3.2
pip install "omega-lock[p2]==0.3.2"
```

Use the PyPI command only after `0.3.2` is visible in the package index you use.
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
