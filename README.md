# Omega-Lock

> Audit tuned candidates before they ship: walk-forward validation, declarative hard constraints, feasible-best selection, and an append-only JSON trail.

[![Package version](https://img.shields.io/badge/version-0.2.5-orange.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Current local package version: `0.2.5`.

Registry status is not asserted here. In this environment, PyPI/GitHub release
verification is `ENVIRONMENT_BLOCKED`; do not treat a local version string as
proof that the same version is published.

## Badge and download analytics boundaries

Static badges in this README identify local metadata surfaces such as package
version and license. They do not prove release readiness, correctness,
trustworthiness, adoption, or package quality.

Downloads or stars may indicate visibility, not correctness, trustworthiness, or
release readiness. Stars/downloads must not be used as audit evidence or release
approval. No PyPI or GitHub download analytics are asserted here.

## What omega-lock audits

Omega-Lock is an audit-first framework for tuned calibration candidates. It sits
after candidate generation and asks whether a candidate survives declared gates:

- **Walk-forward gate (KC-4)**: walk-forward re-evaluation on test target data,
  using Pearson and trade-ratio checks.
- **Declarative hard constraints**: constraints are evaluated and recorded on
  every candidate; `constraint_policy="prefer_feasible"` makes selection prefer
  candidates that satisfy all declared constraints.
- **Feasible-best vs absolute-best**: audit reports expose `best_feasible` and
  `best_any`, so reviewers can see when the highest-fitness candidate violated
  a hard constraint.
- **Append-only audit trail**: every evaluated candidate is appended as an
  `AuditedRun` with phase, role, round, and `call_index` context.
- **Optional tamper evidence**: audit reports can include an opt-in SHA-256 hash
  chain via `report.to_json(with_hash_chain=True)` and can verify it with
  `AuditReport.verify_hash_chain(...)`.

## What it does not do

- It does not grade answers or require gold labels unless your own target
  fitness function requires them.
- It does not prove root cause, guarantee correctness, or replace domain
  validation.
- It does not provide a runtime production wrapper, dashboard, or web app.
- It does not currently ship an installed console CLI. In particular,
  Omega-Lock emits JSON artifacts; it does not currently ship a console
  `omega-lock diff` command.
- It does not assert PyPI publication status for `0.2.5`; verify registries
  separately before treating a version as published.

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

## Run the deterministic demos (no API, no network)

The 60-second demo video is preserved because it shows the actual local demo
flow:

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

It is a paced replay of checked-in `examples/phantom_demo.py` output:
12-axis sensitivity, top-K unlock, grid search, walk-forward validation, KC
reports, and zoom refinement. Both runs are deterministic and require no
network or API keys.

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

## Install and import names

Name boundaries are intentionally distinct:

| Surface | Name |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| Installed console executable | none currently |

From source:

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

From PyPI, only if version `0.2.5` is published in the index you use:

```bash
pip install omega-lock==0.2.5
pip install "omega-lock[p2]==0.2.5"
```

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

Public README claims are tracked in the generated claim ledger:

- Source ledger: [docs/claims/public_claims.yml](docs/claims/public_claims.yml)
- Generated review table:
  [docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md)
- Repository surface:
  [docs/REPO_SURFACE.md](docs/REPO_SURFACE.md)
- Trust model:
  [docs/TRUST_MODEL.md](docs/TRUST_MODEL.md)
- Toolkit positioning:
  [docs/TOOLKIT_POSITIONING.md](docs/TOOLKIT_POSITIONING.md)

Regenerate and check claim artifacts offline:

```bash
python scripts/generate_readme_claims.py
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py --check
```

## Scope

Omega-Lock is a CLI/Python package/CI audit tool. It should remain offline by
default, deterministic where possible, and conservative about public claims.

## License

Apache 2.0. See [LICENSE](LICENSE).
