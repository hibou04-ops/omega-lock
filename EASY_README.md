# Omega-Lock Easy Start

Current local package version: `0.2.6`.

[![Version 0.2.6](https://img.shields.io/badge/version-0.2.6-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

[Full README](README.md) · [한국어 README](README_KR.md) · [쉬운 한국어 README](EASY_README_KR.md)

Omega-Lock audits tuned candidates before they ship. It checks whether a
candidate survives walk-forward validation, declared hard constraints, and a
reviewable append-only audit trail.

It does not grade answers, prove correctness, provide a dashboard, or install a
console command. There is currently no installed `omega-lock diff` command.

## The core idea

The highest-fitness candidate is not always the safest candidate. If it violates
a declared constraint, the audit report can still show it as `best_any`, while
`best_feasible` shows the highest-fitness candidate that satisfies the hard
constraints.

For most audit/CI runs, use:

```python
P1Config(constraint_policy="prefer_feasible")
```

## Run offline demos

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

Both demos are deterministic and require no network or API keys. The replay is
the same demo flow shown in the 60-second video:

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

## Install names

| Surface | Name |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| Installed console executable | none currently |

Use PyPI only if version `0.2.6` is published in your package index:

```bash
pip install omega-lock==0.2.6
pip install "omega-lock[p2]==0.2.6"
```

## Minimal use

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint("must_be_feasible", lambda params, result: result.metadata["sharpe"] > 0.5),
    ],
)

result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))
```

For proof behind README claims, see
[docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md).
