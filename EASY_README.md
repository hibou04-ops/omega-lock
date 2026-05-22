# Omega-Lock Easy Start

Current local package version: `0.2.5`.

[![Package version](https://img.shields.io/badge/version-0.2.5-orange.svg)](pyproject.toml)

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

Use PyPI only if version `0.2.5` is published in your package index:

```bash
pip install omega-lock==0.2.5
pip install "omega-lock[p2]==0.2.5"
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
