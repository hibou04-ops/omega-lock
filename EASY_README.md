# Omega-Lock Easy Start

> You already have candidate parameters from an optimizer or manual tuning. Omega-Lock asks: **did this candidate actually generalize?**

[![Release](https://img.shields.io/badge/release-0.2.2-orange.svg)](https://pypi.org/project/omega-lock/0.2.2/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## What It Does

Omega-Lock is **audit-first, not search-first**. It sits after candidate generation and checks:

- declared constraints
- walk-forward behavior
- holdout evidence when you provide a holdout target
- warnings for non-gating evidence modes
- JSON audit artifacts that reviewers can inspect and diff

## Install

```bash
pip install omega-lock==0.2.2
```

For optional Optuna TPE:

```bash
pip install "omega-lock[p2]==0.2.2"
```

## Start Here

For normal audit and CI usage:

```python
from omega_lock import P1Config, run_p1

result = run_p1(
    train_target=my_target,
    config=P1Config(constraint_policy="prefer_feasible"),
)

print(result.status)
print(result.warnings)
print(result.config_full)
```

Use `AuditingTarget` with `Constraint` objects when you want a full constraint pass/fail trail.

## Constraint Policy

- `record`: backward-compatible. Records violations but does not gate best-candidate selection.
- `prefer_feasible`: recommended for normal use. Feasible candidates are preferred.
- `hard_fail`: stricter release/CI gate.

## Local Demos

```bash
python examples/demo_replay.py
python examples/demo_sram.py
```

They are deterministic and require no network or API keys.

## What Changed in 0.2.2

- The dynamic PyPI version badge was replaced with a static release badge.
- README and PyPI surfaces now show the exact current install command.
- This avoids Shields/PyPI/Camo stale badge rendering.
- No runtime behavior changed beyond version metadata.

See [README.md](README.md) for the full documentation.
