# Omega-Lock — Easy Start

> You already have candidate parameters from some optimizer. Omega-Lock asks: **did this candidate actually generalize?**

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## What It Does

Omega-Lock is **audit-first, not search-first**. It sits after your optimizer and checks whether the tuned candidate is safe to trust.

It records:

- constraints and whether each candidate passed them
- walk-forward behavior on test data
- holdout evidence when you provide a holdout target
- warnings when a mode records evidence but does not gate selection/status
- JSON audit artifacts reviewers can inspect and diff

## Install

```bash
pip install omega-lock
```

For optional Optuna TPE:

```bash
pip install "omega-lock[p2]"
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

Use an `AuditingTarget` with `Constraint` objects when you want a full trail of constraint pass/fail records.

## Constraint Policy

- `record`: backward-compatible. Records violations but does not gate best-candidate selection.
- `prefer_feasible`: recommended for normal use. Feasible candidates are preferred.
- `hard_fail`: stricter release/CI gate.

## What Changed in 0.1.9

- README and PyPI long description were sharpened.
- PyPI badges are dynamic.
- Korean docs were regenerated as valid UTF-8.
- Release checklist was added so version, tag, dist artifacts, and PyPI stay in sync.
- No runtime behavior changed.

See [README.md](README.md) for the full documentation.
