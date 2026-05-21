# Omega-Lock

> **Post-optimizer audit gate for tuned candidates.** Use Omega-Lock after an optimizer or manual tuning step to check whether the chosen candidate generalizes, stays inside declared constraints, and leaves a JSON artifact reviewers can inspect.

[![Release](https://img.shields.io/badge/release-0.2.2-orange.svg)](https://pypi.org/project/omega-lock/0.2.2/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

```bash
pip install omega-lock==0.2.2
```

Omega-Lock is **audit-first, not search-first**. It does not try to be the universal optimizer. It sits after grid search, Optuna, Bayesian search, an internal tuner, or a candidate chosen by hand, then asks the release question:

> Did the tuned candidate actually generalize, and is it still inside the constraints we declared before looking at the result?

The failure it prevents is common: the highest-scoring training candidate gets shipped even though it overfits, violates a hard constraint, or cannot be explained later from a reproducible artifact.

For normal audit and CI usage, start with:

```python
P1Config(constraint_policy="prefer_feasible")
```

`prefer_feasible` keeps selection practical: among candidates that satisfy declared constraints, it prefers the highest-fitness one. Use `hard_fail` for stricter release gates. Use `record` only when you need backward-compatible behavior that records constraint violations without gating selection.

## Quick Start

This example is copy-paste friendly and uses the public `CalibrableTarget`, `Constraint`, `run_p1`, and `P1Config` APIs.

```python
from typing import Any

from omega_lock import EvalResult, P1Config, ParamSpec, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report


class ToyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec("x", "float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec("risk", "float", low=0.0, high=1.0, neutral=0.5),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x = float(params["x"])
        risk = float(params["risk"])
        fitness = 1.0 - abs(x - 0.8) - 0.4 * risk
        return EvalResult(
            fitness=fitness,
            n_trials=100,
            metadata={"risk": risk},
        )


def risk_ok(params: dict[str, Any], result: EvalResult) -> bool:
    return float(result.metadata["risk"]) <= 0.6


target = AuditingTarget(
    ToyTarget(),
    constraints=[
        Constraint(
            "risk_ok",
            risk_ok,
            "Risk must stay at or below 0.6.",
        )
    ],
)

result = run_p1(
    train_target=target,
    config=P1Config(
        unlock_k=2,
        grid_points_per_axis=5,
        constraint_policy="prefer_feasible",
        stress_verbose=False,
        grid_verbose=False,
    ),
)
report = make_report(target, method="run_p1", seed=None)

print(result.status)
if result.warnings:
    print(result.warnings)
print(result.config_full["constraint_policy"])
print(result.search_settings)
print(report.best_feasible.params if report.best_feasible else None)
```

The `P1Result` is JSON-serializable. Use `result.save(path)` when you want to persist the run artifact.

## Constraint Policy

- `record`: backward-compatible. Records constraint violations in the audit trail but does not gate best-candidate selection.
- `prefer_feasible`: recommended for normal audit and CI. Feasible candidates are preferred when selecting `grid_best`.
- `hard_fail`: strict release/CI gate. If no feasible result should pass, the run status fails instead of silently falling back.

## Local Demo

There is no tracked demo video in this repository. The old video placeholder has been removed.

Run deterministic demos locally:

```bash
python examples/demo_replay.py
python examples/demo_sram.py
```

They require no network and no API keys. The replay shows sensitivity measurement, top-K unlock, grid search, walk-forward validation, KC gate results, and artifact output from a captured deterministic run. The SRAM demo shows a 6T bitcell audit across PVT corners with declared constraints and writes `output/audit_sram.json`.

Subtitle/transcript file for a possible screencast: [docs/demo/omega-lock-demo.en.srt](docs/demo/omega-lock-demo.en.srt)

## What Omega-Lock Records

- `schema_version` and `omega_lock_version`
- `config_full`, `kc_thresholds`, and `search_settings`
- selected candidate, grid results, stress ranking, and KC reports
- walk-forward evidence, including `pearson_status` and `pearson_computable`
- optional holdout evidence when you pass `holdout_target`
- warnings when a mode records evidence but does not gate selection/status
- audit trails from `AuditingTarget`, including feasible vs. absolute-best candidates

These fields are intended to make run artifacts reproducible, diffable, and reviewable in CI or release review.

## When To Use It

Use Omega-Lock when a tuned candidate affects a real downstream decision:

- model or strategy parameter tuning
- hardware or simulation calibration with hard physical constraints
- process control or materials discovery
- optimizer governance where reviewers need an artifact, not just a score

It is probably overkill for a throwaway toy search where nobody will inspect the result.

## API Surface

Core protocol:

- `CalibrableTarget.param_space() -> list[ParamSpec]`
- `CalibrableTarget.evaluate(params: dict[str, Any]) -> EvalResult`

Main runners:

- `run_p1`: grid/zoom-grid search with the standard audit gates
- `run_p1_iterative`: repeated lock-in rounds for higher effective dimension
- `run_p2_tpe`: optional Optuna TPE path, installed with `pip install "omega-lock[p2]"`

Audit helpers:

- `AuditingTarget`: wraps any target and records every evaluation
- `Constraint`: named predicate over `(params, EvalResult) -> bool`
- `make_report` / `render_scorecard`: produce human-readable and JSON audit summaries

## Advanced Notes

Omega-Lock still exposes the deeper research machinery for users who need it:

- KC-1..4 kill criteria for time, sensitivity, action count, and walk-forward evidence
- stress measurement and top-K parameter unlock
- zooming grid and iterative coordinate lock-in
- optional random-search baseline and benchmark scorecards
- adapter patterns for bringing your own optimizer

Those concepts are implementation and review tools. The first decision remains simpler: use Omega-Lock after candidate generation to determine whether the candidate deserves trust.

## Release History

**0.2.2** (2026-05-22) - **Badge hardening and release-surface synchronization.** Replaces the dynamic PyPI version badge with a static release badge to avoid Shields/PyPI/Camo stale badge rendering, synchronizes the current install command and citation, and keeps the README/PyPI release surface unambiguous. No runtime behavior changes beyond version metadata.

**0.2.1** (2026-05-22) - **Release sync and badge cache-bust correction.** Updated the dynamic PyPI badge URL with a release-specific cache-bust query, synchronized release metadata after the 0.2.0 upload, and kept README/PyPI surfaces aligned. No runtime behavior changes beyond version metadata.

**0.2.0** (2026-05-22) - **Public README and release-surface polish.** Sharpens the GitHub/PyPI first screen, removes the misleading missing-video placeholder, makes the quickstart API-correct with `Constraint`, updates English and Korean docs, adds cache-conscious dynamic PyPI badges, and updates release checklist examples for 0.2.0. No runtime behavior changes beyond version metadata.

**0.1.9** (2026-05-21) - **README, PyPI metadata, and release hygiene correction.** Sharpened the GitHub/PyPI long description around Omega-Lock's actual position: audit-first, post-optimizer, constraint-aware, and artifact-producing. Kept the PyPI badge dynamic, removed stale badge/version references, rewrote Korean documentation as valid UTF-8, and added an explicit release checklist so GitHub tags, package metadata, fresh `dist/` artifacts, and PyPI uploads stay synchronized. No runtime behavior changes.

**0.1.8** (2026-05-21) - **Audit reliability and static hygiene release.** Established a clean baseline across pytest, pyright, and ruff: 289 tests passing, `pyright src tests` at 0 errors, and `ruff check src tests` clean. Static hygiene work tightened optional `optuna` imports for `run_p2_tpe`, cleaned `CalibrableTarget` Protocol conformance in tests, and fixed hash-chain typing without changing JSON shape. Audit artifacts gained reproducibility metadata such as `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, and `search_settings`. Safety signals became explicit for `constraint_policy="record"`, `holdout_mode="evidence_only"`, iterative test reuse, and walk-forward Pearson computability.

**0.1.4** (2026-04-20) - Added the `omega_lock.audit` module with `AuditingTarget`, `Constraint`, `AuditReport`, `make_report`, and `render_scorecard`, plus the SRAM audit demo.

**0.1.3** (2026-04-18) - Initial public release with P1, iterative P1, optional P2 TPE, perturbation sensitivity, walk-forward validation, KC-1..4, holdout support, benchmark support, and reference keyholes.

## Release Checklist

See [RELEASE.md](RELEASE.md). PyPI does not support overwriting an already uploaded version: always bump the version, delete `dist/`, build fresh artifacts, and verify the package before upload.

## Citation

If you use Omega-Lock in research or a published project, please cite:

```bibtex
@software{omega_lock_2026,
  author  = {hibou},
  title   = {Omega-Lock: post-optimizer audit gate for tuned candidates},
  year    = {2026},
  version = {0.2.2},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

## License

Apache 2.0 License. See [LICENSE](https://github.com/hibou04-ops/omega-lock/blob/main/LICENSE) for details.

Copyright (c) 2026 hibou.

**License history.** PyPI distributions of versions 0.1.0 through 0.1.4 were shipped with an MIT `LICENSE` file. The repository was relicensed to Apache 2.0 on 2026-04-22; 0.1.5 and all later versions ship under Apache 2.0. Anyone who installed 0.1.0 through 0.1.4 holds an MIT license to that copy; license changes do not apply retroactively.
