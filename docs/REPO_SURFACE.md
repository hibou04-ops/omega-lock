# Repository Surface Baseline

Generated from local files on 2026-05-22. This is an inspection document only;
it does not approve a release, create a tag, publish to PyPI, or change runtime
behavior.

## Naming Matrix

| Surface | Current value | Verification status |
| --- | --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` | Local `origin` remote is `https://github.com/hibou04-ops/omega-lock.git`; remote GitHub availability is `ENVIRONMENT_BLOCKED`. |
| PyPI distribution | `omega-lock` | Local `pyproject.toml` project name is `omega-lock`; PyPI registry status is `ENVIRONMENT_BLOCKED`. |
| Python import package | `omega_lock` | Local package exists under `src/omega_lock`. |
| CLI executable status | None currently. | No `[project.scripts]`, `[project.gui-scripts]`, `[project.entry-points]`, `console_scripts`, or `gui_scripts` entries were found locally. |
| Source of current package version | `pyproject.toml` `[project].version = "0.2.6"`; runtime mirror `src/omega_lock/__init__.py` `__version__ = "0.2.6"`. | Verified from local files. |
| Release/tag/PyPI verification status | Local tags are listed below; GitHub releases and PyPI publication status are `ENVIRONMENT_BLOCKED`. | No network registry or GitHub release check was performed. |

## Package Metadata

| Field | Local value |
| --- | --- |
| `project.name` | `omega-lock` |
| `project.version` | `0.2.6` |
| `requires-python` | `>=3.11` |
| Runtime dependencies | `numpy>=1.24` |
| Optional dependency group `dev` | `pytest>=7.0`, `pytest-cov>=4.0`, `pyright>=1.1.0`, `ruff>=0.4.0` |
| Optional dependency group `p2` | `optuna>=3.0` |

## Version Mentions

| File | Local version surface |
| --- | --- |
| `src/omega_lock/__init__.py` | `__version__ = "0.2.6"` |
| `README.md` | Current local package version badge and conditional install examples use `0.2.6`; PyPI publication status is explicitly not asserted. |
| `README_KR.md` | Current local package version badge and conditional install examples use `0.2.6`; PyPI publication status is explicitly not asserted. |
| `EASY_README.md` | Current local package version badge and conditional install examples use `0.2.6`; PyPI publication status is explicitly not asserted. |
| `EASY_README_KR.md` | Current local package version badge and conditional install examples use `0.2.6`; PyPI publication status is explicitly not asserted. |
| `RELEASE.md` | Release checklist uses `<version>` placeholders for current release commands and retains historical notes where explicitly labeled. |

## Repository Structure

### `tests/`

Local tracked-style surface contains 32 Python test files and one fixture file:

- `tests/test_adapters.py`
- `tests/test_artifact_reproducibility.py`
- `tests/test_audit.py`
- `tests/test_audit_hash_chain.py`
- `tests/test_auto_phase_tracking.py`
- `tests/test_benchmark.py`
- `tests/test_benchmark_regression.py`
- `tests/test_config_validation.py`
- `tests/test_constraint_aware_selection.py`
- `tests/test_constraint_uniqueness.py`
- `tests/test_demo_sram.py`
- `tests/test_fitness.py`
- `tests/test_generated_readme_claims.py`
- `tests/test_grid.py`
- `tests/test_holdout.py`
- `tests/test_holdout_gate_mode.py`
- `tests/test_integration_phantom.py`
- `tests/test_integration_phantom_deep.py`
- `tests/test_integration_rosenbrock.py`
- `tests/test_iterative.py`
- `tests/test_iterative_advisory_and_gap.py`
- `tests/test_kc2_stress_edge_cases.py`
- `tests/test_kc4_pearson_status.py`
- `tests/test_kill_criteria.py`
- `tests/test_p2_tpe.py`
- `tests/test_random_search.py`
- `tests/test_repo_consistency_checker.py`
- `tests/test_sc2_advisory.py`
- `tests/test_sc2_negative_fitness.py`
- `tests/test_stress.py`
- `tests/test_walk_forward_cache.py`
- `tests/test_zooming.py`
- `tests/fixtures/benchmark_gold.json`

### `examples/`

- `examples/adapter_example.py`
- `examples/benchmark_battery.py`
- `examples/demo_replay.py`
- `examples/demo_sram.py`
- `examples/full_showcase.py`
- `examples/phantom_demo.py`
- `examples/rosenbrock_demo.py`
- `examples/_demo_output.txt`
- `examples/omega_lock_demos/__init__.py`
- `examples/omega_lock_demos/sram.py`

### `docs/`

- `docs/demo/omega-lock-demo.en.srt`
- `docs/REPO_SURFACE.md`
- `docs/claims/README.md`
- `docs/claims/generated_readme_claims.json`
- `docs/claims/generated_readme_claims.md`
- `docs/claims/public_claims.yml`

### `.github/workflows/`

- `.github/workflows/encoding-check.yml`: push to `main` and pull request encoding guard using Python 3.11 and `scripts/check_encoding.py`.
- `.github/workflows/quality-ci.yml`: push to `main` and pull request release-safe quality CI using Python 3.11; installs `.[dev]`, runs the encoding guard, repository consistency check, generated README claims check, golden audit case check, deterministic demo self-checks, pytest, pyright, and ruff. It does not publish packages or create releases.
- `.github/workflows/release-readiness.yml`: manual workflow_dispatch non-publishing release-readiness workflow; runs offline release audit, local wheel smoke install, and the publish-readiness gate. It does not publish packages, create tags, or create releases.
- `.github/workflows/publish.yml`: release-published and manual workflow_dispatch build/check/publish workflow. Presence of this workflow is not publication approval.

### `scripts/`

- `scripts/check_encoding.py`
- `scripts/check_repo_consistency.py`
- `scripts/generate_readme_claims.py`

## Changelog

`CHANGELOG.md` is present and records local repository release notes without
asserting registry publication status.

## Verification Notes

| Command or check | Status | Result |
| --- | --- | --- |
| `python` on PATH | `TOOLING_MISSING` | PowerShell could not resolve `python`. |
| `.venv/Scripts/python.exe` | `TOOLING_MISSING` | File exists, but execution failed because its base interpreter could not be created. |
| Bundled Python `pyproject.toml` read | Verified | Printed `omega-lock 0.2.6`. |
| Bundled Python source import with `PYTHONPATH=src` | Verified | Printed `0.2.6`. |
| `python -m pytest -q` | `TOOLING_MISSING` | PATH `python` is missing; bundled Python also lacks `pytest`. |

## Local Git Surface

Default `git` commands were blocked by the sandbox user/repository ownership
check. Read-only git inspection was completed with command-scoped
`safe.directory`; no global git configuration was changed.

### Remote

- `origin`: `https://github.com/hibou04-ops/omega-lock.git` for fetch and push.

### Local Tags

- `0.1.4`
- `pre-employment-ip-snapshot-2026-04-28`
- `v0.1.6`
- `v0.1.7`
- `v0.1.8`
- `v0.1.9`
- `v0.2.0`
- `v0.2.1`
- `v0.2.2`
- `v0.2.3`
- `v0.2.4`

No local `v0.2.6` tag was observed during the retarget. Existing `v0.2.4`
and any `v0.2.5` release/tag/artifact state are historical state and are not
release approval for `0.2.6`.
Remote tag state is `ENVIRONMENT_BLOCKED`.
