# Release Checklist

Use this checklist for every omega-lock release. The current target is `0.3.7`.

## Before Building

1. Update `pyproject.toml` to the intended version.
2. Update `src/omega_lock/__init__.py` `__version__`.
3. Update the README family release surfaces, documentation badges, and install
   pins.
4. Regenerate deterministic artifacts whose content embeds the version:

   ```bash
   python scripts/generate_readme_claims.py
   python scripts/run_golden_audit_cases.py --update
   ```

5. Search for stale current-version references:

   ```bash
   rg "<previous-version>|v<previous-version>"
   ```

6. Run quality gates:

   ```bash
   python -m pytest -q
   python -m pyright src tests
   python -m ruff check src tests
   ```

## Build Fresh Artifacts

PyPI does not support force-overwriting an already uploaded version. Always bump
the version and build fresh artifacts. Always delete `dist/` before building.

```powershell
python -m pip install -U build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
Get-ChildItem dist
python -m twine check dist/*.whl dist/*.tar.gz
```

## 0.3.7 Release Note

- packaging / metadata release; no library or public-API change. Shortens the
  composite Action's `action.yml` `description` to under GitHub's 125-character
  Marketplace limit so the overfit gate can be published to the GitHub
  Marketplace and consumed as `uses: hibou04-ops/omega-lock@v0.3.7` in CI. The
  action's inputs, outputs, and run steps are unchanged. The consumed-surface
  contract, docking guard, and every public symbol are untouched, so consumers
  pinning `omega-lock>=0.3.0,<0.4.0` are unaffected.

## Historical Release Notes

### 0.3.6

- packaging / distribution release; no library or public-API change. Shipped a
  composite GitHub Action (`action.yml`) at the repository root so the overfit
  gate can run as `uses: hibou04-ops/omega-lock@v0.3.6` in CI and be published
  to the GitHub Marketplace. The action installs `omega-lock` from PyPI (pinned
  to this version by default) and runs `omega-lock gate`. The consumed-surface
  contract, docking guard, and every public symbol are untouched, so consumers
  pinning `omega-lock>=0.3.0,<0.4.0` are unaffected.

### 0.3.5

- documentation / distribution release; no API change. The README family
  (EN/KR + easy variants) was rewritten to be jargon-free and
  search-optimized, with comparison tables and two new docs
  (`docs/HOW_IT_WORKS.md`, `docs/API.md`). The consumed-surface contract,
  docking guard, and every public symbol are untouched, so consumers pinning
  `omega-lock>=0.3.0,<0.4.0` are unaffected.

### 0.3.4

- additive feature release; no existing public symbol was renamed, moved, or
  re-defaulted, so consumers pinning `omega-lock>=0.3.0,<0.4.0` are unaffected.
- new installed console command `omega-lock` (`[project.scripts]`) with
  `demo`, `gate`, and `report` subcommands (argparse only; still no
  `omega-lock diff` command). The walk-forward case-study engine moved into
  the package as `omega_lock._demo` so the wheel can run it;
  `examples/walkforward_gate_demo.py` re-exports the same symbols and prints
  the identical narrative.
- new Optuna bridge API `omega_lock.integrations.optuna_bridge.
  audit_optuna_study` (re-exported at package root): gates an existing study's
  completed trials through the reused WalkForward + check_kc4 machinery,
  splits `best_any` vs `best_feasible` via per-trial `user_attrs["feasible"]`
  flags, and lazily imports optuna inside the function (optional `[p2]` extra
  unchanged).
- new stdlib-only `omega_lock.report_html.render_html`: deterministic
  single-file dark-theme HTML scorecard for `P1Result` / `AuditReport` /
  `StudyAuditReport` / `GateVerdict` (verdict banner, best_any vs
  best_feasible table, stress ranking, inline SVG train-vs-holdout scatter).
- new plain-language facade `omega_lock.simple`: `gate_scores` -> `GateVerdict`
  and `audit` -> `P1Result`. `simple.audit()` is deliberately not re-exported
  at the package root because `omega_lock.audit` is already the audit
  subpackage.
- golden audit fixtures regenerated only to carry the new version string; the
  audit report schema and SHA-256 hash chain are unchanged (the version is not
  part of the hashed payload, so every chain digest is byte-identical).

### 0.3.3

- classifier promotion only: `Development Status` advanced from `3 - Alpha` to
  `4 - Beta`. No functional code change since 0.3.2 — the dormant, default-off
  parallel-execution executor seam and the sdist packaging fix shipped in 0.3.2
  both stand unchanged.
- golden audit fixtures regenerated only to carry the new version string; the
  audit report schema and SHA-256 hash chain are unchanged.

### 0.3.2

- source-distribution packaging fix: the sdist now ships `scripts/`, `tests/`
  (including `tests/fixtures/`), and `examples/` via `MANIFEST.in`, so
  `pip download omega-lock --no-binary :all:` then `pytest --collect-only` on
  the unpacked sdist has zero collection errors. The wheel is unchanged and
  still ships only the import package (no tests/scripts/examples leak in).
- dormant, default-off parallel-execution seam: `GridSearch.run`,
  `ZoomingGridSearch.run`, `measure_stress`, and `WalkForward.run` accept an
  optional `executor: concurrent.futures.Executor | None = None` and dispatch
  via a shared `_ordered_eval_map` helper that reassembles results in INPUT
  order. The default (`None`) is serial and byte-identical; these are additive
  optional keyword arguments only, so the consumed-surface contract is
  unchanged.
- golden audit fixtures regenerated only to carry the new version string; the
  default-off seam produces zero additional golden change, and the audit report
  schema and SHA-256 hash chain are unchanged.

### 0.3.1

- internal guard and documentation hardening only; no public API or runtime
  behavior changes
- hardened the omega family docking guard surface: producer contract manifest
  (`src/omega_lock/contract.py`), the family docking convention (`DOCKING.md`),
  and a tier-aware offline presence-lint (`scripts/check_docking_presence.py`)
- `DOCKING.md` updated to record the presence-lint as the now-implemented C4
  "teeth" (previously deferred as an owner decision)
- golden audit fixtures regenerated only to carry the new version string; the
  audit report schema and hash-chain semantics are unchanged

### 0.3.0

- new `KCThresholds.pure_objective()` preset: disables the action-count gates
  (KC-3 and the KC-4 trade-ratio sub-gate) for non-action objectives while
  keeping the domain-neutral gates
- domain-neutral public field names with backward-compatible aliases
  (`EvalResult.sample_count`, `ParamSpec.stress_suppressed`,
  `StressResult.stress_suppressed`, config `exclude_suppressed_in_unlock`,
  result `top_k_excl_suppressed`); old names retained as deprecated aliases
- documentation/example wording cleanup; README family + version surfaces
  synchronized to `0.3.0`
- tamper-evident audit report schema and golden fixtures unchanged
  (hash chain preserved); all changes are backward compatible

### 0.2.1

0.2.1 is a release sync and badge cache-bust correction:

- badge cache-bust query updated
- release metadata synchronized
- README/PyPI surface sync corrected
- no runtime behavior changes beyond version metadata

## Expected Artifacts

For 0.3.7, the expected files are:

- `omega_lock-0.3.7-py3-none-any.whl`
- `omega_lock-0.3.7.tar.gz`

If an isolated PEP 517 build cannot download build dependencies in a restricted
environment, `python -m build --no-isolation` is acceptable for local sandbox
verification only after pytest, pyright, ruff, dist filename checks, and
`twine check` all pass.

## Commit, Tag, Push, Publish

Preferred path: GitHub Actions Trusted Publishing. `.github/workflows/publish.yml`
builds, checks, runs the publish-readiness gate, and publishes to PyPI when a
GitHub Release is published. Manual `twine upload` remains the fallback.

Review `git status` first, then:

```bash
git add -A
git commit -m "Prepare release 0.3.7"
git tag v0.3.7
git push origin main
git push origin v0.3.7
```

Create GitHub Release `v0.3.7` to trigger `.github/workflows/publish.yml`, or
publish manually:

```bash
python -m twine upload dist/*
```

Do not upload if the GitHub tag, package metadata, and `dist/` filenames do not
all agree on the same version.

## Verify PyPI

After PyPI publication is expected to exist:

```bash
python scripts/post_release_verify.py --version 0.3.7 --distribution omega-lock
python -m pip index versions omega-lock
python -m pip install --no-cache-dir --upgrade omega-lock==0.3.7
python -c "import omega_lock; print(omega_lock.__version__)"
```

For cache-sensitive releases, verify the exact wheel URL exposed by PyPI JSON:

```bash
python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('https://pypi.org/pypi/omega-lock/0.3.7/json')); print([u['url'] for u in data['urls'] if u['packagetype']=='bdist_wheel'][0])"
python -m pip install --no-cache-dir --force-reinstall "<wheel-url-from-pypi-json>"
python -c "import omega_lock; print(omega_lock.__version__)"
```

`TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are blockers, not approvals. Local
version metadata is not proof of registry publication.
