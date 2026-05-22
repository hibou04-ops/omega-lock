# Release Checklist

Use this checklist for every omega-lock release. The current target is `0.2.7`.

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

## 0.2.7 Release Note

- install command, documentation badges, and the README family synchronized to
  `0.2.7`
- README family top-section refactor: "Use it when", "Trust loop",
  verification/evidence links, and a "How is this different?" comparison near
  the top
- `docs/claims/generated_readme_claims.*` and the golden audit fixtures
  regenerated so embedded version metadata matches `0.2.7`
- no runtime behavior changes beyond version metadata, unless a tested code
  change is explicitly included

## Historical Release Notes

### 0.2.1

0.2.1 is a release sync and badge cache-bust correction:

- badge cache-bust query updated
- release metadata synchronized
- README/PyPI surface sync corrected
- no runtime behavior changes beyond version metadata

## Expected Artifacts

For 0.2.7, the expected files are:

- `omega_lock-0.2.7-py3-none-any.whl`
- `omega_lock-0.2.7.tar.gz`

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
git commit -m "Prepare release 0.2.7"
git tag v0.2.7
git push origin main
git push origin v0.2.7
```

Create GitHub Release `v0.2.7` to trigger `.github/workflows/publish.yml`, or
publish manually:

```bash
python -m twine upload dist/*
```

Do not upload if the GitHub tag, package metadata, and `dist/` filenames do not
all agree on the same version.

## Verify PyPI

After PyPI publication is expected to exist:

```bash
python scripts/post_release_verify.py --version 0.2.7 --distribution omega-lock
python -m pip index versions omega-lock
python -m pip install --no-cache-dir --upgrade omega-lock==0.2.7
python -c "import omega_lock; print(omega_lock.__version__)"
```

For cache-sensitive releases, verify the exact wheel URL exposed by PyPI JSON:

```bash
python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('https://pypi.org/pypi/omega-lock/0.2.7/json')); print([u['url'] for u in data['urls'] if u['packagetype']=='bdist_wheel'][0])"
python -m pip install --no-cache-dir --force-reinstall "<wheel-url-from-pypi-json>"
python -c "import omega_lock; print(omega_lock.__version__)"
```

`TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are blockers, not approvals. Local
version metadata is not proof of registry publication.
