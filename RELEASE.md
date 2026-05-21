# Release Checklist

Use this checklist for every omega-lock release.

## Before Building

1. Update `pyproject.toml` to the intended version.
2. Update `src/omega_lock/__init__.py` `__version__`.
3. Update README release notes, citation version, and documentation badges.
4. Search for stale current-version references:

   ```bash
   rg "<previous-version>|v<previous-version>|pypi-v<previous-version>|old hardcoded test badge|old static PyPI badge"
   ```

5. Run quality gates:

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
python -m twine check dist/*
```

## 0.2.2 Release Note

0.2.2 is a badge hardening and release-surface synchronization release:

- dynamic PyPI version badge replaced with a static release badge
- Shields/PyPI/Camo stale badge rendering avoided
- current install command and citation synchronized
- no runtime behavior changes beyond version metadata

## Historical Release Notes

### 0.2.1

0.2.1 is a release sync and badge cache-bust correction:

- badge cache-bust query updated
- release metadata synchronized
- README/PyPI surface sync corrected
- no runtime behavior changes beyond version metadata

## Expected Artifacts

For 0.2.2, the expected files are:

- `omega_lock-0.2.2-py3-none-any.whl`
- `omega_lock-0.2.2.tar.gz`

If an isolated PEP 517 build cannot download build dependencies in a restricted
environment, `python -m build --no-isolation` is acceptable for local sandbox
verification only after pytest, pyright, ruff, dist filename checks, and
`twine check` all pass.

## Commit, Tag, Push, Publish

```bash
git status
git add pyproject.toml src/omega_lock/__init__.py README.md README_KR.md EASY_README.md EASY_README_KR.md RELEASE.md
git commit -m "Prepare release 0.2.2"
git tag v0.2.2
git push origin main
git push origin v0.2.2
python -m twine upload dist/*
```

## Verify PyPI

```bash
python -m pip index versions omega-lock
python -m pip install --no-cache-dir --upgrade omega-lock==0.2.2
python -c "import omega_lock; print(omega_lock.__version__)"
```

For cache-sensitive releases, verify the exact wheel URL exposed by PyPI JSON:

```bash
python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('https://pypi.org/pypi/omega-lock/0.2.2/json')); print([u['url'] for u in data['urls'] if u['packagetype']=='bdist_wheel'][0])"
python -m pip install --no-cache-dir --force-reinstall "<wheel-url-from-pypi-json>"
python -c "import omega_lock; print(omega_lock.__version__)"
```

Do not upload if the GitHub tag, package metadata, and `dist/` filenames do not
all agree on the same version.
