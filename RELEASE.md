# Release Checklist

Use this checklist for every omega-lock release.

## Before Building

1. Update `pyproject.toml` to the intended version.
2. Update `src/omega_lock/__init__.py` `__version__`.
3. Update README release notes and citation version.
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
the version and build fresh artifacts.

```powershell
python -m pip install -U build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

Verify the filenames match the intended version:

```powershell
Get-ChildItem dist
```

For 0.1.9, the expected files are:

- `omega_lock-0.1.9-py3-none-any.whl`
- `omega_lock-0.1.9.tar.gz`

## Commit, Tag, Push, Publish

```bash
git status
git add pyproject.toml src/omega_lock/__init__.py README.md README_KR.md EASY_README.md EASY_README_KR.md RELEASE.md
git commit -m "Prepare release 0.1.9"
git tag v0.1.9
git push origin main
git push origin v0.1.9
python -m twine upload dist/*
```

## Verify PyPI

```bash
python -m pip index versions omega-lock
python -m pip install --no-cache-dir --upgrade omega-lock==0.1.9
```

Do not upload if the GitHub tag, package metadata, and `dist/` filenames do not
all agree on the same version.
