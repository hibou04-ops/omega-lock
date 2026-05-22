# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_checker() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_consistency.py"
    spec = importlib.util.spec_from_file_location("check_repo_consistency", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_fixture_repo(
    root: Path,
    *,
    version: str = "1.2.3",
    init_version: str | None = None,
    scripts: dict[str, str] | None = None,
) -> None:
    scripts_block = ""
    if scripts:
        entries = "\n".join(f'{name} = "{target}"' for name, target in scripts.items())
        scripts_block = f"\n[project.scripts]\n{entries}\n"

    _write(
        root / "pyproject.toml",
        f"""
        [project]
        name = "omega-lock"
        version = "{version}"
        dependencies = ["numpy>=1.24"]
        {scripts_block}
        """,
    )
    _write(root / "src" / "omega_lock" / "__init__.py", f'__version__ = "{init_version or version}"')

    readme = f"""
        # Omega-Lock

        [![Release](https://img.shields.io/badge/release-{version}-orange.svg)](https://pypi.org/project/omega-lock/{version}/)
        [![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

        ```bash
        pip install omega-lock=={version}
        ```

        Omega-Lock emits JSON artifacts; it does not currently ship a console `omega-lock diff` command.

        ```python
        from omega_lock import P1Config
        ```
        """
    _write(root / "README.md", readme)
    _write(root / "README_KR.md", readme)

    _write(
        root / "EASY_README.md",
        f"""
        # Omega-Lock Easy Start

        [![Release](https://img.shields.io/badge/release-{version}-orange.svg)](https://pypi.org/project/omega-lock/{version}/)

        ```bash
        pip install omega-lock=={version}
        pip install "omega-lock[p2]=={version}"
        ```

        ## What Changed in {version}
        """,
    )
    _write(
        root / "EASY_README_KR.md",
        f"""
        # Omega-Lock Easy Start

        [![Release](https://img.shields.io/badge/release-{version}-orange.svg)](https://pypi.org/project/omega-lock/{version}/)

        ```bash
        pip install omega-lock=={version}
        pip install "omega-lock[p2]=={version}"
        ```

        ## {version}에서 바뀐 점
        """,
    )
    _write(
        root / "RELEASE.md",
        f"""
        # Release Checklist

        ## {version} Release Note

        For {version}, the expected files are:

        - `omega_lock-{version}-py3-none-any.whl`
        - `omega_lock-{version}.tar.gz`

        ```bash
        git commit -m "Prepare release {version}"
        git tag v{version}
        git push origin v{version}
        python -m pip install --no-cache-dir --upgrade omega-lock=={version}
        python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('https://pypi.org/pypi/omega-lock/{version}/json'))"
        ```
        """,
    )


def _results_by_name(results: list[object]) -> dict[str, object]:
    return {getattr(result, "name"): result for result in results}


def _details(result: object) -> str:
    return "\n".join(getattr(result, "details"))


def test_repo_consistency_checker_passes_clean_fixture(tmp_path: Path):
    _write_fixture_repo(tmp_path)

    results = CHECKER.run_checks(tmp_path)
    by_name = _results_by_name(results)

    assert not CHECKER.has_blocking_status(results, strict=False)
    assert getattr(by_name["project-name"], "status") == "PASS"
    assert getattr(by_name["version-match"], "status") == "PASS"
    assert getattr(by_name["cli-documentation"], "status") == "PASS"
    assert getattr(by_name["changelog"], "status") == "WARN"


def test_repo_consistency_checker_catches_stale_doc_versions(tmp_path: Path):
    _write_fixture_repo(tmp_path, version="1.2.3")
    stale = "1.2.2"
    _write(
        tmp_path / "README_KR.md",
        f"""
        # Omega-Lock
        [![Release](https://img.shields.io/badge/release-{stale}-orange.svg)](https://pypi.org/project/omega-lock/{stale}/)
        ```bash
        pip install omega-lock=={stale}
        ```
        """,
    )
    _write(
        tmp_path / "EASY_README.md",
        f"""
        # Omega-Lock Easy Start
        [![Release](https://img.shields.io/badge/release-1.2.3-orange.svg)](https://pypi.org/project/omega-lock/1.2.3/)
        ```bash
        pip install omega-lock==1.2.3
        ```
        ## What Changed in {stale}
        """,
    )
    _write(
        tmp_path / "RELEASE.md",
        f"""
        # Release Checklist
        For {stale}, the expected files are:
        - `omega_lock-{stale}-py3-none-any.whl`
        - `omega_lock-{stale}.tar.gz`
        """,
    )

    results = CHECKER.run_checks(tmp_path)
    current_versions = _results_by_name(results)["current-version-surfaces"]

    assert getattr(current_versions, "status") == "FAIL"
    details = _details(current_versions)
    assert "README_KR.md" in details
    assert "EASY_README.md" in details
    assert "RELEASE.md" in details


def test_repo_consistency_checker_catches_nonexistent_cli_docs(tmp_path: Path):
    _write_fixture_repo(tmp_path)
    with (tmp_path / "README.md").open("a", encoding="utf-8") as f:
        f.write("\nRun `omega-lock audit` after installing the package.\n")

    results = CHECKER.run_checks(tmp_path)
    cli = _results_by_name(results)["cli-documentation"]

    assert getattr(cli, "status") == "FAIL"
    assert "omega-lock audit" in _details(cli)


def test_repo_consistency_checker_allows_project_scripts_cli_docs(tmp_path: Path):
    _write_fixture_repo(
        tmp_path,
        scripts={"omega-lock": "omega_lock.cli:main"},
    )
    with (tmp_path / "README.md").open("a", encoding="utf-8") as f:
        f.write("\nRun `omega-lock audit` after installing the package.\n")

    results = CHECKER.run_checks(tmp_path)
    cli = _results_by_name(results)["cli-documentation"]

    assert getattr(cli, "status") == "PASS"


def test_repo_consistency_checker_catches_init_version_mismatch(tmp_path: Path):
    _write_fixture_repo(tmp_path, version="1.2.3", init_version="1.2.4")

    results = CHECKER.run_checks(tmp_path)
    version_match = _results_by_name(results)["version-match"]

    assert getattr(version_match, "status") == "FAIL"
    assert "1.2.3" in _details(version_match)
    assert "1.2.4" in _details(version_match)


def test_strict_mode_blocks_tooling_missing_status():
    result = CHECKER.CheckResult("example", "TOOLING_MISSING", "missing tool")

    assert not CHECKER.has_blocking_status([result], strict=False)
    assert CHECKER.has_blocking_status([result], strict=True)
