# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_release_audit() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "release_audit.py"
    spec = importlib.util.spec_from_file_location("release_audit", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_release_audit()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_fixture_repo(
    root: Path,
    *,
    pyproject_version: str = "1.2.3",
    init_version: str = "1.2.3",
    doc_version: str = "1.2.3",
) -> None:
    _write(
        root / "pyproject.toml",
        f"""
        [project]
        name = "omega-lock"
        version = "{pyproject_version}"
        dependencies = ["numpy>=1.24"]
        """,
    )
    _write(root / "src" / "omega_lock" / "__init__.py", f'__version__ = "{init_version}"')

    readme = f"""
        # Omega-Lock

        [![Release](https://img.shields.io/badge/release-{doc_version}-orange.svg)](https://pypi.org/project/omega-lock/{doc_version}/)

        ```bash
        pip install omega-lock=={doc_version}
        ```

        There is no installed console `omega-lock diff` command.

        ```python
        import omega_lock
        ```
        """
    for name in ("README.md", "README_KR.md", "EASY_README.md", "EASY_README_KR.md"):
        _write(root / name, readme)

    _write(
        root / "RELEASE.md",
        f"""
        # Release Checklist

        ## {doc_version} Release Note

        For {doc_version}, the expected files are:

        - `omega_lock-{doc_version}-py3-none-any.whl`
        - `omega_lock-{doc_version}.tar.gz`

        ```bash
        git tag v{doc_version}
        git push origin v{doc_version}
        python -c "import json, urllib.request; json.load(urllib.request.urlopen('https://pypi.org/pypi/omega-lock/{doc_version}/json'))"
        ```
        """,
    )


def _patch_environment_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        AUDIT,
        "check_generated_claims",
        lambda root: AUDIT.AuditResult("generated-claims", "PASS", "claims current"),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_golden_artifacts",
        lambda root: AUDIT.AuditResult("golden-audit-artifacts", "PASS", "goldens current"),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_dist_artifacts",
        lambda root, version: AUDIT.AuditResult("dist-artifacts", "WARN", "dist absent"),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_git_tag_status",
        lambda root, version: AUDIT.AuditResult("git-tags", "WARN", "git tags unchecked"),
    )


def _by_name(results: list[object]) -> dict[str, object]:
    return {getattr(result, "name"): result for result in results}


def test_release_audit_current_repo_offline_json_is_stable():
    root = Path(__file__).resolve().parents[1]

    results = AUDIT.run_release_audit(root, intended_version="0.2.5", offline=True)
    payload = AUDIT.to_payload(results, root=root, intended_version="0.2.5", offline=True)
    rendered_once = AUDIT.render_json(payload)
    rendered_twice = AUDIT.render_json(payload)

    assert rendered_once == rendered_twice
    parsed = json.loads(rendered_once)
    assert parsed["schema_version"] == 1
    assert parsed["mode"] == "offline"
    assert not AUDIT.has_blocking_status(results, strict=False)


def test_release_audit_fails_on_pyproject_version_drift(tmp_path: Path, monkeypatch):
    _write_fixture_repo(tmp_path, pyproject_version="1.2.2", init_version="1.2.3")
    _patch_environment_checks(monkeypatch)

    results = AUDIT.run_release_audit(tmp_path, intended_version="1.2.3", offline=True)
    by_name = _by_name(results)

    assert getattr(by_name["pyproject-version"], "status") == "FAIL"
    assert AUDIT.has_blocking_status(results, strict=False)


def test_release_audit_fails_on_init_version_drift(tmp_path: Path, monkeypatch):
    _write_fixture_repo(tmp_path, pyproject_version="1.2.3", init_version="1.2.2")
    _patch_environment_checks(monkeypatch)

    results = AUDIT.run_release_audit(tmp_path, intended_version="1.2.3", offline=True)
    by_name = _by_name(results)

    assert getattr(by_name["init-version"], "status") == "FAIL"
    assert getattr(by_name["version-match"], "status") == "FAIL"


def test_release_audit_fails_on_readme_and_release_doc_version_drift(
    tmp_path: Path, monkeypatch
):
    _write_fixture_repo(tmp_path, doc_version="1.2.2")
    _patch_environment_checks(monkeypatch)

    results = AUDIT.run_release_audit(tmp_path, intended_version="1.2.3", offline=True)
    by_name = _by_name(results)

    assert getattr(by_name["readme-family-versions"], "status") == "FAIL"
    assert getattr(by_name["release-doc-versions"], "status") == "FAIL"


def test_release_audit_reports_generated_claim_drift(tmp_path: Path, monkeypatch):
    _write_fixture_repo(tmp_path)
    monkeypatch.setattr(
        AUDIT,
        "check_generated_claims",
        lambda root: AUDIT.AuditResult(
            "generated-claims",
            "FAIL",
            "Generated README claim files are stale or invalid.",
            ("generated file is stale: docs/claims/generated_readme_claims.md",),
        ),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_golden_artifacts",
        lambda root: AUDIT.AuditResult("golden-audit-artifacts", "PASS", "goldens current"),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_dist_artifacts",
        lambda root, version: AUDIT.AuditResult("dist-artifacts", "WARN", "dist absent"),
    )
    monkeypatch.setattr(
        AUDIT,
        "check_git_tag_status",
        lambda root, version: AUDIT.AuditResult("git-tags", "WARN", "git tags unchecked"),
    )

    results = AUDIT.run_release_audit(tmp_path, intended_version="1.2.3", offline=True)
    by_name = _by_name(results)

    assert getattr(by_name["generated-claims"], "status") == "FAIL"
    assert "stale" in getattr(by_name["generated-claims"], "details")[0]


def test_release_audit_fails_on_stale_dist_artifact_version(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "omega_lock-1.2.2-py3-none-any.whl").write_text("not a real wheel", encoding="utf-8")

    result = AUDIT.check_dist_artifacts(tmp_path, "1.2.3")

    assert result.status == "FAIL"
    assert "omega_lock-1.2.2-py3-none-any.whl" in result.details


def test_release_audit_ignores_release_draft_markdown_in_dist(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "omega_lock-1.2.3-py3-none-any.whl").write_text("not a real wheel", encoding="utf-8")
    (dist / "omega_lock-1.2.3.tar.gz").write_text("not a real sdist", encoding="utf-8")
    (dist / "release_draft_v1.2.3.md").write_text("draft", encoding="utf-8")

    result = AUDIT.check_dist_artifacts(tmp_path, "1.2.3")

    assert result.status == "PASS"
    assert not any("release_draft" in detail for detail in result.details)


def test_release_audit_offline_network_checks_are_warn_not_pass():
    pypi = AUDIT.check_pypi_status("0.2.5", offline=True)
    github = AUDIT.check_github_status("0.2.5", offline=True)

    assert pypi.status == "WARN"
    assert github.status == "WARN"
    assert "not release approval" in pypi.message
    assert "not release approval" in github.message


def test_release_audit_strict_blocks_environment_blocked():
    results = [AUDIT.AuditResult("pypi-status", "ENVIRONMENT_BLOCKED", "network blocked")]

    assert not AUDIT.has_blocking_status(results, strict=False)
    assert AUDIT.has_blocking_status(results, strict=True)
