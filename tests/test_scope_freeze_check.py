# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_scope_freeze() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "scope_freeze_check.py"
    spec = importlib.util.spec_from_file_location("scope_freeze_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCOPE = _load_scope_freeze()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_release_candidate_config_parses_marker_and_release_audit_ref(tmp_path: Path):
    _write(
        tmp_path / "docs" / "RELEASE_CANDIDATE.md",
        """
        # Release Candidate

        release_candidate_marker: abc123
        release_audit_after_commit: def456
        release_audit_command: python scripts/release_audit.py --intended-version 1.2.3 --offline --json
        """,
    )

    config, result = SCOPE.load_release_candidate_config(tmp_path)

    assert result.status == "PASS"
    assert config.marker == "abc123"
    assert config.release_audit_ref == "def456"
    assert config.release_audit_command.endswith("--offline --json")


def test_unset_marker_is_advisory_not_success(tmp_path: Path):
    _write(
        tmp_path / "docs" / "RELEASE_CANDIDATE.md",
        """
        release_candidate_marker: RC_MARKER_UNSET
        release_audit_after_commit: RC_AUDIT_UNSET
        """,
    )

    config, result = SCOPE.load_release_candidate_config(tmp_path)
    order = SCOPE.check_code_changes_after_generated_artifacts(
        config.marker,
        ["src/omega_lock/audit/_types.py"],
    )

    assert result.status == "WARN"
    assert config.marker is None
    assert order.status == "WARN"


def test_code_change_without_generated_artifacts_fails():
    result = SCOPE.check_code_changes_after_generated_artifacts(
        "abc123",
        ["src/omega_lock/audit/_types.py", "scripts/scope_freeze_check.py"],
    )

    assert result.status == "FAIL"
    assert "src/omega_lock/audit/_types.py" in result.details


def test_code_change_with_generated_artifacts_passes_order_check():
    result = SCOPE.check_code_changes_after_generated_artifacts(
        "abc123",
        [
            "src/omega_lock/audit/_types.py",
            "docs/claims/generated_readme_claims.md",
            "tests/fixtures/golden_audits/all_constraints_pass.json",
        ],
    )

    assert result.status == "PASS"


def test_git_unavailable_is_tooling_missing(tmp_path: Path):
    def fake_git(root: Path, args: list[str], **_kwargs):
        return SCOPE.GitResult(127, "", "git not found")

    result = SCOPE.check_git_available(tmp_path, fake_git)

    assert result.status == "TOOLING_MISSING"


def test_release_audit_unset_after_marker_fails(tmp_path: Path):
    result = SCOPE.check_release_audit_freshness(
        tmp_path,
        marker="abc123",
        release_audit_ref=None,
        changed_files=[],
        git_runner=lambda root, args, **kwargs: SCOPE.GitResult(0, "unused\n", ""),
    )

    assert result.status == "FAIL"
    assert "not recorded" in result.message


def test_release_audit_fails_when_relevant_files_changed_after_recorded_audit(
    tmp_path: Path,
):
    result = SCOPE.check_release_audit_freshness(
        tmp_path,
        marker="abc123",
        release_audit_ref="def456",
        changed_files=["README.md", "src/omega_lock/orchestrator.py"],
        git_runner=lambda root, args, **kwargs: SCOPE.GitResult(0, "unused\n", ""),
    )

    assert result.status == "FAIL"
    assert "rerun release audit" in result.message


def test_release_audit_ref_after_latest_relevant_change_passes(tmp_path: Path):
    def fake_git(root: Path, args: list[str], **_kwargs):
        if args[:3] == ["log", "-n", "1"]:
            return SCOPE.GitResult(0, "latest\n", "")
        if args[:2] == ["rev-parse", "--verify"]:
            return SCOPE.GitResult(0, "audit\n", "")
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return SCOPE.GitResult(0, "", "")
        return SCOPE.GitResult(1, "", f"unexpected args: {args}")

    result = SCOPE.check_release_audit_freshness(
        tmp_path,
        marker="marker",
        release_audit_ref="audit",
        changed_files=[],
        git_runner=fake_git,
    )

    assert result.status == "PASS"


def test_scope_freeze_script_does_not_contain_release_actions():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "scope_freeze_check.py").read_text(
        encoding="utf-8"
    ).lower()

    for forbidden in (
        "twine upload",
        "git tag ",
        "git push --tags",
        "gh release create",
        "pypa/gh-action-pypi-publish",
    ):
        assert forbidden not in text
