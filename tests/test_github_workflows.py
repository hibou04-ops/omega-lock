# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_quality_ci_runs_all_default_offline_gates():
    text = _text("quality-ci.yml")

    assert "pull_request:" in text
    assert "branches: [main]" in text
    for command in (
        "python scripts/check_encoding.py",
        "python scripts/check_repo_consistency.py --check",
        "python scripts/generate_readme_claims.py --check",
        "python scripts/run_golden_audit_cases.py --check",
        "python examples/demo_replay.py --check",
        "python examples/demo_sram.py --check",
        "python -m pytest -q",
        "python -m pyright src tests",
        "python -m ruff check src tests",
    ):
        assert command in text


def test_quality_ci_is_release_safe_and_has_no_publish_actions():
    text = _text("quality-ci.yml").lower()

    forbidden = (
        "pypa/gh-action-pypi-publish",
        "twine upload",
        "gh release create",
        "git tag ",
        "git push --tags",
        "secrets.",
        "--network",
    )
    for fragment in forbidden:
        assert fragment not in text


def test_release_readiness_workflow_is_manual_and_non_publishing():
    text = _text("release-readiness.yml")
    lower = text.lower()

    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "--offline" in text
    assert "scripts/release_audit.py" in text
    assert "scripts/wheel_smoke_install.py" in text
    assert "scripts/publish_readiness.py" in text
    assert "python -m pip install build twine" in text

    forbidden = (
        "pypa/gh-action-pypi-publish",
        "twine upload",
        "gh release create",
        "git tag ",
        "git push --tags",
        "secrets.",
        "--network",
    )
    for fragment in forbidden:
        assert fragment not in lower


def test_publish_workflow_trigger_and_publish_action_remain_explicit():
    text = _text("publish.yml")

    assert "release:" in text
    assert "types: [published]" in text
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
