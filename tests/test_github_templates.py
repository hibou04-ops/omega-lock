# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITHUB = ROOT / ".github"
ISSUE_TEMPLATE = GITHUB / "ISSUE_TEMPLATE"


TEMPLATE_PATHS = (
    GITHUB / "pull_request_template.md",
    ISSUE_TEMPLATE / "bug_report.yml",
    ISSUE_TEMPLATE / "release_blocker.yml",
    ISSUE_TEMPLATE / "claim_request.yml",
    ISSUE_TEMPLATE / "config.yml",
)

REQUIRED_TRUST_MARKERS = (
    "affected audit invariant",
    "hard constraints",
    "walk-forward",
    "schema, artifact, or hash-chain",
    "claim ledger",
    "exact verification commands",
    "live provider/api",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_github_templates_exist():
    for path in TEMPLATE_PATHS:
        assert path.exists(), path


def test_pr_template_contains_release_safety_and_trust_checklist():
    text = _text(GITHUB / "pull_request_template.md").lower()

    for marker in REQUIRED_TRUST_MARKERS:
        assert marker in text

    assert "release-safety checklist" in text
    assert "does not publish to pypi" in text
    assert "does not create or push git tags" in text
    assert "does not create github releases" in text
    assert "tooling_missing" in text
    assert "environment_blocked" in text
    assert "n/a" in text


def test_issue_templates_ask_for_required_trust_surfaces():
    for filename in ("bug_report.yml", "release_blocker.yml", "claim_request.yml"):
        text = _text(ISSUE_TEMPLATE / filename).lower()
        for marker in REQUIRED_TRUST_MARKERS:
            assert marker in text, f"{filename}: {marker}"


def test_claim_request_requires_proof_type():
    text = _text(ISSUE_TEMPLATE / "claim_request.yml")

    for proof_type in (
        "source_of_truth",
        "generated_doc",
        "reproducible_command",
        "deterministic_artifact",
        "qualitative_marker",
        "TODO / not yet proven",
    ):
        assert proof_type in text

    assert "downloads, stars, or badges as correctness proof" in text


def test_templates_do_not_include_publish_or_tag_commands():
    forbidden_command_fragments = (
        "twine upload",
        "git tag ",
        "git push --tags",
        "gh release create",
        "pypa/gh-action-pypi-publish",
    )

    for path in TEMPLATE_PATHS:
        text = _text(path).lower()
        for fragment in forbidden_command_fragments:
            assert fragment not in text, path


def test_release_blocker_treats_blocked_statuses_as_non_approval():
    text = _text(ISSUE_TEMPLATE / "release_blocker.yml")

    assert "TOOLING_MISSING or ENVIRONMENT_BLOCKED is treated as non-approval" in text
    assert "not approval to publish, tag, or create a GitHub release" in text
