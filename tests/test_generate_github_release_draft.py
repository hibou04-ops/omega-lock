# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType


def _load_draft_generator() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_github_release_draft.py"
    spec = importlib.util.spec_from_file_location("generate_github_release_draft", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DRAFT = _load_draft_generator()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path, *, stale_claims: bool = False, release_audit: bool = True) -> None:
    _write(
        root / "scripts" / "generate_readme_claims.py",
        f"""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Diagnostic:
            status: str
            message: str

        def check_outputs(root, ledger_path=None):
            return {[{"status": "FAIL", "message": "generated file is stale: docs/claims/generated_readme_claims.md"}] if stale_claims else []}
        """,
    )
    if stale_claims:
        generator = (root / "scripts" / "generate_readme_claims.py").read_text(encoding="utf-8")
        generator = generator.replace(
            "return [{'status': 'FAIL', 'message': 'generated file is stale: docs/claims/generated_readme_claims.md'}]",
            "return [Diagnostic('FAIL', 'generated file is stale: docs/claims/generated_readme_claims.md')]",
        )
        (root / "scripts" / "generate_readme_claims.py").write_text(generator, encoding="utf-8")

    _write(
        root / "RELEASE.md",
        """
        # Release Checklist

        ## Current Release Note Template

        - current install command, documentation badges, and citation synchronized
        - no runtime behavior changes beyond version metadata, unless a tested code change is explicitly included

        ## Historical Release Notes

        ### 1.2.3

        1.2.3 is a local fixture release note.

        - fixture release note item
        """,
    )
    claims = [
        {
            "id": "append_only_audit_trail",
            "claim": "README.md says append-only audit trails exist.",
            "classification": "source_of_truth",
            "status": "validated",
            "readme_markers": ["append-only JSON trail"],
            "proof": [
                {"type": "source_of_truth", "path": "src/omega_lock/audit/_types.py"},
                {
                    "type": "reproducible_command",
                    "command": "python -m pytest tests/test_audit.py -q",
                    "network": False,
                },
            ],
        },
        {
            "id": "badge_download_analytics_boundaries",
            "claim": "README.md says badges and downloads are not proof.",
            "classification": "qualitative_marker",
            "status": "qualitative",
            "readme_markers": ["Badge and download analytics boundaries"],
            "proof": [],
        },
    ]
    ledger = {"schema_version": 1, "readme": "README.md", "claims": claims}
    payload = {
        "schema_version": 1,
        "readme": "README.md",
        "claim_count": len(claims),
        "classification_counts": {"source_of_truth": 1, "qualitative_marker": 1},
        "status_counts": {"validated": 1, "qualitative": 1},
        "claims": claims,
    }
    _write_json(root / "docs" / "claims" / "public_claims.yml", ledger)
    _write_json(root / "docs" / "claims" / "generated_readme_claims.json", payload)
    _write(root / "docs" / "claims" / "generated_readme_claims.md", "# generated claims")
    _write_json(root / "tests" / "fixtures" / "golden_audits" / "fixture.json", {"case_id": "fixture"})
    _write(root / "dist" / "omega_lock-1.2.3-py3-none-any.whl", "not a real wheel")

    if release_audit:
        _write_json(
            root / "dist" / "release_audit_v1.2.3.json",
            {
                "schema_version": 1,
                "mode": "offline",
                "summary": {"PASS": 3, "WARN": 1, "FAIL": 0},
                "results": [
                    {"name": "pyproject-version", "status": "PASS", "message": "ok", "details": []},
                    {
                        "name": "pypi-status",
                        "status": "WARN",
                        "message": "Offline mode: PyPI status was not checked and is not release approval.",
                        "details": [],
                    },
                ],
            },
        )


def test_release_draft_generation_is_deterministic(tmp_path: Path):
    _write_fixture_repo(tmp_path)

    first = DRAFT.generate_release_draft(tmp_path, version="1.2.3")
    first_text = first.output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    second = DRAFT.generate_release_draft(tmp_path, version="1.2.3")
    second_text = second.output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

    assert first.status == "PASS"
    assert second.status == "PASS"
    assert first.output_path == tmp_path / "release_drafts" / "release_draft_v1.2.3.md"
    assert first_text == second_text
    assert "# omega-lock v1.2.3 GitHub Release Draft" in first_text
    assert "## Summary" in first_text
    assert "## Verified Changes" in first_text
    assert "## Audit Artifacts" in first_text
    assert "## Deterministic Commands" in first_text
    assert "## Known Limitations" in first_text
    assert "## Post-Release Verification Command" in first_text
    assert "python scripts/post_release_verify.py --version 1.2.3 --distribution omega-lock --json" in first_text
    assert "This script performs none of them" in first_text
    assert "`append_only_audit_trail`" in first_text


def test_release_draft_fails_when_generated_claims_are_stale(tmp_path: Path):
    _write_fixture_repo(tmp_path, stale_claims=True)

    result = DRAFT.generate_release_draft(tmp_path, version="1.2.3")

    assert result.status == "FAIL"
    assert "stale" in result.details[0]
    assert not (tmp_path / "dist" / "release_draft_v1.2.3.md").exists()
    assert not (tmp_path / "release_drafts" / "release_draft_v1.2.3.md").exists()


def test_release_draft_records_missing_release_audit_as_limitation(tmp_path: Path):
    _write_fixture_repo(tmp_path, release_audit=False)

    result = DRAFT.generate_release_draft(tmp_path, version="1.2.3")
    text = result.output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

    assert result.status == "PASS"
    assert "No release audit JSON was found" in text
    assert "This draft is offline documentation, not release approval" in text


def test_release_draft_rejects_invalid_version(tmp_path: Path):
    _write_fixture_repo(tmp_path)

    result = DRAFT.generate_release_draft(tmp_path, version="1.2")

    assert result.status == "FAIL"
    assert "--version must use X.Y.Z" in result.message


def test_release_draft_generator_does_not_call_github_or_release_commands():
    script_text = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_github_release_draft.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "gh release create",
        "api.github.com",
        "twine upload",
        "git tag ",
        "git push --tags",
        "pypa/gh-action-pypi-publish",
    )
    for fragment in forbidden:
        assert fragment not in script_text


def test_rendered_draft_does_not_contain_unsupported_release_claims(tmp_path: Path):
    _write_fixture_repo(tmp_path)

    result = DRAFT.generate_release_draft(tmp_path, version="1.2.3")
    text = result.output_path.read_text(encoding="utf-8").lower()  # type: ignore[union-attr]

    assert "not release approval" in text
    assert "downloads, stars, and badges are not correctness" in text
    assert "published to pypi" not in text
    assert "release ready" not in text
    assert "trusted by" not in text
