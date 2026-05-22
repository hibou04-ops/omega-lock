# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_generator() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_readme_claims.py"
    spec = importlib.util.spec_from_file_location("generate_readme_claims", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_claim(claim_id: str) -> dict[str, object]:
    return {
        "id": claim_id,
        "claim": f"Fixture claim for {claim_id}.",
        "classification": "source_of_truth",
        "status": "validated",
        "readme_refs": ["README.md:1"],
        "readme_markers": [f"marker-{claim_id}"],
        "proof": [
            {
                "type": "source_of_truth",
                "path": "README.md",
                "note": "Fixture source proof.",
            }
        ],
    }


def _write_fixture(root: Path, *, claims: list[dict[str, object]] | None = None) -> None:
    markers = [f"marker-{claim_id}" for claim_id in GEN.REQUIRED_CLAIM_IDS]
    _write(root / "README.md", "\n".join(markers) + "\n")
    _write(root / "artifact.txt", "deterministic fixture\n")
    _write(root / "generated_source.md", "generated fixture\n")

    if claims is None:
        claims = [_base_claim(claim_id) for claim_id in GEN.REQUIRED_CLAIM_IDS]
        for claim in claims:
            if claim["id"] in {"deterministic_offline_demos", "stress_rank_spearman"}:
                claim["classification"] = "deterministic_artifact"
                claim["proof"] = [
                    {
                        "type": "deterministic_artifact",
                        "path": "artifact.txt",
                        "note": "Fixture deterministic artifact.",
                    }
                ]
            if claim["id"] == "benchmark_scorecard":
                claim["classification"] = "reproducible_command"
                claim["proof"] = [
                    {
                        "type": "reproducible_command",
                        "command": "python -c \"print('offline')\"",
                        "network": False,
                    }
                ]
            if claim["id"] == "no_omega_lock_diff_cli":
                claim["classification"] = "generated_doc"
                claim["proof"] = [
                    {
                        "type": "generated_doc",
                        "path": "generated_source.md",
                        "note": "Fixture generated document.",
                    }
                ]

    ledger = {
        "schema_version": 1,
        "readme": "README.md",
        "claims": claims,
    }
    _write(root / "docs" / "claims" / "public_claims.yml", json.dumps(ledger, indent=2))


def _failures(diagnostics: list[object]) -> list[str]:
    return [
        getattr(diagnostic, "message")
        for diagnostic in diagnostics
        if getattr(diagnostic, "status") == "FAIL"
    ]


def test_generated_readme_claims_valid_fixture_generates_and_checks(tmp_path: Path):
    _write_fixture(tmp_path)

    diagnostics = GEN.write_outputs(tmp_path)
    assert not _failures(diagnostics)

    first_json = (tmp_path / GEN.GENERATED_JSON).read_text(encoding="utf-8")
    first_md = (tmp_path / GEN.GENERATED_MD).read_text(encoding="utf-8")
    diagnostics = GEN.write_outputs(tmp_path)
    assert not _failures(diagnostics)
    assert (tmp_path / GEN.GENERATED_JSON).read_text(encoding="utf-8") == first_json
    assert (tmp_path / GEN.GENERATED_MD).read_text(encoding="utf-8") == first_md

    diagnostics = GEN.check_outputs(tmp_path)
    assert not _failures(diagnostics)

    payload = json.loads(first_json)
    assert payload["claim_count"] == len(GEN.REQUIRED_CLAIM_IDS)
    assert payload["classification_counts"]["source_of_truth"] >= 1
    assert payload["classification_counts"]["deterministic_artifact"] >= 1
    assert payload["classification_counts"]["generated_doc"] >= 1
    assert payload["classification_counts"]["reproducible_command"] >= 1


def test_generated_readme_claims_missing_proof_fails(tmp_path: Path):
    claims = [_base_claim(claim_id) for claim_id in GEN.REQUIRED_CLAIM_IDS]
    claims[0]["proof"] = []
    _write_fixture(tmp_path, claims=claims)

    ledger = GEN.load_ledger(tmp_path)
    diagnostics = GEN.validate_ledger(tmp_path, ledger)

    failures = _failures(diagnostics)
    assert failures
    assert any("missing proof" in failure for failure in failures)


def test_generated_readme_claims_stale_output_fails_check(tmp_path: Path):
    _write_fixture(tmp_path)
    diagnostics = GEN.write_outputs(tmp_path)
    assert not _failures(diagnostics)

    with (tmp_path / GEN.GENERATED_MD).open("a", encoding="utf-8") as f:
        f.write("\nstale edit\n")

    diagnostics = GEN.check_outputs(tmp_path)
    failures = _failures(diagnostics)
    assert any("generated file is stale" in failure for failure in failures)


def test_generated_readme_claims_allows_qualitative_todo_without_proof(tmp_path: Path):
    claims = [_base_claim(claim_id) for claim_id in GEN.REQUIRED_CLAIM_IDS]
    claims.append(
        {
            "id": "fixture_qualitative_marker",
            "claim": "Fixture qualitative marker.",
            "classification": "qualitative_marker",
            "status": "todo",
            "readme_refs": ["README.md:1"],
            "readme_markers": ["marker-walk_forward_validation"],
            "proof": [],
        }
    )
    _write_fixture(tmp_path, claims=claims)

    ledger = GEN.load_ledger(tmp_path)
    diagnostics = GEN.validate_ledger(tmp_path, ledger)

    assert not _failures(diagnostics)
