# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRUST_MODEL = ROOT / "docs" / "TRUST_MODEL.md"


REQUIRED_CLAIM_IDS = (
    "append_only_audit_trail",
    "deterministic_offline_demos",
    "feasible_best_vs_absolute_best",
    "hard_constraint_compliance",
    "sha256_hash_chain_tamper_detection",
    "walk_forward_validation",
)

REQUIRED_EVIDENCE_PATHS = (
    "tests/test_audit_hash_chain.py",
    "tests/test_artifact_reproducibility.py",
    "tests/test_constraint_aware_selection.py",
    "tests/test_golden_audit_cases.py",
    "tests/test_holdout_gate_mode.py",
    "tests/test_kill_criteria.py",
    "tests/test_walk_forward_cache.py",
    "docs/claims/public_claims.yml",
    "scripts/post_release_verify.py",
    "scripts/publish_readiness.py",
)


def test_trust_model_exists_and_is_linked_from_readme():
    assert TRUST_MODEL.exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/TRUST_MODEL.md" in readme


def test_trust_model_links_strong_guarantees_to_existing_evidence():
    text = TRUST_MODEL.read_text(encoding="utf-8")
    ledger = (ROOT / "docs" / "claims" / "public_claims.yml").read_text(encoding="utf-8")

    for claim_id in REQUIRED_CLAIM_IDS:
        assert claim_id in text
        assert f'"id": "{claim_id}"' in ledger

    for rel_path in REQUIRED_EVIDENCE_PATHS:
        assert rel_path in text
        assert (ROOT / rel_path).exists(), rel_path


def test_trust_model_uses_conservative_security_language():
    text = TRUST_MODEL.read_text(encoding="utf-8").lower()

    assert "tamper-proof" not in text
    assert "does not guarantee future production performance" in text
    assert "formal verification" in text
    assert "does not imply formal verification" in text
    assert "not a signature" in text
    assert "not cryptographic immutability" in text
    assert "tooling_missing" in text
    assert "environment_blocked" in text
