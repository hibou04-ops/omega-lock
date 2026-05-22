# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSITIONING = ROOT / "docs" / "TOOLKIT_POSITIONING.md"


def test_toolkit_positioning_exists_and_is_linked_from_readme():
    assert POSITIONING.exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/TOOLKIT_POSITIONING.md" in readme


def test_toolkit_positioning_states_conservative_scope():
    text = POSITIONING.read_text(encoding="utf-8")
    lower = " ".join(text.lower().split())

    assert "measurement-grade optimization audit / release gate" in lower
    assert "not a universal optimizer" in lower
    assert "not a dashboard" in lower
    assert "not a replacement for domain validation" in lower
    assert "does not claim" in lower
    assert "global optimum" in lower
    assert "tooling_missing" in lower
    assert "environment_blocked" in lower


def test_adjacent_tool_names_are_not_invented_relationships():
    text = POSITIONING.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for tool_name in ("antemortem-cli", "mini-omega-lock", "omegaprompt"):
        assert f"`{tool_name}`" in text

    assert "not referenced in the current repository docs or source" in normalized
    assert (
        "does not claim integration, compatibility, ownership, or lineage"
        in normalized
    )
    assert "No local reference found" in text


def test_positioning_claim_remains_qualitative_in_claim_ledger():
    text = POSITIONING.read_text(encoding="utf-8")
    ledger = (ROOT / "docs" / "claims" / "public_claims.yml").read_text(
        encoding="utf-8"
    )

    assert "audit_first_positioning" in text
    assert '"id": "audit_first_positioning"' in ledger
    assert '"classification": "qualitative_marker"' in ledger
    assert '"status": "qualitative"' in ledger


def test_toolkit_positioning_avoids_unsupported_competitive_or_adoption_claims():
    lower = POSITIONING.read_text(encoding="utf-8").lower()
    forbidden_phrases = (
        "market leader",
        "widely adopted",
        "downloads",
        "users trust",
        "better than",
        "beats ",
        "superior to",
        "ranked above",
        "competing tools are",
    )

    for phrase in forbidden_phrases:
        assert phrase not in lower
