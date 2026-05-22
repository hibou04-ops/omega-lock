# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLAIM_LEDGER = ROOT / "docs" / "claims" / "public_claims.yml"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_readme_has_conservative_badge_and_download_boundary():
    text = _normalized(README)

    assert "badge and download analytics boundaries" in text
    assert "do not prove release readiness, correctness, trustworthiness" in text
    assert "downloads or stars may indicate visibility" in text
    assert "stars/downloads must not be used as audit evidence or release approval" in text
    assert "no pypi or github download analytics are asserted here" in text


def test_badge_download_claim_is_qualitative_in_claim_ledger():
    ledger = json.loads(CLAIM_LEDGER.read_text(encoding="utf-8"))
    claims = {claim["id"]: claim for claim in ledger["claims"]}

    claim = claims["badge_download_analytics_boundaries"]
    assert claim["classification"] == "qualitative_marker"
    assert claim["status"] == "qualitative"
    assert claim["proof"] == []
    assert "Downloads or stars may indicate visibility" in claim["readme_markers"]


def test_readme_family_does_not_add_vanity_badges_or_download_metrics():
    vanity_badge_fragments = (
        "img.shields.io/pypi/d",
        "img.shields.io/github/downloads",
        "img.shields.io/github/stars",
        "pepy.tech",
    )

    for rel_path in (
        "README.md",
        "README_KR.md",
        "EASY_README.md",
        "EASY_README_KR.md",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8").lower()
        for fragment in vanity_badge_fragments:
            assert fragment not in text, rel_path


def test_docs_do_not_equate_downloads_or_stars_with_quality():
    suspicious_phrases = (
        "downloads prove",
        "downloads show quality",
        "downloads show trust",
        "download count proves",
        "stars prove",
        "stars show quality",
        "stars show trust",
        "badge proves release readiness",
        "badge proves quality",
    )

    paths = [
        README,
        ROOT / "docs" / "claims" / "README.md",
        ROOT / "docs" / "claims" / "public_claims.yml",
    ]
    for path in paths:
        text = _normalized(path)
        for phrase in suspicious_phrases:
            assert phrase not in text, path
