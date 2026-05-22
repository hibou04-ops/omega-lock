# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "docs" / "EXAMPLES_GALLERY.md"


EXPECTED_COMMANDS = (
    "python examples/rosenbrock_demo.py",
    "python examples/phantom_demo.py",
    "python examples/benchmark_battery.py",
    "python examples/adapter_example.py",
    "python examples/demo_replay.py --check",
    "python examples/demo_sram.py --check",
    "python examples/demo_sram.py",
)

EXPECTED_CLAIM_IDS = (
    "append_only_audit_trail",
    "benchmark_scorecard",
    "deterministic_offline_demos",
    "feasible_best_vs_absolute_best",
    "hard_constraint_compliance",
    "package_naming_install",
    "sha256_hash_chain_tamper_detection",
    "stress_rank_spearman",
    "walk_forward_validation",
)


def test_examples_gallery_lists_existing_commands():
    text = GALLERY.read_text(encoding="utf-8")

    for command in EXPECTED_COMMANDS:
        assert f"`{command}`" in text
        parts = command.split()
        assert parts[0] == "python"
        script_path = ROOT / parts[1]
        assert script_path.exists(), command


def test_self_checking_examples_have_check_mode_in_source():
    for script in ("examples/demo_replay.py", "examples/demo_sram.py"):
        text = (ROOT / script).read_text(encoding="utf-8")
        assert "--check" in text
        assert f"`python {script} --check`" in GALLERY.read_text(encoding="utf-8")


def test_examples_gallery_links_claim_ids_to_ledger():
    gallery_text = GALLERY.read_text(encoding="utf-8")
    ledger_text = (ROOT / "docs" / "claims" / "public_claims.yml").read_text(encoding="utf-8")

    for claim_id in EXPECTED_CLAIM_IDS:
        assert claim_id in gallery_text
        assert f'"id": "{claim_id}"' in ledger_text
