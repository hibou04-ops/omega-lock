"""Tests for deterministic golden audit fixtures."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest

from omega_lock.audit._types import AuditReport


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_golden_audit_cases.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_golden_audit_cases", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_golden_audit_fixtures_match_generated_cases():
    runner = _load_runner()

    diagnostics = runner.check_golden_cases()

    assert not runner.has_failures(diagnostics), runner.format_diagnostics(diagnostics)


def test_golden_check_detects_semantic_drift(tmp_path: Path):
    runner = _load_runner()
    fixture_dir = tmp_path / "golden_audits"
    shutil.copytree(runner.FIXTURE_DIR, fixture_dir)
    path = fixture_dir / "all_constraints_pass.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["semantic"]["summary"]["n_total"] = 999
    path.write_text(runner.canonical_json(payload), encoding="utf-8")

    diagnostics = runner.check_golden_cases(fixture_dir)

    assert runner.has_failures(diagnostics)
    assert any(
        diagnostic.name == "all_constraints_pass"
        and diagnostic.status == "FAIL"
        and "drift" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_hash_chain_fixture_detects_tampering():
    runner = _load_runner()
    payload = runner.load_golden_case("append_only_hash_chain")
    signed_report = deepcopy(payload["signed_report"])

    assert runner.verify_signed_report(signed_report) is True

    signed_report["runs"][1]["params"]["x"] = 99.0

    assert runner.verify_signed_report(signed_report) is False


def test_schema_validation_fixture_rejects_mismatch():
    runner = _load_runner()
    payload = runner.load_golden_case("schema_validation_roundtrip")
    signed_report = deepcopy(payload["signed_report"])
    signed_report["schema_version"] = "omega-lock.audit-report.v999"

    with pytest.raises(ValueError, match="schema_version"):
        AuditReport.from_json(json.dumps(signed_report, sort_keys=True))


def test_append_only_prefix_is_preserved_in_fixture():
    runner = _load_runner()
    payload = runner.load_golden_case("append_only_hash_chain")
    semantic = payload["semantic"]

    assert semantic["append_only_prefix_preserved"] is True
    assert semantic["base"]["hash_chain"] == semantic["extended"]["hash_chain"][:3]
    assert semantic["extended"]["hash_chain_valid"] is True
