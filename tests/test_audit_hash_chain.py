# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P2: optional tamper-evident SHA-256 hash chain on AuditReport.

The README's "append-only audit trail" claim was an in-process
guarantee — runs were appended in order during execution, but nothing
stopped a reader from editing the serialized JSON later and presenting
it as the original. The hash chain gives a cryptographic tamper-
evidence layer without changing the default artifact shape (opt-in
via ``with_hash_chain=True``).

Each entry covers the canonical JSON of the AuditedRun PLUS the
previous_hash. Any edit to any run breaks every subsequent hash.
"""
from __future__ import annotations

from copy import deepcopy
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from omega_lock.audit._target import AuditingTarget
from omega_lock.audit._types import (
    AUDIT_REPORT_SCHEMA_VERSION,
    AuditedRun,
    AuditReport,
)
from omega_lock.target import EvalResult, ParamSpec


class _Target:
    def param_space(self) -> list[ParamSpec]:
        return [ParamSpec(name="x", dtype="float", neutral=0.0, low=-1.0, high=1.0)]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=float(params["x"]) * 2.0, n_trials=1)


def _build_report_via_audit(n_runs: int = 5) -> AuditReport:
    """Run a few evaluations through AuditingTarget and bundle into AuditReport."""
    wrapped = AuditingTarget(_Target())
    for i in range(n_runs):
        wrapped.evaluate({"x": float(i) / 10.0})
    return AuditReport(
        method="hash_chain_test",
        omega_lock_version="0.1.6",
        seed=42,
        started_iso=datetime.now(timezone.utc).isoformat(),
        ended_iso=datetime.now(timezone.utc).isoformat(),
        constraints=(),
        runs=tuple(wrapped.trail),
    )


def _fixed_run(i: int) -> AuditedRun:
    return AuditedRun(
        params={"flag": i % 2 == 0, "x": round(i * 0.25, 2)},
        fitness=float(i) + 0.125,
        n_trials=10 + i,
        metadata={"label": f"run-{i}", "nested": {"rank": i}},
        timestamp_iso=f"2026-01-01T00:00:0{i}+00:00",
        constraints_passed=("nonnegative",),
        constraints_failed=(),
        phase="search",
        call_index=i,
        target_role="train",
        round_index=0,
    )


def _build_fixed_report(n_runs: int = 3) -> AuditReport:
    return AuditReport(
        method="hash_chain_fixture",
        omega_lock_version="0.2.6",
        seed=7,
        started_iso="2026-01-01T00:00:00+00:00",
        ended_iso=f"2026-01-01T00:00:0{max(n_runs - 1, 0)}+00:00",
        constraints=(),
        runs=tuple(_fixed_run(i) for i in range(n_runs)),
    )


def _copy_report(report: AuditReport, runs: tuple[AuditedRun, ...]) -> AuditReport:
    return AuditReport(
        method=report.method,
        omega_lock_version=report.omega_lock_version,
        seed=report.seed,
        started_iso=report.started_iso,
        ended_iso=report.ended_iso,
        constraints=report.constraints,
        runs=runs,
        stress_ranking=report.stress_ranking,
    )


def _signed_payload(report: AuditReport) -> dict[str, Any]:
    return json.loads(report.to_json(with_hash_chain=True))


def _verify_signed_payload(payload: dict[str, Any]) -> bool:
    report = AuditReport.from_json(json.dumps(payload))
    return report.verify_hash_chain(payload["hash_chain"])


# ---------------------------------------------------------------------------
# Default behaviour: hash chain is opt-in.
# ---------------------------------------------------------------------------


def test_to_dict_omits_hash_chain_by_default():
    rpt = _build_report_via_audit()
    d = rpt.to_dict()
    assert "hash_chain" not in d


def test_to_dict_includes_hash_chain_when_requested():
    rpt = _build_report_via_audit()
    d = rpt.to_dict(with_hash_chain=True)
    assert "hash_chain" in d
    assert len(d["hash_chain"]) == len(rpt.runs)


# ---------------------------------------------------------------------------
# Chain shape.
# ---------------------------------------------------------------------------


def test_hash_chain_first_entry_has_no_previous_hash():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    assert chain[0]["previous_hash"] is None


def test_hash_chain_each_entry_links_to_previous():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    for prev, cur in zip(chain, chain[1:]):
        assert cur["previous_hash"] == prev["run_hash"]


def test_hash_chain_call_index_matches_run_call_index():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    for entry, run in zip(chain, rpt.runs):
        assert entry["call_index"] == run.call_index


def test_hash_chain_run_hashes_are_unique_per_run():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    hashes = [e["run_hash"] for e in chain]
    assert len(set(hashes)) == len(hashes)  # collision-free


def test_hash_chain_each_run_hash_is_64_hex_chars():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    for entry in chain:
        assert isinstance(entry["run_hash"], str)
        assert len(entry["run_hash"]) == 64
        int(entry["run_hash"], 16)  # parses as hex


def test_hash_chain_canonical_hash_format_is_stable():
    rpt = _build_fixed_report()

    assert rpt.hash_chain() == [
        {
            "call_index": 0,
            "previous_hash": None,
            "run_hash": "f5340180d46660312ef88f8ed0f8e12ae047dd4e6c6f5335ed50dd1787a03e71",
        },
        {
            "call_index": 1,
            "previous_hash": "f5340180d46660312ef88f8ed0f8e12ae047dd4e6c6f5335ed50dd1787a03e71",
            "run_hash": "663ad0b395182274de0947cc5a3bfb6bb97227a3dea337bf09fc2e48ccecc5ad",
        },
        {
            "call_index": 2,
            "previous_hash": "663ad0b395182274de0947cc5a3bfb6bb97227a3dea337bf09fc2e48ccecc5ad",
            "run_hash": "b553fa1eb85f3bfd316e7a333e75ffd625c4f1de7833929b252eadcb2670c220",
        },
    ]


# ---------------------------------------------------------------------------
# Tamper detection.
# ---------------------------------------------------------------------------


def test_verify_hash_chain_passes_on_unmodified_chain():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    assert rpt.verify_hash_chain(chain) is True


def test_verify_hash_chain_fails_on_swapped_run_hash():
    """Mutating the chain itself (without touching the runs) breaks it."""
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()
    chain[2]["run_hash"] = "0" * 64
    assert rpt.verify_hash_chain(chain) is False


def test_verify_hash_chain_detects_run_mutation_after_chain_was_signed():
    """The whole point of hash-chains: if a run is edited after the
    chain was originally computed, recomputation produces a different
    hash and verify_hash_chain returns False."""
    rpt = _build_report_via_audit()
    original_chain = rpt.hash_chain()

    # Build a tampered AuditReport that has one run with a different
    # fitness — chain recomputation must diverge.
    tampered_runs = list(rpt.runs)
    bad = tampered_runs[2]
    tampered_runs[2] = AuditedRun(
        params=bad.params,
        fitness=999.999,  # tampered
        n_trials=bad.n_trials,
        metadata=bad.metadata,
        timestamp_iso=bad.timestamp_iso,
        constraints_passed=bad.constraints_passed,
        constraints_failed=bad.constraints_failed,
        phase=bad.phase,
        call_index=bad.call_index,
        target_role=bad.target_role,
        round_index=bad.round_index,
    )
    tampered_report = AuditReport(
        method=rpt.method,
        omega_lock_version=rpt.omega_lock_version,
        seed=rpt.seed,
        started_iso=rpt.started_iso,
        ended_iso=rpt.ended_iso,
        constraints=rpt.constraints,
        runs=tuple(tampered_runs),
    )
    # Chain originally computed on the untampered report should not
    # validate against the tampered one.
    assert tampered_report.verify_hash_chain(original_chain) is False


def test_verify_hash_chain_fails_on_truncated_chain():
    rpt = _build_report_via_audit()
    chain = rpt.hash_chain()[:-1]  # drop last
    assert rpt.verify_hash_chain(chain) is False


def test_signed_report_detects_modified_run_params():
    payload = _signed_payload(_build_fixed_report())
    payload["runs"][1]["params"]["x"] = 99.0

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_modified_fitness():
    payload = _signed_payload(_build_fixed_report())
    payload["runs"][1]["fitness"] = -123.0

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_modified_metadata():
    payload = _signed_payload(_build_fixed_report())
    payload["runs"][1]["metadata"]["nested"]["rank"] = 999

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_deleted_run():
    payload = _signed_payload(_build_fixed_report())
    del payload["runs"][1]

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_reordered_runs():
    payload = _signed_payload(_build_fixed_report())
    payload["runs"][0], payload["runs"][1] = payload["runs"][1], payload["runs"][0]

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_duplicated_run():
    payload = _signed_payload(_build_fixed_report())
    payload["runs"][2] = deepcopy(payload["runs"][1])

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_changed_previous_hash():
    payload = _signed_payload(_build_fixed_report())
    payload["hash_chain"][1]["previous_hash"] = "f" * 64

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_changed_run_hash():
    payload = _signed_payload(_build_fixed_report())
    payload["hash_chain"][1]["run_hash"] = "0" * 64

    assert _verify_signed_payload(payload) is False


def test_signed_report_detects_changed_chain_call_index():
    payload = _signed_payload(_build_fixed_report())
    payload["hash_chain"][1]["call_index"] = 99

    assert _verify_signed_payload(payload) is False


def test_audit_report_from_json_rejects_schema_version_mismatch():
    payload = _signed_payload(_build_fixed_report())
    payload["schema_version"] = "omega-lock.audit-report.v999"

    with pytest.raises(ValueError, match="schema_version"):
        AuditReport.from_json(json.dumps(payload))


def test_audit_report_from_json_accepts_current_schema_version():
    payload = _signed_payload(_build_fixed_report())
    assert payload["schema_version"] == AUDIT_REPORT_SCHEMA_VERSION

    report = AuditReport.from_json(json.dumps(payload))

    assert report.verify_hash_chain(payload["hash_chain"]) is True


def test_hash_chain_disabled_vs_enabled_serialization_behavior():
    rpt = _build_fixed_report()
    unsigned_payload = json.loads(rpt.to_json())
    signed_payload = json.loads(rpt.to_json(with_hash_chain=True))

    assert "hash_chain" not in unsigned_payload
    assert "hash_chain" in signed_payload
    assert AuditReport.from_json(json.dumps(signed_payload)).verify_hash_chain(
        signed_payload["hash_chain"]
    ) is True


def test_append_only_extension_preserves_prior_hash_prefix():
    base = _build_fixed_report()
    extended = _copy_report(base, base.runs + (_fixed_run(3),))

    base_chain = base.hash_chain()
    extended_chain = extended.hash_chain()

    assert extended_chain[: len(base_chain)] == base_chain
    assert extended.verify_hash_chain(extended_chain) is True


def test_non_append_mutation_invalidates_prior_hash_prefix():
    base = _build_fixed_report()
    tampered_first = AuditedRun(
        params={"flag": True, "x": 42.0},
        fitness=base.runs[0].fitness,
        n_trials=base.runs[0].n_trials,
        metadata=base.runs[0].metadata,
        timestamp_iso=base.runs[0].timestamp_iso,
        constraints_passed=base.runs[0].constraints_passed,
        constraints_failed=base.runs[0].constraints_failed,
        phase=base.runs[0].phase,
        call_index=base.runs[0].call_index,
        target_role=base.runs[0].target_role,
        round_index=base.runs[0].round_index,
    )
    tampered_extended = _copy_report(
        base,
        (tampered_first, *base.runs[1:], _fixed_run(3)),
    )

    base_chain = base.hash_chain()
    tampered_chain = tampered_extended.hash_chain()
    forged_chain = base_chain + tampered_chain[len(base_chain):]

    assert tampered_chain[: len(base_chain)] != base_chain
    assert tampered_extended.verify_hash_chain(forged_chain) is False


# ---------------------------------------------------------------------------
# JSON roundtrip with the chain.
# ---------------------------------------------------------------------------


def test_to_json_with_hash_chain_round_trips_dict_keys():
    rpt = _build_report_via_audit()
    s = rpt.to_json(with_hash_chain=True)
    d = json.loads(s)
    assert "hash_chain" in d
    assert len(d["hash_chain"]) == len(rpt.runs)


def test_default_to_json_excludes_hash_chain():
    rpt = _build_report_via_audit()
    s = rpt.to_json()
    d = json.loads(s)
    assert "hash_chain" not in d
