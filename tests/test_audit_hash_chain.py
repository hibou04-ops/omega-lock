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

import json
from datetime import datetime, timezone
from itertools import count
from typing import Any

import pytest

from omega_lock.audit._target import AuditingTarget
from omega_lock.audit._types import AuditedRun, AuditReport, Constraint
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
