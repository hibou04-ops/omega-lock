# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P1: holdout has two semantics now.

Default ``holdout_mode="evidence_only"`` preserves pre-v0.2 behaviour
(holdout never affects status). New ``holdout_mode="gate"`` reads
``holdout_min_fitness`` and ``holdout_min_trade_ratio`` and flips the
status to ``FAIL:HOLDOUT`` when either threshold is violated.

This file pins both halves: evidence_only continues to never affect
status (existing test_holdout.py covers this for the default), and
gate mode actually fails when expected.
"""
from __future__ import annotations

from typing import Any

from omega_lock import (
    EvalResult,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1,
)
from omega_lock.keyholes.phantom import PhantomKeyhole


def _base_config(**overrides: Any) -> P1Config:
    return P1Config(
        unlock_k=3,
        grid_points_per_axis=5,
        kc_thresholds=KCThresholds(trade_count_min=50),
        stress_verbose=False,
        grid_verbose=False,
        **overrides,
    )


class _LowFitnessHoldout:
    """Holdout target that always returns a known-failing fitness."""

    def __init__(self, fitness: float = -10.0, n_trials: int = 200) -> None:
        self._fitness = fitness
        self._n_trials = n_trials

    def param_space(self) -> list[ParamSpec]:
        return PhantomKeyhole(seed=42).param_space()

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=self._fitness, n_trials=self._n_trials)


# ---------------------------------------------------------------------------
# evidence_only mode (default) — existing contract.
# ---------------------------------------------------------------------------


def test_evidence_only_default_keeps_status_unchanged():
    """No threshold + default mode = catastrophic holdout doesn't move status."""
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=_LowFitnessHoldout(fitness=-1e6),
        config=_base_config(),
    )
    # Pre-fix status semantics: holdout fitness has no effect.
    assert "HOLDOUT" not in r.status
    assert r.holdout_result is not None
    assert r.holdout_result["mode"] == "evidence_only"
    assert r.holdout_result["gate_status"] == "EVIDENCE_ONLY"


def test_evidence_only_thresholds_ignored_even_if_set():
    """Setting thresholds without flipping mode is a no-op (defensive)."""
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=_LowFitnessHoldout(fitness=-1e6),
        config=_base_config(
            holdout_mode="evidence_only",
            holdout_min_fitness=0.0,  # would fail in gate mode, ignored here
        ),
    )
    assert "HOLDOUT" not in r.status
    assert r.holdout_result["gate_status"] == "EVIDENCE_ONLY"


# ---------------------------------------------------------------------------
# gate mode — new behaviour.
# ---------------------------------------------------------------------------


def test_gate_mode_fails_status_on_low_fitness():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=_LowFitnessHoldout(fitness=-1e6),
        config=_base_config(
            holdout_mode="gate",
            holdout_min_fitness=0.0,
        ),
    )
    assert "HOLDOUT" in r.status
    assert r.holdout_result["gate_status"] == "FAIL"
    reasons = r.holdout_result.get("gate_failed_reasons", [])
    assert any("fitness" in reason for reason in reasons)


def test_gate_mode_fails_status_on_low_trade_ratio():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        # high enough fitness, but n_trials=1 will produce a near-zero ratio
        # against the hundreds of trials a phantom keyhole grid_best produces.
        holdout_target=_LowFitnessHoldout(fitness=1e9, n_trials=1),
        config=_base_config(
            holdout_mode="gate",
            holdout_min_trade_ratio=0.5,
        ),
    )
    assert "HOLDOUT" in r.status
    reasons = r.holdout_result.get("gate_failed_reasons", [])
    assert any("trade_ratio" in reason for reason in reasons)


def test_gate_mode_passes_when_thresholds_satisfied():
    """High-fitness, high-trade-count holdout passes the gate cleanly."""
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=_LowFitnessHoldout(fitness=1e9, n_trials=10000),
        config=_base_config(
            holdout_mode="gate",
            holdout_min_fitness=-1e6,
            holdout_min_trade_ratio=0.0,
        ),
    )
    assert "HOLDOUT" not in r.status
    assert r.holdout_result["gate_status"] == "PASS"
    assert "gate_failed_reasons" not in r.holdout_result


def test_gate_mode_appends_to_existing_kc_failure_status():
    """If KC-3 already failed, holdout failure appends to the FAIL list
    rather than overwriting — both signals visible to the reviewer."""
    # trade_count_min huge -> KC-3 will likely fail, plus our holdout fails.
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        holdout_target=_LowFitnessHoldout(fitness=-1e6),
        config=P1Config(
            unlock_k=3,
            grid_points_per_axis=5,
            kc_thresholds=KCThresholds(trade_count_min=10**9),  # forces KC-3 fail
            stress_verbose=False,
            grid_verbose=False,
            holdout_mode="gate",
            holdout_min_fitness=0.0,
        ),
    )
    assert r.status.startswith("FAIL")
    assert "HOLDOUT" in r.status
    # Other KC failure markers should also be in there.
    assert "KC-3" in r.status or "KC3" in r.status or "," in r.status


def test_gate_mode_skip_when_no_holdout_target():
    r = run_p1(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=_base_config(
            holdout_mode="gate",
            holdout_min_fitness=0.0,
        ),
    )
    assert r.holdout_result is None
    assert "HOLDOUT" not in r.status
