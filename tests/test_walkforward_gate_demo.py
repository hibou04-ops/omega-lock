# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Regression tests for the walk-forward gate case study and the Optuna
bridge demo.

These demos are deterministic by construction (hash-based noise, fixed
seeds), and the README Quickstart quotes their exact numbers. The tests
below pin those headline numbers so a behavior drift in the library (or
an accidental edit to the demo) cannot silently invalidate the README's
self-described output.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

import walkforward_gate_demo


def _run_main(module) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        code = module.main()
    assert code == 0
    return out.getvalue()


def test_walkforward_gate_demo_runs_and_gate_fails_the_naive_ranking():
    output = _run_main(walkforward_gate_demo)

    assert "pipeline status      = FAIL:KC-4" in output
    assert "Walk-forward gate demo PASSED." in output
    # The feasible-only re-gate must pass.
    assert "KC-4 verdict         = PASS" in output


def test_walkforward_gate_demo_numbers_match_readme_quickstart():
    """The README Quickstart quotes these exact deterministic numbers."""
    output = _run_main(walkforward_gate_demo)

    # best_any: train 5.967 -> holdout 1.527 (collapse)
    assert "train fitness : 5.967" in output
    assert "holdout       : 1.527" in output
    # best_feasible: train 5.233 -> holdout 5.276 (holds)
    assert "train fitness : 5.233" in output
    assert "holdout       : 5.276" in output
    # KC-4 Pearson on the naive top-10
    assert "Pearson(train, test) = 0.179" in output


def test_walkforward_gate_demo_target_is_deterministic_per_slice():
    t_a = walkforward_gate_demo.NoisySliceTarget(walkforward_gate_demo.TRAIN_SEED)
    t_b = walkforward_gate_demo.NoisySliceTarget(walkforward_gate_demo.TRAIN_SEED)
    t_other = walkforward_gate_demo.NoisySliceTarget(walkforward_gate_demo.TEST_SEED)
    params = {"gain": 7.5, "bias": 0.5, "offset": 0.0}

    assert t_a.evaluate(params).fitness == t_b.evaluate(params).fitness
    # A different slice re-draws the noise in the fragile region.
    assert t_a.evaluate(params).fitness != t_other.evaluate(params).fitness


def test_optuna_bridge_demo_gate_fails_study_ranking():
    pytest.importorskip("optuna")
    import optuna_audit_demo

    output = _run_main(optuna_audit_demo)

    assert "KC-4 verdict         = FAIL" in output
    assert "Optuna bridge demo PASSED." in output
