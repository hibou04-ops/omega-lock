# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P2: KC-2 stress edge cases.

Two issues:

1. Boundary clipping under-reported sensitivity. ``measure_stress``
   computed ``raw = max(df_plus, df_minus) / eps`` even when one or
   both perturbed values were clipped to the parameter bound. The
   actual perturbation magnitude is smaller than ``eps`` after
   clipping, so dividing by ``eps`` shrinks the apparent stress.

2. Single-spike profiles passed Gini + top/bot trivially. With one
   non-zero stress and the rest zero, top mean is positive, bot mean
   is zero, and the ratio goes to infinity. Could be legitimate
   single-axis structure OR a noisy run that hit one axis by luck —
   the audit can't tell. ``min_nonzero_stress_count`` lets the user
   demand evidence of multi-axis sensitivity.
"""
from __future__ import annotations

from typing import Any

from omega_lock.kill_criteria import KCThresholds, check_kc2
from omega_lock.stress import StressOptions, measure_stress
from omega_lock.target import EvalResult, ParamSpec


# ---------------------------------------------------------------------------
# Boundary-clipping denominator fix.
# ---------------------------------------------------------------------------


class _ClippingTarget:
    """Target with one bounded float param. The plus perturbation hits
    the upper bound and clips back; the minus perturbation has full
    headroom. Pre-fix this would have under-reported the plus side's
    contribution because both sides divided by the original eps.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def param_space(self) -> list[ParamSpec]:
        return [ParamSpec(name="x", dtype="float", neutral=0.95, low=0.0, high=1.0)]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        self.calls.append(dict(params))
        # Steep slope -> any movement gives a large fitness delta.
        return EvalResult(fitness=10.0 * float(params["x"]), n_trials=1)


def test_stress_uses_actual_delta_after_clipping():
    """Plus side clips to bound; denominator must reflect actual
    movement, not the originally-requested eps."""
    target = _ClippingTarget()
    base_value = 0.95  # close to upper bound 1.0
    stresses = measure_stress(
        target=target,
        baseline_params={"x": base_value},
        baseline_result=EvalResult(fitness=10.0 * base_value, n_trials=1),
        options=StressOptions(epsilons={"x": 0.1}, verbose=False),
    )
    s = stresses[0]
    # Plus side clipped: 0.95 + 0.1 -> 1.05 -> clipped to 1.0; actual
    # delta = 0.05. Minus side: 0.95 - 0.1 = 0.85; actual delta = 0.10.
    assert s.clipped_plus is True
    assert s.clipped_minus is False
    # Stress on either side should reflect 10.0 (the slope) — using the
    # original eps would under-report the plus side as 5.0.
    # max(stress_plus=10.0, stress_minus=10.0) -> 10.0
    assert abs(s.raw_stress - 10.0) < 1e-9


def test_stress_zero_when_perturbation_clipped_to_baseline_value():
    """If the user-supplied epsilon is forced to 0 by ranges that
    don't permit movement, both sides evaluate identically and stress
    is 0 — no signal to attribute. Pre-fix this would have produced
    NaN-like output via division by 0 when df_plus was non-zero."""
    class _NarrowRange:
        def param_space(self) -> list[ParamSpec]:
            return [ParamSpec(name="x", dtype="float", neutral=0.5, low=0.5, high=0.5)]

        def evaluate(self, params: dict[str, Any]) -> EvalResult:
            # Even if fitness moved, the perturbation didn't:
            return EvalResult(fitness=1.0, n_trials=1)

    stresses = measure_stress(
        target=_NarrowRange(),
        baseline_params={"x": 0.5},
        baseline_result=EvalResult(fitness=1.0, n_trials=1),
        options=StressOptions(epsilons={"x": 0.0}, verbose=False),
    )
    s = stresses[0]
    # eps=0 means actual_plus_delta = 0 and actual_minus_delta = 0,
    # so both sides' stress is the safe 0.0 fallback. raw_stress is 0,
    # which is the correct behavior.
    assert s.raw_stress == 0.0


# ---------------------------------------------------------------------------
# KC-2 single-spike threshold.
# ---------------------------------------------------------------------------


def test_kc2_passes_single_spike_when_min_nonzero_disabled():
    """Default behaviour (min_nonzero_stress_count=None) is unchanged —
    a single-spike profile passes Gini + ratio gates."""
    stresses = [10.0, 0.0, 0.0, 0.0, 0.0]
    report = check_kc2(stresses, KCThresholds())
    assert report.status == "PASS"
    assert report.detail["nonzero_stress_count"] == 1
    assert report.detail["nonzero_ok"] is True


def test_kc2_single_spike_fails_when_min_nonzero_count_required():
    stresses = [10.0, 0.0, 0.0, 0.0, 0.0]
    report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=2)
    )
    assert report.status == "FAIL"
    assert report.detail["nonzero_stress_count"] == 1
    assert report.detail["nonzero_ok"] is False
    assert "nonzero_stress_count=1<2" in report.message


def test_kc2_passes_when_nonzero_count_meets_threshold():
    stresses = [10.0, 5.0, 1.0, 0.0, 0.0]
    report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=3)
    )
    assert report.detail["nonzero_stress_count"] == 3
    assert report.detail["nonzero_ok"] is True
    assert report.status == "PASS"


def test_kc2_records_min_nonzero_threshold_in_detail():
    """Even when the threshold isn't enforced, the detail dict captures
    what was configured so the artifact is self-documenting."""
    stresses = [1.0, 1.0, 1.0]
    default_report = check_kc2(stresses, KCThresholds())
    assert default_report.detail["min_nonzero_stress_count"] is None
    assert default_report.detail["nonzero_stress_count"] == 3

    enforced_report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=2)
    )
    assert enforced_report.detail["min_nonzero_stress_count"] == 2
