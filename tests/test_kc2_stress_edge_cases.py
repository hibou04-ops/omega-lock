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
    """With KC-2c explicitly disabled (min_nonzero_stress_count=None) a
    single-spike profile passes Gini + ratio gates. (The bare default is
    now 2 — see test_kc2_single_spike_fails_on_default below.)"""
    stresses = [10.0, 0.0, 0.0, 0.0, 0.0]
    report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=None)
    )
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


def test_kc2_single_spike_fails_on_default():
    """REGRESSION (threshold recalibration plan T2): a single-spike stress
    profile — one axis non-zero, every other axis flat 0.0 — must now FAIL
    KC-2c on the BARE default thresholds (min_nonzero_stress_count flipped
    None -> 2). This closes the verified degeneracy where the single spike
    sails through BOTH gini and the top/bot ratio:

        ratio = top_mean / bot_mean if bot_mean > 0 else inf

    With every non-top axis at 0.0, bot_mean is 0 so ratio is inf, and
    inf >= top_bot_ratio_min is always True; gini is near-maximal too. So
    gini_ok and ratio_ok BOTH pass — proving raising gini_min/ratio_min is
    theater. Only the non-zero count gate (KC-2c) catches it.
    """
    stresses = [10.0, 0.0, 0.0, 0.0, 0.0]
    report = check_kc2(stresses, KCThresholds())  # bare default

    # The whole point: gini + ratio are FOOLED (both pass)...
    assert report.detail["gini_ok"] is True
    assert report.detail["ratio_ok"] is True
    # ...but KC-2c on the default floor catches the single spike.
    assert report.detail["min_nonzero_stress_count"] == 2
    assert report.detail["nonzero_stress_count"] == 1
    assert report.detail["nonzero_ok"] is False
    assert report.status == "FAIL"
    assert "nonzero_stress_count=1<2" in report.message


def test_kc2_pure_objective_none_escape_passes_single_spike():
    """A genuine 1-D objective can opt out of KC-2c via the documented
    escape: KCThresholds.pure_objective(min_nonzero_stress_count=None).
    The single spike then passes (KC-2c is advisory-only again), while the
    action gates KC-3/KC-4b remain disabled by the pure_objective preset.
    """
    stresses = [10.0, 0.0, 0.0, 0.0, 0.0]
    t = KCThresholds.pure_objective(min_nonzero_stress_count=None)
    report = check_kc2(stresses, t)

    assert t.min_nonzero_stress_count is None
    # pure_objective still disables the action-count gates:
    assert t.trade_count_min is None
    assert t.trade_ratio_min is None
    assert report.status == "PASS"
    assert report.detail["nonzero_ok"] is True


def test_kc2_pure_objective_bare_inherits_default_floor():
    """Bare pure_objective() does NOT silently re-disable KC-2c: it only
    overrides the action gates, so min_nonzero_stress_count keeps the new
    default of 2 (escape must be explicit)."""
    t = KCThresholds.pure_objective()
    assert t.min_nonzero_stress_count == 2


def test_kc2_passes_when_nonzero_count_meets_threshold():
    stresses = [10.0, 5.0, 1.0, 0.0, 0.0]
    report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=3)
    )
    assert report.detail["nonzero_stress_count"] == 3
    assert report.detail["nonzero_ok"] is True
    assert report.status == "PASS"


def test_kc2_records_min_nonzero_threshold_in_detail():
    """The detail dict captures what was configured so the artifact is
    self-documenting. The bare default is now 2; the disabled value (None)
    is also faithfully recorded."""
    stresses = [1.0, 1.0, 1.0]
    default_report = check_kc2(stresses, KCThresholds())
    assert default_report.detail["min_nonzero_stress_count"] == 2
    assert default_report.detail["nonzero_stress_count"] == 3
    # 3 nonzero axes >= default floor of 2, so this flat-but-multi-axis
    # profile still passes KC-2c on the default.
    assert default_report.detail["nonzero_ok"] is True

    disabled_report = check_kc2(
        stresses, KCThresholds(min_nonzero_stress_count=None)
    )
    assert disabled_report.detail["min_nonzero_stress_count"] is None
