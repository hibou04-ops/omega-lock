"""Tests for stress measurement + top-K selection + Gini."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.stress import (
    gini_coefficient,
    measure_stress,
    select_unlock_top_k,
)
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


class LinearTarget:
    """f(x, y, z) = a*x + b*y + c*z. Stress should be proportional to |coef| * eps."""

    def __init__(self, a: float = 10.0, b: float = 1.0, c: float = 0.1):
        self.a, self.b, self.c = a, b, c

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="y", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="z", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        v = self.a * params["x"] + self.b * params["y"] + self.c * params["z"]
        return EvalResult(fitness=v, n_trials=1)


def test_gini_all_equal_is_zero():
    assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_gini_monotone_with_inequality():
    even = gini_coefficient([1.0] * 10)
    uneven = gini_coefficient([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    assert uneven > even
    assert uneven > 0.8   # near-maximal inequality


def test_gini_empty_is_zero():
    assert gini_coefficient([]) == 0.0


def test_gini_all_zero_is_zero():
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


def test_stress_ranking_matches_coefficients():
    """For a linear target, stress of each param should scale with |coef|."""
    target = LinearTarget(a=10.0, b=1.0, c=0.1)
    base_params = {"x": 5.0, "y": 5.0, "z": 5.0}
    baseline = target.evaluate(base_params)

    results = measure_stress(
        target=target,
        baseline_params=base_params,
        baseline_result=baseline,
    )
    # eps = 10% of (10 - 0) = 1.0 for each axis; stress_i = |coef| * eps / eps = |coef|
    by_name = {r.name: r.raw_stress for r in results}
    assert by_name["x"] == pytest.approx(10.0, abs=1e-6)
    assert by_name["y"] == pytest.approx(1.0, abs=1e-6)
    assert by_name["z"] == pytest.approx(0.1, abs=1e-6)


def test_top_k_selects_highest_stress():
    target = LinearTarget(a=10.0, b=1.0, c=0.1)
    base_params = {"x": 5.0, "y": 5.0, "z": 5.0}
    baseline = target.evaluate(base_params)
    results = measure_stress(
        target=target,
        baseline_params=base_params,
        baseline_result=baseline,
        options=None,
    )

    top1 = select_unlock_top_k(results, k=1)
    top2 = select_unlock_top_k(results, k=2)

    assert top1 == ["x"]
    assert top2 == ["x", "y"]


class BoolTarget:
    """flip a single bool param — used to verify bool stress handling."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="flag", dtype="bool", neutral=False),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=7.0 if params["flag"] else 2.0, n_trials=1)


def test_bool_stress_is_absolute_delta():
    target = BoolTarget()
    base_params = {"flag": False}
    baseline = target.evaluate(base_params)
    results = measure_stress(target, base_params, baseline)
    assert len(results) == 1
    assert results[0].is_boolean
    assert results[0].raw_stress == pytest.approx(5.0, abs=1e-9)
    assert results[0].epsilon == 1.0


class IntTarget:
    """integer param, f(n) = n."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="n", dtype="int", low=1, high=10, neutral=5),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=float(params["n"]), n_trials=1)


def test_int_stress_uses_unit_epsilon():
    target = IntTarget()
    base_params = {"n": 5}
    baseline = target.evaluate(base_params)
    results = measure_stress(target, base_params, baseline)
    assert len(results) == 1
    assert not results[0].is_boolean
    # plus=6, minus=4, raw = max(|6-5|, |4-5|)/1 = 1.0
    assert results[0].raw_stress == pytest.approx(1.0, abs=1e-9)


def test_ofi_exclusion():
    """Params marked ofi_biased should be filtered when exclude_ofi=True."""

    class OFILinear:
        def param_space(self):
            return [
                ParamSpec(name="fast", dtype="float", low=0.0, high=10.0, neutral=5.0),
                ParamSpec(name="ofi_fast", dtype="float", low=0.0, high=10.0, neutral=5.0, ofi_biased=True),
            ]

        def evaluate(self, params):
            return EvalResult(fitness=100.0 * params["ofi_fast"] + 1.0 * params["fast"], n_trials=1)

    t = OFILinear()
    base = {"fast": 5.0, "ofi_fast": 5.0}
    baseline = t.evaluate(base)
    results = measure_stress(t, base, baseline)

    top_all = select_unlock_top_k(results, k=1, exclude_ofi=False)
    top_ex_ofi = select_unlock_top_k(results, k=1, exclude_ofi=True)

    assert top_all == ["ofi_fast"]
    assert top_ex_ofi == ["fast"]
