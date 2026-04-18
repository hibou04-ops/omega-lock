"""Tests for BaseFitness + HybridFitness two-stage re-ranking."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.fitness import BaseFitness, HybridFitness
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


class ScaledTarget:
    """f(x) = scale * x. Used to construct A vs B with different rankings."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale = scale
        self.offset = offset

    def param_space(self) -> list[ParamSpec]:
        return [ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x = float(params["x"])
        return EvalResult(fitness=self.scale * x + self.offset, n_trials=1)


class InvertedTarget:
    """f(x) = -x. So B ranking is inverted relative to A."""

    def param_space(self) -> list[ParamSpec]:
        return [ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=-float(params["x"]), n_trials=1)


def test_base_fitness_passthrough():
    t = ScaledTarget(scale=2.0)
    bf = BaseFitness(target=t)
    r = bf.evaluate({"x": 3.0})
    assert r.fitness == 6.0
    assert bf.param_space() == t.param_space()


def test_hybrid_re_ranks_top_k_by_validation():
    """A ranks by x ascending order (f = x). B inverts (f = -x).
    Top-3 by A: x=10, 9, 8. After B validation, those become -10, -9, -8
    → final order is the same top-3 but internally reversed."""
    search = ScaledTarget(scale=1.0)
    validation = InvertedTarget()
    hybrid = HybridFitness(search_target=search, validation_target=validation, validation_top_k=3)

    candidates = [{"x": float(i)} for i in range(1, 11)]   # x = 1..10
    ordered = hybrid.orchestrate(candidates)

    # A top-3 should have been x=10, 9, 8. After B inverts, final top-3 should be among those
    # but ordered by B fitness (most negative rank first, i.e. x=8 wins).
    top3_final = [h.params["x"] for h in ordered[:3]]
    assert set(top3_final) == {10.0, 9.0, 8.0}
    # Of those three, x=8 has highest (least negative) B fitness
    assert ordered[0].params["x"] == 8.0
    assert ordered[0].validation_result is not None
    assert ordered[0].validation_result.fitness == -8.0


def test_hybrid_only_validates_top_k_not_all():
    """validation_target.evaluate should be called only validation_top_k times."""
    search = ScaledTarget(scale=1.0)

    call_count = {"n": 0}

    class CountingValidation:
        def param_space(self):
            return search.param_space()

        def evaluate(self, params):
            call_count["n"] += 1
            return EvalResult(fitness=-float(params["x"]))

    hybrid = HybridFitness(search_target=search, validation_target=CountingValidation(), validation_top_k=2)
    candidates = [{"x": float(i)} for i in range(1, 11)]
    hybrid.orchestrate(candidates)
    assert call_count["n"] == 2


def test_hybrid_validation_top_k_validates_required_positive():
    with pytest.raises(ValueError):
        HybridFitness(
            search_target=ScaledTarget(),
            validation_target=ScaledTarget(),
            validation_top_k=0,
        )


def test_hybrid_result_final_fitness_falls_back_to_search():
    """Top-K stays on top (sorted by B). Rest keeps search fitness, sorted by A."""
    search = ScaledTarget(scale=1.0)
    validation = InvertedTarget()
    hybrid = HybridFitness(search_target=search, validation_target=validation, validation_top_k=1)

    candidates = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    ordered = hybrid.orchestrate(candidates)

    # Top-1 by A is x=3.0 → validated → final = -3.0. Stays on top.
    # Rest: x=2.0 (final=2), x=1.0 (final=1) — sorted by A descending.
    assert ordered[0].params["x"] == 3.0
    assert ordered[0].validation_result is not None
    assert ordered[0].final_fitness == -3.0

    assert ordered[1].params["x"] == 2.0
    assert ordered[1].validation_result is None
    assert ordered[1].final_fitness == 2.0

    assert ordered[-1].params["x"] == 1.0
    assert ordered[-1].validation_result is None
    assert ordered[-1].final_fitness == 1.0
