"""Fitness adapters — BaseFitness + HybridFitness.

BaseFitness: identity wrapper around a target (for symmetry with HybridFitness).

HybridFitness: two-stage A+B pattern.
    A (search_target): fast, cheap — used for grid/stress exploration (125+ runs).
    B (validation_target): slow, precise — used only on top-K candidates (≤5 runs).

The final ranking is driven by B's fitness (fall back to A if B absent).
Typical use:
    - A = diversity heuristic from history.jsonl (offline, ~15ms/eval)
    - B = LLM-judge rubric via Gemini-pro (~3s/eval)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


@dataclass
class BaseFitness:
    """Pass-through wrapper. Use as a no-op adapter when no hybrid pattern is needed."""
    target: CalibrableTarget

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return self.target.evaluate(params)

    def param_space(self) -> list[ParamSpec]:
        return self.target.param_space()


@dataclass
class HybridResult:
    params: dict[str, Any]
    search_result: EvalResult         # A
    validation_result: EvalResult | None = None   # B (None → not re-evaluated)

    @property
    def final_fitness(self) -> float:
        if self.validation_result is not None:
            return self.validation_result.fitness
        return self.search_result.fitness


@dataclass
class HybridFitness:
    """Two-stage evaluator: explore with A, validate top-K with B.

    Contract:
        - search_target.param_space() must equal validation_target.param_space()
          (identity comparison is enough in practice — pass the same list)
        - Both targets evaluate the same params dict
        - orchestrate() returns candidates sorted by final_fitness (desc)
    """
    search_target: CalibrableTarget
    validation_target: CalibrableTarget
    validation_top_k: int = 5

    def __post_init__(self) -> None:
        if self.validation_top_k < 1:
            raise ValueError(f"validation_top_k must be >= 1, got {self.validation_top_k}")

    def param_space(self) -> list[ParamSpec]:
        return self.search_target.param_space()

    def search(self, params: dict[str, Any]) -> EvalResult:
        return self.search_target.evaluate(params)

    def validate(self, params: dict[str, Any]) -> EvalResult:
        return self.validation_target.evaluate(params)

    def orchestrate(self, candidates: list[dict[str, Any]]) -> list[HybridResult]:
        """Stage A: evaluate all candidates. Stage B: validate top-K.

        Semantics: A produces a short-list, B picks the winner among the
        short-list. The returned ordering is:
            [top-K sorted by final_fitness (B)] + [rest sorted by search fitness (A)]

        This matches the "A ranks coarsely, B refines among A's top" use case.
        A candidate demoted by B does NOT drop below unvalidated candidates —
        because they never entered B's scrutiny.
        """
        stage_a: list[HybridResult] = []
        for p in candidates:
            r = self.search(p)
            stage_a.append(HybridResult(params=dict(p), search_result=r))

        stage_a.sort(key=lambda h: h.search_result.fitness, reverse=True)

        top = stage_a[: self.validation_top_k]
        rest = stage_a[self.validation_top_k:]

        for h in top:
            h.validation_result = self.validate(h.params)

        top.sort(key=lambda h: h.final_fitness, reverse=True)
        return top + rest
