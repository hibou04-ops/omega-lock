# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Perturbation sensitivity measurement.

For each parameter p_i:
    continuous/int: stress_i = max(|f(x + eps_i) - f(x)|, |f(x - eps_i) - f(x)|) / eps_i
    bool:           stress_i = |f(flip) - f(baseline)|

Generic version: takes any CalibrableTarget. The target handles clipping
via its own param_space() ranges (we re-clip here defensively too).
"""
from __future__ import annotations

import time
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from omega_lock._parallel import _ordered_eval_map
from omega_lock.params import clip, default_epsilon
from omega_lock.target import CalibrableTarget, EvalResult


@dataclass(init=False)
class StressResult:
    name: str
    baseline_fitness: float
    plus_fitness: float
    minus_fitness: float
    epsilon: float
    raw_stress: float
    normalized_stress: float = 0.0
    is_boolean: bool = False
    stress_suppressed: bool = False
    clipped_plus: bool = False
    clipped_minus: bool = False
    plus_n_trials: int = 0
    minus_n_trials: int = 0

    def __init__(
        self,
        name: str,
        baseline_fitness: float,
        plus_fitness: float,
        minus_fitness: float,
        epsilon: float,
        raw_stress: float,
        normalized_stress: float = 0.0,
        is_boolean: bool = False,
        stress_suppressed: bool = False,
        clipped_plus: bool = False,
        clipped_minus: bool = False,
        plus_n_trials: int = 0,
        minus_n_trials: int = 0,
        *,
        ofi_biased: bool = False,
    ) -> None:
        # `ofi_biased` is a deprecated keyword alias for `stress_suppressed`
        # (OR-merge; both are equal-valued bools, so there is no conflict).
        self.name = name
        self.baseline_fitness = baseline_fitness
        self.plus_fitness = plus_fitness
        self.minus_fitness = minus_fitness
        self.epsilon = epsilon
        self.raw_stress = raw_stress
        self.normalized_stress = normalized_stress
        self.is_boolean = is_boolean
        self.stress_suppressed = bool(stress_suppressed or ofi_biased)
        self.clipped_plus = clipped_plus
        self.clipped_minus = clipped_minus
        self.plus_n_trials = plus_n_trials
        self.minus_n_trials = minus_n_trials

    @property
    def ofi_biased(self) -> bool:
        """Deprecated read alias for `stress_suppressed`."""
        return self.stress_suppressed

    def to_dict(self) -> dict[str, Any]:
        # Dual-key: emit canonical `stress_suppressed` AND deprecated
        # `ofi_biased` (equal value) so 0.2.x readers of either key keep working.
        d = asdict(self)
        d["ofi_biased"] = self.stress_suppressed
        return d


@dataclass
class StressOptions:
    epsilons: dict[str, float] = field(default_factory=dict)
    verbose: bool = True
    progress_callback: Callable[[int, int, StressResult], None] | None = None


def measure_stress(
    target: CalibrableTarget,
    baseline_params: dict[str, Any],
    baseline_result: EvalResult,
    subset: list[str] | None = None,
    options: StressOptions | None = None,
    *,
    executor: Executor | None = None,
) -> list[StressResult]:
    """Measure perturbation sensitivity for each parameter in the target.

    Args:
        target: any CalibrableTarget implementation
        baseline_params: neutral defaults (dict name -> value)
        baseline_result: pre-computed baseline evaluation (fitness for comparison)
        subset: optional list of param names to measure (default: all in target)
        options: StressOptions (custom epsilons, progress callback, verbosity)
        executor: optional, default-off ``concurrent.futures.Executor`` seam.
            When provided, per-parameter measurements are dispatched through it
            and reassembled in input order (the z-score normalization is an
            order-dependent float sum, so input order is load-bearing). When
            ``None`` (the default) measurement is strictly serial and
            byte-identical to prior behavior.

    Returns:
        List of StressResult, one per measured param. Normalized by z-score
        across raw_stress values (for cross-param comparability in ranking).
    """
    opts = options or StressOptions()
    specs = {s.name: s for s in target.param_space()}
    names_to_measure = subset if subset is not None else list(specs.keys())

    baseline_fitness = baseline_result.fitness

    def _measure_one(name: str) -> tuple[StressResult, float]:
        if name not in specs:
            raise KeyError(f"param '{name}' not in target param_space()")
        spec = specs[name]

        t0 = time.time()
        if spec.dtype == "bool":
            p_flip = dict(baseline_params)
            p_flip[name] = not bool(p_flip[name])
            r = target.evaluate(p_flip)
            res = StressResult(
                name=name,
                baseline_fitness=baseline_fitness,
                plus_fitness=r.fitness,
                minus_fitness=baseline_fitness,
                epsilon=1.0,
                raw_stress=abs(r.fitness - baseline_fitness),
                is_boolean=True,
                stress_suppressed=spec.stress_suppressed,
                plus_n_trials=r.sample_count,
                minus_n_trials=baseline_result.sample_count,
            )
        else:
            eps = opts.epsilons.get(name, default_epsilon(spec))
            base_val = baseline_params[name]

            raw_plus_val = base_val + eps
            plus_val = clip(spec, raw_plus_val)
            p_plus = dict(baseline_params)
            p_plus[name] = plus_val
            r_plus = target.evaluate(p_plus)

            raw_minus_val = base_val - eps
            minus_val = clip(spec, raw_minus_val)
            p_minus = dict(baseline_params)
            p_minus[name] = minus_val
            r_minus = target.evaluate(p_minus)

            df_plus = abs(r_plus.fitness - baseline_fitness)
            df_minus = abs(r_minus.fitness - baseline_fitness)
            # Reviewer P2: when clipping pulls plus_val/minus_val back
            # toward base_val, the *effective* perturbation is smaller
            # than `eps` — but pre-fix we still divided by `eps`,
            # under-reporting sensitivity at boundaries. Use the actual
            # signed delta per side as the denominator. When a side is
            # fully clipped to base_val (delta = 0), that side's stress
            # is 0 so it can't dominate the max.
            actual_plus_delta = abs(float(plus_val) - float(base_val))
            actual_minus_delta = abs(float(base_val) - float(minus_val))
            stress_plus = (
                df_plus / actual_plus_delta if actual_plus_delta > 0 else 0.0
            )
            stress_minus = (
                df_minus / actual_minus_delta if actual_minus_delta > 0 else 0.0
            )
            raw = max(stress_plus, stress_minus)

            res = StressResult(
                name=name,
                baseline_fitness=baseline_fitness,
                plus_fitness=r_plus.fitness,
                minus_fitness=r_minus.fitness,
                epsilon=float(eps),
                raw_stress=raw,
                is_boolean=False,
                stress_suppressed=spec.stress_suppressed,
                clipped_plus=(plus_val != raw_plus_val),
                clipped_minus=(minus_val != raw_minus_val),
                plus_n_trials=r_plus.sample_count,
                minus_n_trials=r_minus.sample_count,
            )

        return res, t0

    measured = _ordered_eval_map(executor, names_to_measure, _measure_one)

    results: list[StressResult] = []
    for idx, (res, t0) in enumerate(measured):
        results.append(res)
        dur = time.time() - t0
        if opts.verbose:
            flag = " SUPPRESSED" if res.stress_suppressed else ""
            print(
                f"  [{idx+1:3d}/{len(names_to_measure)}] {res.name:30s} "
                f"stress={res.raw_stress:.4f}{flag} ({dur:.1f}s)"
            )
        if opts.progress_callback is not None:
            opts.progress_callback(idx, len(names_to_measure), res)

    _normalize(results)
    return results


def _normalize(results: list[StressResult]) -> None:
    if not results:
        return
    raws = [r.raw_stress for r in results]
    mean = sum(raws) / len(raws)
    var = sum((s - mean) ** 2 for s in raws) / len(raws)
    std = var ** 0.5
    for r in results:
        r.normalized_stress = ((r.raw_stress - mean) / std) if std > 0 else 0.0


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient on non-negative values. 0 = equal, 1 = max inequality.

    Uses absolute values; all-zero input returns 0.
    """
    if not values:
        return 0.0
    vs = sorted(abs(float(v)) for v in values)
    total = sum(vs)
    if total == 0:
        return 0.0
    n = len(vs)
    cum = sum((i + 1) * v for i, v in enumerate(vs))
    return (2 * cum) / (n * total) - (n + 1) / n


def select_unlock_top_k(
    results: list[StressResult],
    k: int = 3,
    exclude_suppressed: bool | None = None,
    *,
    exclude_ofi: bool | None = None,
) -> list[str]:
    """Top-k parameters by raw_stress.

    Args:
        results: StressResult list
        k: how many to pick (default 3, matches P1 SPEC)
        exclude_suppressed: drop stress-suppressed params before ranking
            (for ablation). Stress-suppressed params are those whose spec
            marks their sensitivity as artificially damped by the eval
            environment.
        exclude_ofi: deprecated keyword alias for ``exclude_suppressed``,
            kept for 0.2.x back-compat. Prefer ``exclude_suppressed``.

    The canonical kwarg wins when both are passed (unlike a simple OR-merge,
    which would let a deprecated True override a canonical False). Either being
    omitted (``None``) falls through to the other; both omitted defaults to
    ``False``. ``exclude_suppressed`` occupies the same 3rd-positional slot the
    old ``exclude_ofi`` held, so existing positional callers keep their meaning.
    """
    if exclude_suppressed is not None:
        exclude = exclude_suppressed
    elif exclude_ofi is not None:
        exclude = exclude_ofi
    else:
        exclude = False
    candidates = [r for r in results if (not exclude or not r.stress_suppressed)]
    ranked = sorted(candidates, key=lambda r: r.raw_stress, reverse=True)
    return [r.name for r in ranked[:k]]
