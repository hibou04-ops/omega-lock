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
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from omega_lock.params import clip, default_epsilon
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


@dataclass
class StressResult:
    name: str
    baseline_fitness: float
    plus_fitness: float
    minus_fitness: float
    epsilon: float
    raw_stress: float
    normalized_stress: float = 0.0
    is_boolean: bool = False
    ofi_biased: bool = False
    clipped_plus: bool = False
    clipped_minus: bool = False
    plus_n_trials: int = 0
    minus_n_trials: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> list[StressResult]:
    """Measure perturbation sensitivity for each parameter in the target.

    Args:
        target: any CalibrableTarget implementation
        baseline_params: neutral defaults (dict name -> value)
        baseline_result: pre-computed baseline evaluation (fitness for comparison)
        subset: optional list of param names to measure (default: all in target)
        options: StressOptions (custom epsilons, progress callback, verbosity)

    Returns:
        List of StressResult, one per measured param. Normalized by z-score
        across raw_stress values (for cross-param comparability in ranking).
    """
    opts = options or StressOptions()
    specs = {s.name: s for s in target.param_space()}
    names_to_measure = subset if subset is not None else list(specs.keys())

    results: list[StressResult] = []
    baseline_fitness = baseline_result.fitness

    for idx, name in enumerate(names_to_measure):
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
                ofi_biased=spec.ofi_biased,
                plus_n_trials=r.n_trials,
                minus_n_trials=baseline_result.n_trials,
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
                ofi_biased=spec.ofi_biased,
                clipped_plus=(plus_val != raw_plus_val),
                clipped_minus=(minus_val != raw_minus_val),
                plus_n_trials=r_plus.n_trials,
                minus_n_trials=r_minus.n_trials,
            )

        results.append(res)
        dur = time.time() - t0
        if opts.verbose:
            flag = " OFI" if res.ofi_biased else ""
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
    exclude_ofi: bool = False,
) -> list[str]:
    """Top-k parameters by raw_stress.

    Args:
        results: StressResult list
        k: how many to pick (default 3, matches P1 SPEC)
        exclude_ofi: drop ofi_biased params before ranking (for ablation)
    """
    candidates = [r for r in results if (not exclude_ofi or not r.ofi_biased)]
    ranked = sorted(candidates, key=lambda r: r.raw_stress, reverse=True)
    return [r.name for r in ranked[:k]]
