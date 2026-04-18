"""Random-search baseline for low-dim grid comparison.

Implements the baseline half of P1 SPEC §4 SC-2:
    "저차원 grid top-quartile >= 1.5 x random baseline"

Given the same unlock set and same n_samples budget, uniformly sample
the parameter space and compare top-quartile mean fitness to a grid
search over the same region. The grid is expected to beat random
sampling by >= 1.5x when the landscape has sensible local structure.

Design notes:
- Output shape matches `GridSearch.run` (list[GridPoint]) so downstream
  walk-forward validation, fitness aggregation, and reporting are reusable.
- `numpy.random.default_rng(seed)` gives deterministic draws across
  runs for a given seed.
- `compare_to_grid` assumes fitness is ordered such that higher is
  better AND non-negative. If either grid or random can produce
  negative top-quartile means, the ratio sign flips; callers should
  shift their fitness (e.g. add a large constant) before comparing.
  Division by zero is guarded by returning float('inf') when the
  random top-quartile is exactly zero and grid is positive, or 0.0
  when both are zero.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from omega_lock.grid import GridPoint
from omega_lock.params import clip
from omega_lock.target import CalibrableTarget, ParamSpec


@dataclass
class RandomSearch:
    """Uniform random sampling over unlocked params.

    Usage:
        rs = RandomSearch(target=t, unlocked=["a", "b", "c"], n_samples=125, seed=42)
        points = rs.run(base_params=neutral_defaults)
        best = max(points, key=lambda p: p.result.fitness)

    Args:
        target: CalibrableTarget implementation (same protocol as GridSearch).
        unlocked: names of parameters to randomize. Other params stay at the
            values supplied in `base_params`.
        n_samples: number of evaluations. Default 125 matches the P1 SPEC
            default grid budget (5^3 = 125) for apples-to-apples comparison.
        seed: RNG seed for reproducibility (numpy default_rng).
        verbose: print progress every ~25 samples.
    """
    target: CalibrableTarget
    unlocked: list[str]
    n_samples: int = 125
    seed: int = 42
    verbose: bool = True
    progress_every: int = 25

    def _specs_by_name(self) -> dict[str, ParamSpec]:
        return {s.name: s for s in self.target.param_space()}

    def _sample_one(self, spec: ParamSpec, rng: np.random.Generator) -> Any:
        """Draw a single value for a spec, typed to match spec.dtype."""
        if spec.dtype == "bool":
            return bool(rng.integers(0, 2))
        if spec.dtype == "int":
            # endpoint=True -> inclusive on both ends
            return int(rng.integers(int(spec.low), int(spec.high), endpoint=True))
        # float
        return float(rng.uniform(float(spec.low), float(spec.high)))

    def run(self, base_params: dict[str, Any]) -> list[GridPoint]:
        """Uniform random sampling within each spec's range.

        For float: uniform on [low, high].
        For int:   discrete uniform on [low, high] inclusive.
        For bool:  coin flip.

        Respects clip() defensively (values will never be out of range).
        Returns a list of GridPoint — same shape as GridSearch.run, so
        walk-forward and reporting work downstream.
        """
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {self.n_samples}")

        specs = self._specs_by_name()
        for name in self.unlocked:
            if name not in specs:
                raise KeyError(f"unlocked param '{name}' not in target.param_space()")

        rng = np.random.default_rng(self.seed)

        if self.verbose:
            print(f"  random: {self.n_samples} samples (seed={self.seed}, "
                  f"unlocked={self.unlocked})")

        out: list[GridPoint] = []
        t_start = time.time()
        for i in range(self.n_samples):
            # Iterate self.unlocked in declared order -> deterministic draws.
            unlocked_vals: dict[str, Any] = {}
            params = dict(base_params)
            for name in self.unlocked:
                spec = specs[name]
                raw_val = self._sample_one(spec, rng)
                v = clip(spec, raw_val)
                unlocked_vals[name] = v
                params[name] = v

            t0 = time.time()
            r = self.target.evaluate(params)
            dt = time.time() - t0
            out.append(GridPoint(
                idx=i,
                unlocked=unlocked_vals,
                params=params,
                result=r,
                wall_seconds=dt,
            ))

            if self.verbose and ((i + 1) % self.progress_every == 0 or i == self.n_samples - 1):
                elapsed = time.time() - t_start
                best_so_far = max(out, key=lambda p: p.result.fitness)
                print(
                    f"  [{i+1:4d}/{self.n_samples}] elapsed={elapsed:.0f}s, "
                    f"best_fitness={best_so_far.result.fitness:.3f} "
                    f"(n_trials={best_so_far.result.n_trials})"
                )
        return out


def top_quartile_fitness(points: list[GridPoint]) -> float:
    """Mean fitness of the top 25% of points (P1 SPEC §4 SC-2 metric).

    Takes the top `max(1, n // 4)` points by fitness and returns their
    arithmetic mean. For n < 4 the quartile collapses to the single best
    point; for n=125 it is the mean of the top 31.

    Raises ValueError on empty input.
    """
    if not points:
        raise ValueError("top_quartile_fitness requires at least one point")
    fitnesses = sorted((p.result.fitness for p in points), reverse=True)
    k = max(1, len(fitnesses) // 4)
    top = fitnesses[:k]
    return sum(top) / len(top)


def compare_to_grid(
    grid_points: list[GridPoint],
    random_points: list[GridPoint],
) -> dict[str, float]:
    """Compare grid top-quartile to random top-quartile (SC-2).

    Returns a dict with:
        - 'grid_top_quartile':   mean fitness of grid's top 25%
        - 'random_top_quartile': mean fitness of random's top 25%
        - 'ratio':               grid / random (see note on negative fitness)
        - 'sc2_pass':            1.0 if ratio >= 1.5 else 0.0
                                 (per original P1 SPEC §4 SC-2)

    Sign / zero handling:
        If both top-quartiles are zero -> ratio = 0.0 (undefined, treated
        as a non-pass).
        If random top-quartile is exactly zero but grid is positive ->
        ratio = float('inf') (a trivially passing SC-2).
        If random top-quartile is exactly zero and grid is negative ->
        ratio = float('-inf').
        The SC-2 threshold (>=1.5) assumes non-negative fitness scaling;
        callers with negative-valued fitness should shift first.
    """
    grid_tq = top_quartile_fitness(grid_points)
    rand_tq = top_quartile_fitness(random_points)

    if rand_tq == 0.0:
        if grid_tq == 0.0:
            ratio = 0.0
        elif grid_tq > 0.0:
            ratio = float("inf")
        else:
            ratio = float("-inf")
    else:
        ratio = grid_tq / rand_tq

    sc2_pass = 1.0 if (not math.isnan(ratio) and ratio >= 1.5) else 0.0

    return {
        "grid_top_quartile": float(grid_tq),
        "random_top_quartile": float(rand_tq),
        "ratio": float(ratio),
        "sc2_pass": sc2_pass,
    }
