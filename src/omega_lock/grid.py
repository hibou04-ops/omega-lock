"""K-dim grid search over unlocked parameters.

Default: 5 points per axis (K=3 → 125 combos per P1 SPEC).

For bool params, the axis is always {False, True} regardless of grid_points.
For int params with a small range (<= grid_points), use the full range.
For continuous/large-int params, evenly spaced linspace.

`ZoomingGridSearch` repeatedly narrows the axis ranges around the current
best point (fractal-vise refinement). Each zoom round shrinks ranges by
`zoom_factor`, so resolution improves geometrically.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from omega_lock.params import clip
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


@dataclass
class GridPoint:
    """One grid evaluation."""
    idx: int
    unlocked: dict[str, Any]       # the unlocked-param values at this point
    params: dict[str, Any]         # full param dict (locked values merged in)
    result: EvalResult
    wall_seconds: float = 0.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "unlocked": dict(self.unlocked),
            "fitness": self.result.fitness,
            "n_trials": self.result.n_trials,
            "wall_s": self.wall_seconds,
        }


def grid_points(spec: ParamSpec, n: int = 5) -> list[Any]:
    """Evenly spaced grid for one param axis, respecting type/range."""
    if spec.dtype == "bool":
        return [False, True]
    if spec.dtype == "int":
        span = spec.high - spec.low
        if span + 1 <= n:
            return list(range(spec.low, spec.high + 1))
        step = span / (n - 1)
        return sorted({int(round(spec.low + step * i)) for i in range(n)})
    # continuous
    return [spec.low + (spec.high - spec.low) * i / (n - 1) for i in range(n)]


def grid_points_in(spec: ParamSpec, lo: Any, hi: Any, n: int = 5) -> list[Any]:
    """Like grid_points, but over a sub-range [lo, hi] clipped to spec bounds.

    For bool: always [False, True] (cannot zoom a 2-valued axis).
    For int: use full enumerated range if it fits within n, else linspace; always
        returns at least two distinct values when range allows.
    For float: linspace(lo, hi, n).
    """
    if spec.dtype == "bool":
        return [False, True]
    lo_c = clip(spec, lo)
    hi_c = clip(spec, hi)
    if hi_c < lo_c:
        lo_c, hi_c = hi_c, lo_c
    if spec.dtype == "int":
        lo_i, hi_i = int(lo_c), int(hi_c)
        if hi_i - lo_i + 1 <= n:
            return list(range(lo_i, hi_i + 1))
        step = (hi_i - lo_i) / (n - 1)
        return sorted({int(round(lo_i + step * i)) for i in range(n)})
    if hi_c == lo_c:
        return [float(lo_c)]
    return [float(lo_c) + (float(hi_c) - float(lo_c)) * i / (n - 1) for i in range(n)]


@dataclass
class GridSearch:
    """Cartesian grid over unlocked params.

    Usage:
        gs = GridSearch(target=t, unlocked=["a", "b"], grid_points_per_axis=5)
        results = gs.run(base_params=neutral_defaults)
        best = max(results, key=lambda p: p.result.fitness)
    """
    target: CalibrableTarget
    unlocked: list[str]
    grid_points_per_axis: int = 5
    verbose: bool = True
    progress_every: int = 25

    def _specs_by_name(self) -> dict[str, ParamSpec]:
        return {s.name: s for s in self.target.param_space()}

    def axes(self) -> dict[str, list[Any]]:
        specs = self._specs_by_name()
        axes: dict[str, list[Any]] = {}
        for name in self.unlocked:
            if name not in specs:
                raise KeyError(f"unlocked param '{name}' not in target.param_space()")
            axes[name] = grid_points(specs[name], self.grid_points_per_axis)
        return axes

    def run(self, base_params: dict[str, Any]) -> list[GridPoint]:
        specs = self._specs_by_name()
        axes = self.axes()
        combos = list(itertools.product(*[axes[n] for n in self.unlocked]))
        if self.verbose:
            sizes = [len(axes[n]) for n in self.unlocked]
            print(f"  grid: {len(combos)} combos (axes={sizes})")

        out: list[GridPoint] = []
        t_start = time.time()
        for i, combo in enumerate(combos):
            unlocked_vals: dict[str, Any] = {}
            params = dict(base_params)
            for name, raw_val in zip(self.unlocked, combo):
                v = clip(specs[name], raw_val)
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
            if self.verbose and ((i + 1) % self.progress_every == 0 or i == len(combos) - 1):
                elapsed = time.time() - t_start
                best_so_far = max(out, key=lambda p: p.result.fitness)
                print(
                    f"  [{i+1:4d}/{len(combos)}] elapsed={elapsed:.0f}s, "
                    f"best_fitness={best_so_far.result.fitness:.3f} "
                    f"(n_trials={best_so_far.result.n_trials})"
                )
        return out


@dataclass
class ZoomingGridSearch:
    """Fractal-vise grid search — repeated narrowing around the current best.

    Round 0 uses the full parameter range. After round r's winner is found,
    each continuous/int axis is re-centered on the winner's value and its
    range is shrunk by `zoom_factor` (ranges are clipped to the original
    spec bounds so we never escape the legal region). Bool axes are always
    {False, True} and don't participate in zooming.

    Args:
        zoom_rounds: total number of grid scans (1 = behaves like GridSearch).
        zoom_factor: fraction of the previous range to keep. 0.5 halves ranges
            each round → resolution improves geometrically.

    Budget: `zoom_rounds * grid_points_per_axis^K` evaluations (some reuse
    across rounds is NOT attempted — winner of round r is typically near the
    centre of round r+1's grid but not exactly on it after re-linspace).

    Returns a flat list of all GridPoints from all rounds, in evaluation
    order. The global best is `max(returned, key=fitness)` — naturally the
    final-round winner in smooth landscapes.
    """
    target: CalibrableTarget
    unlocked: list[str]
    grid_points_per_axis: int = 5
    zoom_rounds: int = 3
    zoom_factor: float = 0.5
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.zoom_rounds < 1:
            raise ValueError(f"zoom_rounds must be >= 1, got {self.zoom_rounds}")
        if not (0.0 < self.zoom_factor < 1.0):
            raise ValueError(f"zoom_factor must be in (0, 1), got {self.zoom_factor}")

    def _specs_by_name(self) -> dict[str, ParamSpec]:
        return {s.name: s for s in self.target.param_space()}

    def run(self, base_params: dict[str, Any]) -> list[GridPoint]:
        specs = self._specs_by_name()
        for name in self.unlocked:
            if name not in specs:
                raise KeyError(f"unlocked param '{name}' not in target.param_space()")

        # Initial ranges = full spec ranges
        current_lo: dict[str, Any] = {}
        current_hi: dict[str, Any] = {}
        for name in self.unlocked:
            spec = specs[name]
            if spec.dtype == "bool":
                current_lo[name] = False
                current_hi[name] = True
            else:
                current_lo[name] = spec.low
                current_hi[name] = spec.high

        all_points: list[GridPoint] = []
        global_idx = 0
        global_t0 = time.time()

        for zoom_r in range(self.zoom_rounds):
            axes = {
                name: grid_points_in(specs[name], current_lo[name], current_hi[name],
                                      self.grid_points_per_axis)
                for name in self.unlocked
            }
            combos = list(itertools.product(*[axes[n] for n in self.unlocked]))
            if self.verbose:
                sizes = [len(axes[n]) for n in self.unlocked]
                ranges = {n: (current_lo[n], current_hi[n]) for n in self.unlocked}
                print(f"  zoom[{zoom_r+1}/{self.zoom_rounds}] {len(combos)} combos "
                      f"axes={sizes} ranges={ranges}")

            round_points: list[GridPoint] = []
            for combo in combos:
                unlocked_vals: dict[str, Any] = {}
                params = dict(base_params)
                for name, raw_val in zip(self.unlocked, combo):
                    v = clip(specs[name], raw_val)
                    unlocked_vals[name] = v
                    params[name] = v
                t_eval0 = time.time()
                r = self.target.evaluate(params)
                round_points.append(GridPoint(
                    idx=global_idx,
                    unlocked=unlocked_vals,
                    params=params,
                    result=r,
                    wall_seconds=time.time() - t_eval0,
                ))
                global_idx += 1
            all_points.extend(round_points)

            winner = max(round_points, key=lambda p: p.result.fitness)
            if self.verbose:
                elapsed = time.time() - global_t0
                print(f"    round winner: fitness={winner.result.fitness:.4f} "
                      f"at {winner.unlocked} ({elapsed:.1f}s cumulative)")

            # Zoom ranges for next round
            if zoom_r < self.zoom_rounds - 1:
                for name in self.unlocked:
                    spec = specs[name]
                    if spec.dtype == "bool":
                        continue
                    prev_lo = current_lo[name]
                    prev_hi = current_hi[name]
                    prev_span = float(prev_hi) - float(prev_lo)
                    center = float(winner.unlocked[name])
                    half_new = prev_span * self.zoom_factor / 2.0
                    new_lo = max(float(spec.low), center - half_new)
                    new_hi = min(float(spec.high), center + half_new)
                    if spec.dtype == "int":
                        new_lo_i = int(round(new_lo))
                        new_hi_i = int(round(new_hi))
                        # Don't collapse int axes to a single point until fully converged
                        if new_hi_i == new_lo_i and prev_hi > prev_lo:
                            if new_hi_i < spec.high:
                                new_hi_i += 1
                            elif new_lo_i > spec.low:
                                new_lo_i -= 1
                        current_lo[name] = new_lo_i
                        current_hi[name] = new_hi_i
                    else:
                        current_lo[name] = new_lo
                        current_hi[name] = new_hi

        return all_points
