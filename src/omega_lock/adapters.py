# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Adapter helpers ??bridge arbitrary external systems into CalibrableTarget.

`CalibrableTarget` is a structural Protocol; any object with `param_space()`
and `evaluate()` satisfies it. These adapters remove boilerplate for common
wrapping patterns.

`CallableAdapter` ??wrap a plain `(params_dict) -> float` function.
    Fastest way to calibrate an existing black-box: provide a param spec
    list and a scoring function. No subclassing required.

Example:
    def score(params):
        return -((params["a"] - 3) ** 2 + (params["b"] - 7) ** 2)

    target = CallableAdapter(
        fitness_fn=score,
        specs=[
            ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ],
    )
    result = run_p1(train_target=target, ...)

External integration pattern (what the original HeartCore adapter would do):
    Write a thin wrapper class that:
      1. Holds a reference to your external system (e.g. a strategy engine,
         a scoring service, a trained model).
      2. Translates `params: dict[str, Any]` into that system's native call.
      3. Returns EvalResult with `fitness`, `n_trials` (action count for
         KC-3), and optional `metadata` for diagnostics.
    See `examples/adapter_example.py` for a runnable template.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from omega_lock.target import EvalResult, ParamSpec


FitnessFn = Callable[[dict[str, Any]], float]
NTrialsFn = Callable[[dict[str, Any]], int]
MetadataFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class CallableAdapter:
    """Wrap `(params) -> float` into a CalibrableTarget.

    Args:
        fitness_fn: the scalar objective to MAXIMIZE.
            Must accept a dict keyed by `ParamSpec.name`. Omega-Lock's
            orchestrator always maximizes; if your native score is a loss,
            negate it before returning.
        specs: the parameter space.
        n_trials_fn: optional callable returning the evaluation's action
            count (used by KC-3). If omitted, reports 1 per evaluation ??
            which is fine for toy problems but will trip `KCThresholds.
            trade_count_min` on realistic configs. Override for real work.
        metadata_fn: optional callable returning per-evaluation diagnostic
            info attached to `EvalResult.metadata`. Purely informational.
    """
    fitness_fn: FitnessFn
    specs: list[ParamSpec]
    n_trials_fn: NTrialsFn | None = None
    metadata_fn: MetadataFn | None = None

    def __post_init__(self) -> None:
        if not callable(self.fitness_fn):
            raise TypeError("fitness_fn must be callable")
        if not self.specs:
            raise ValueError("specs must be non-empty")
        if self.n_trials_fn is not None and not callable(self.n_trials_fn):
            raise TypeError("n_trials_fn must be callable or None")
        if self.metadata_fn is not None and not callable(self.metadata_fn):
            raise TypeError("metadata_fn must be callable or None")

    def param_space(self) -> list[ParamSpec]:
        return list(self.specs)

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        fitness = float(self.fitness_fn(params))
        n_trials = int(self.n_trials_fn(params)) if self.n_trials_fn else 1
        metadata = dict(self.metadata_fn(params)) if self.metadata_fn else {}
        return EvalResult(
            fitness=fitness,
            n_trials=n_trials,
            metadata=metadata,
        )
