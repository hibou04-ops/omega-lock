# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""CalibrableTarget Protocol + core types.

Any system that implements param_space() + evaluate() can be calibrated
by the Omega-Lock pipeline. Examples: a trading strategy, a selector
pipeline, a prompt template, a hyperparameter search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


ParamDType = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    """Declarative spec of one parameter.

    For bool params, low/high are ignored (use False/True implicitly).
    For int params, low/high are inclusive integer bounds.
    For float params, low/high are inclusive float bounds.

    `neutral` is the baseline value used as the starting point for
    stress measurement. Must lie within [low, high] for numeric types.
    `ofi_biased` (optional flag): mark params whose stress is known
    to be artificially suppressed by the evaluation environment
    (e.g. HeartCore's OFI engine with zero-mocked orderbook). The
    orchestrator uses this to annotate results; it does not filter.
    """
    name: str
    dtype: ParamDType
    neutral: Any
    low: Any = None
    high: Any = None
    ofi_biased: bool = False

    def __post_init__(self) -> None:
        if self.dtype == "bool":
            if not isinstance(self.neutral, bool):
                raise ValueError(f"{self.name}: bool neutral must be bool, got {type(self.neutral)}")
            return
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: numeric param needs low and high")
        if self.low > self.high:
            raise ValueError(f"{self.name}: low {self.low} > high {self.high}")
        if not (self.low <= self.neutral <= self.high):
            raise ValueError(f"{self.name}: neutral {self.neutral} outside [{self.low}, {self.high}]")
        if self.dtype == "int":
            if not all(isinstance(v, int) for v in (self.low, self.high, self.neutral)):
                raise ValueError(f"{self.name}: int param values must be int")


@dataclass
class EvalResult:
    """Single-evaluation result from a target.

    fitness     ??scalar to maximize (required)
    n_trials    ??action count (e.g. trades, posts); used by KC-3
    metadata    ??structured info for diagnostics (regime histogram, errors, etc.)
    artifacts   ??large/binary objects (full trade log, raw signals); optional
    """
    fitness: float
    n_trials: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CalibrableTarget(Protocol):
    """A system whose parameters can be calibrated.

    Implementations should be deterministic (or report ensemble averages)
    for stress measurement to be meaningful. Non-determinism inflates
    stress noise and weakens walk-forward correlation.
    """
    def param_space(self) -> list[ParamSpec]: ...
    def evaluate(self, params: dict[str, Any]) -> EvalResult: ...
