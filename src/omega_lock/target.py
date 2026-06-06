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


@dataclass(frozen=True, init=False)
class ParamSpec:
    """Declarative spec of one parameter.

    For bool params, low/high are ignored (use False/True implicitly).
    For int params, low/high are inclusive integer bounds.
    For float params, low/high are inclusive float bounds.

    `neutral` is the baseline value used as the starting point for
    stress measurement. Must lie within [low, high] for numeric types.
    `stress_suppressed` (optional flag): mark params whose stress is
    known to be artificially suppressed by the evaluation environment
    (e.g. a metric that saturates, or an input stream that is mocked
    or low-variance in the test harness). The orchestrator uses this
    to annotate results; it does not filter. `ofi_biased` is a
    deprecated alias kept for backward compatibility.
    """
    name: str
    dtype: ParamDType
    neutral: Any
    low: Any = None
    high: Any = None
    stress_suppressed: bool = False

    def __init__(
        self,
        name: str,
        dtype: ParamDType,
        neutral: Any,
        low: Any = None,
        high: Any = None,
        stress_suppressed: bool = False,
        *,
        ofi_biased: bool = False,
    ) -> None:
        # `ofi_biased` is a deprecated boolean alias for `stress_suppressed`.
        # The OR-merge is deliberate: either flag being True sets it (two
        # bools have no meaningful "conflict" to raise on, unlike EvalResult's
        # int count). `dataclasses.replace()` with the `ofi_biased` alias is
        # unsupported; use `stress_suppressed`.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "neutral", neutral)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "stress_suppressed", bool(stress_suppressed or ofi_biased))
        self._validate()

    def _validate(self) -> None:
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

    @property
    def ofi_biased(self) -> bool:
        """Deprecated alias for `stress_suppressed`."""
        return self.stress_suppressed


@dataclass(init=False)
class EvalResult:
    """Single-evaluation result from a target.

    fitness      — scalar to maximize (required)
    sample_count — number of actions/observations (e.g. trades, posts,
                   samples); used by KC-3. `n_trials` is a deprecated alias.
    metadata     — structured info for diagnostics (regime histogram, errors, etc.)
    artifacts    — large/binary objects (full logs, raw signals); optional
    """
    fitness: float
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        fitness: float,
        sample_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        *,
        n_trials: int | None = None,
    ) -> None:
        """`n_trials` is a deprecated keyword alias for `sample_count`.

        Passing both with different values raises (genuine-mistake guard).
        Note: `dataclasses.replace()` with the `n_trials` alias is unsupported
        (replace re-passes the stored `sample_count`, which then conflicts);
        use the canonical field name. `vars()`/`__dict__` expose `sample_count`
        only.
        """
        if n_trials is not None:
            if sample_count is not None and sample_count != n_trials:
                raise ValueError(
                    "EvalResult: pass sample_count (n_trials is a deprecated alias), "
                    "not both with different values"
                )
            sample_count = n_trials
        self.fitness = fitness
        self.sample_count = 0 if sample_count is None else sample_count
        self.metadata = {} if metadata is None else metadata
        self.artifacts = {} if artifacts is None else artifacts

    @property
    def n_trials(self) -> int:
        """Deprecated alias for `sample_count`."""
        return self.sample_count

    @n_trials.setter
    def n_trials(self, value: int) -> None:
        self.sample_count = value


@runtime_checkable
class CalibrableTarget(Protocol):
    """A system whose parameters can be calibrated.

    Implementations should be deterministic (or report ensemble averages)
    for stress measurement to be meaningful. Non-determinism inflates
    stress noise and weakens walk-forward correlation.
    """
    def param_space(self) -> list[ParamSpec]: ...
    def evaluate(self, params: dict[str, Any]) -> EvalResult: ...
