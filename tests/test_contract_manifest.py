# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Advisory producer-side smoke for the consumed-surface contract.

Asserts the manifest in ``omega_lock.contract`` against omega-lock's PUBLIC
output (``to_summary()`` / ``to_dict()`` keys) and call signatures only -- never
private internals (``_eval_to_dict``) and never ``dataclasses.fields()`` -- so a
normal internal/perf refactor whose emitted wire output is unchanged does NOT
red this smoke. A rename of a *consumed* name does.

This is an EARLY-WARNING signal at the omega-lock PR, not the contract of
record (that is the consumer's executable test run against omega-lock@main).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from omega_lock import (
    EvalResult,
    P1Config,
    ParamSpec,
    StressResult,
    measure_stress,
    run_p1,
)
from omega_lock.contract import CONSUMED_CONTRACT, CONTRACT_VERSION
from omega_lock.grid import GridPoint


def _sample_grid_summary() -> dict[str, Any]:
    point = GridPoint(
        idx=0,
        unlocked={"a": 1},
        params={"a": 1},
        result=EvalResult(fitness=1.0, sample_count=3),
    )
    return point.to_summary()


def _sample_stress_dict() -> dict[str, Any]:
    stress = StressResult(
        name="a",
        baseline_fitness=1.0,
        plus_fitness=1.0,
        minus_fitness=1.0,
        epsilon=0.1,
        raw_stress=0.0,
    )
    return stress.to_dict()


def test_grid_summary_carries_consumed_wire_keys() -> None:
    keys = set(_sample_grid_summary())
    assert set(CONSUMED_CONTRACT.grid_summary_keys) <= keys


def test_stress_to_dict_carries_consumed_keys() -> None:
    keys = set(_sample_stress_dict())
    assert set(CONSUMED_CONTRACT.stress_dict_keys) <= keys


def test_eval_result_exposes_read_fields() -> None:
    result = EvalResult(fitness=1.0, sample_count=3)
    for name in CONSUMED_CONTRACT.eval_result_read_fields:
        assert hasattr(result, name), name


def test_signatures_cover_consumed_params() -> None:
    checks: list[tuple[Callable[..., Any], tuple[str, ...]]] = [
        (run_p1, CONSUMED_CONTRACT.run_p1_params),
        (P1Config, CONSUMED_CONTRACT.p1config_params),
        (measure_stress, CONSUMED_CONTRACT.measure_stress_params),
        (ParamSpec, CONSUMED_CONTRACT.param_spec_ctor),
    ]
    for obj, expected in checks:
        params = set(inspect.signature(obj).parameters)
        assert set(expected) <= params, (obj, expected, params)


def test_contract_version_present() -> None:
    assert isinstance(CONTRACT_VERSION, str) and CONTRACT_VERSION
