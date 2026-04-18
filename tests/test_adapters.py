"""Tests for adapters — CallableAdapter and the bridge pattern."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock import (
    CallableAdapter, EvalResult, KCThresholds, P1Config, ParamSpec, run_p1,
)


def test_callable_adapter_wraps_fitness_fn():
    def score(p):
        return p["x"] * 2.0

    target = CallableAdapter(
        fitness_fn=score,
        specs=[ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)],
    )
    r = target.evaluate({"x": 3.5})
    assert r.fitness == 7.0
    assert r.n_trials == 1


def test_callable_adapter_uses_custom_n_trials():
    target = CallableAdapter(
        fitness_fn=lambda p: p["x"],
        specs=[ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)],
        n_trials_fn=lambda p: int(p["x"] * 10),
    )
    r = target.evaluate({"x": 2.5})
    assert r.n_trials == 25


def test_callable_adapter_uses_custom_metadata():
    target = CallableAdapter(
        fitness_fn=lambda p: 1.0,
        specs=[ParamSpec(name="x", dtype="float", low=0.0, high=10.0, neutral=5.0)],
        metadata_fn=lambda p: {"input_x": p["x"]},
    )
    r = target.evaluate({"x": 7.0})
    assert r.metadata == {"input_x": 7.0}


def test_callable_adapter_param_space_returns_specs():
    specs = [
        ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5),
        ParamSpec(name="b", dtype="int", low=1, high=10, neutral=5),
    ]
    target = CallableAdapter(fitness_fn=lambda p: 0.0, specs=specs)
    assert target.param_space() == specs


def test_callable_adapter_rejects_non_callable_fitness():
    with pytest.raises(TypeError):
        CallableAdapter(
            fitness_fn="not a function",  # type: ignore[arg-type]
            specs=[ParamSpec("x", "float", low=0.0, high=1.0, neutral=0.5)],
        )


def test_callable_adapter_rejects_empty_specs():
    with pytest.raises(ValueError):
        CallableAdapter(fitness_fn=lambda p: 0.0, specs=[])


def test_callable_adapter_runs_through_p1_pipeline():
    """Full pipeline: wrap a function, run calibration, find optimum."""
    def score(p):
        # Quadratic bowl, optimum at (3, 7)
        return -((p["a"] - 3.0) ** 2 + (p["b"] - 7.0) ** 2)

    target = CallableAdapter(
        fitness_fn=score,
        specs=[
            ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
            ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ],
    )
    r = run_p1(
        train_target=target,
        config=P1Config(
            unlock_k=2, grid_points_per_axis=5, zoom_rounds=4,
            kc_thresholds=KCThresholds(trade_count_min=1, gini_min=0.0, top_bot_ratio_min=1.0),
            stress_verbose=False, grid_verbose=False,
        ),
    )
    assert r.grid_best is not None
    # Zoom should get within 0.25 of true optimum
    err = ((r.grid_best["unlocked"]["a"] - 3.0) ** 2
           + (r.grid_best["unlocked"]["b"] - 7.0) ** 2) ** 0.5
    assert err <= 0.25, f"zoom should find optimum within 0.25; got err={err}"


def test_callable_adapter_minimal_docstring_contract():
    """The adapter promises: callable + specs in → CalibrableTarget out.
    No subclassing, no boilerplate. This test documents the minimum call."""
    target = CallableAdapter(
        fitness_fn=lambda p: -(p["x"] - 1.0) ** 2,
        specs=[ParamSpec(name="x", dtype="float", low=-5.0, high=5.0, neutral=0.0)],
    )
    # Satisfies the CalibrableTarget Protocol structurally
    from omega_lock import CalibrableTarget
    assert isinstance(target, CalibrableTarget)  # runtime_checkable on Protocol
