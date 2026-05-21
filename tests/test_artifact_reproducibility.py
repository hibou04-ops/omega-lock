"""Artifact completeness and reproducibility metadata tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omega_lock import (
    EvalResult,
    IterativeConfig,
    KCThresholds,
    P1Config,
    ParamSpec,
    run_p1,
    run_p1_iterative,
)
from omega_lock.keyholes.phantom import PhantomKeyhole
from omega_lock.p2_tpe import P2Config, _OPTUNA_AVAILABLE, run_p2_tpe


class _TinyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", neutral=0.0, low=0.0, high=1.0),
            ParamSpec(name="y", dtype="float", neutral=0.0, low=0.0, high=1.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        fitness = float(params["x"]) * 2.0 + float(params["y"])
        return EvalResult(fitness=fitness, n_trials=10)


def _loose_kc() -> KCThresholds:
    return KCThresholds(gini_min=0.0, top_bot_ratio_min=0.0, trade_count_min=1)


def test_p1_artifact_includes_full_config_and_schema(tmp_path: Path):
    out = tmp_path / "p1.json"
    cfg = P1Config(
        unlock_k=2,
        grid_points_per_axis=3,
        zoom_rounds=2,
        zoom_factor=0.5,
        kc_thresholds=_loose_kc(),
        stress_verbose=False,
        grid_verbose=False,
        constraint_policy="prefer_feasible",
        holdout_mode="gate",
        holdout_min_fitness=-1.0,
        holdout_min_trade_ratio=0.1,
    )
    result = run_p1(train_target=_TinyTarget(), config=cfg, output_path=out)

    assert result.schema_version == "omega-lock.p1-result.v2"
    assert result.omega_lock_version
    assert result.config["unlock_k"] == 2
    assert result.config_full["zoom_rounds"] == 2
    assert result.config_full["constraint_policy"] == "prefer_feasible"
    assert result.config_full["holdout_mode"] == "gate"
    assert result.kc_thresholds["trade_count_min"] == 1
    assert result.search_settings["method"] == "zooming_grid"

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == result.schema_version
    assert payload["config"]["unlock_k"] == 2
    assert payload["config_full"]["holdout_min_trade_ratio"] == 0.1


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_p2_artifact_includes_full_config_and_seed():
    cfg = P2Config(
        unlock_k=2,
        n_trials=4,
        seed=123,
        kc_thresholds=_loose_kc(),
        stress_verbose=False,
        trial_verbose=False,
    )
    result = run_p2_tpe(train_target=_TinyTarget(), config=cfg)

    assert result.schema_version == "omega-lock.p2-result.v2"
    assert result.omega_lock_version
    assert result.config["seed"] == 123
    assert result.config_full["trial_verbose"] is False
    assert result.seed == 123
    assert result.search_settings["method"] == "tpe"
    assert result.kc_thresholds["trade_count_min"] == 1


def test_iterative_artifact_includes_full_config_and_reuse_warning(tmp_path: Path):
    out = tmp_path / "iterative.json"
    cfg = IterativeConfig(
        rounds=2,
        per_round_unlock_k=2,
        grid_points_per_axis=3,
        kc_thresholds=_loose_kc(),
        stop_on_kc_fail=False,
        min_improvement=-1e9,
        stress_verbose=False,
        grid_verbose=False,
    )
    result = run_p1_iterative(
        train_target=PhantomKeyhole(seed=42),
        test_target=PhantomKeyhole(seed=1337),
        config=cfg,
        output_path=out,
    )

    assert result.schema_version == "omega-lock.iterative-result.v2"
    assert result.config["rounds"] == 2
    assert result.config_full["stop_on_kc_fail"] is False
    assert result.kc_thresholds["trade_count_min"] == 1
    assert result.search_settings["method"] == "iterative_grid"
    assert result.test_reuse_warning == "repeated test reuse weakens KC-4 evidence."
    assert any(result.test_reuse_warning in msg for msg in result.warnings)

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == result.schema_version
    assert payload["config_full"]["min_improvement"] == -1e9
    assert payload["test_reuse_warning"] == result.test_reuse_warning
