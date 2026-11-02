"""Exact decimal relative bounds complement score-unit and correlation gates."""
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
import math
from typing import Any, cast

import pytest

from omega_lock import audit_optuna_study, gate_scores
from omega_lock.holdout_defense import score_degradation
from omega_lock.kill_criteria import KCThresholds

requires_optuna = pytest.mark.skipif(find_spec("optuna") is None, reason="optional [p2] dependency")


def study(values, *, direction="maximize", feasible=None):
    optuna = cast(Any, import_module("optuna"))
    result = optuna.create_study(direction=direction)
    for i, value in enumerate(values):
        result.add_trial(optuna.trial.create_trial(
            params={"x": i}, distributions={"x": optuna.distributions.IntDistribution(0, len(values)-1)},
            value=value, user_attrs={} if feasible is None else {"feasible": feasible[i]},
        ))
    return result


def test_normalized_collapse_is_blocked_by_helper_and_mean_api_despite_small_absolute_gap():
    train, holdout = [0.7, 0.8, 0.9], [0.01, 0.02, 0.03]
    result = score_degradation(train, holdout)
    assert result["absolute_degradation_ok"] is True
    assert result["relative_degradation"] == 0.975
    assert result["relative_degradation_ok"] is False and result["degradation_ok"] is False
    verdict = gate_scores(train, holdout)
    assert verdict.degradation_gap is not None
    assert verdict.pearson == 1.0 and verdict.degradation_gap < 1.0
    assert not verdict.passed and not verdict.deployment_authorized
    assert verdict.to_dict()["max_relative_degradation"] == 0.5


@requires_optuna
def test_normalized_optuna_collapse_never_authorizes_deployment():
    report = audit_optuna_study(study([0.7, 0.8, 0.9]), top_n=3,
                               holdout_evaluate=lambda p: [0.01, 0.02, 0.03][p["x"]])
    assert report.degradation_gap is not None
    assert report.pearson == 1.0 and report.degradation_gap < 1.0
    assert report.relative_degradation == 0.975
    assert not report.passed and not report.deployment_authorized and report.gated_best is None


@pytest.mark.parametrize("direction,holdout", [
    ("maximize", [0.0, 0.1, 0.2]), ("minimize", [0.2, 0.3, 0.4]),
])
def test_decimal_mean_half_drop_boundary_is_inclusive_in_both_directions(direction, holdout):
    result = score_degradation([0.1, 0.2, 0.3], holdout, direction=direction)
    assert result["relative_degradation"] == 0.5
    assert result["relative_degradation_ok"] is True and result["degradation_ok"] is True
    assert gate_scores([0.1, 0.2, 0.3], holdout, direction=direction).deployment_authorized


def test_exact_relative_excess_is_blocked_even_when_the_display_rounds_down_to_half():
    train, holdout = [0.0, 1.0, 2.0], [-1e-17, 0.5, 1.0]
    result = score_degradation(train, holdout)
    assert result["relative_degradation"] == 0.5
    assert result["absolute_degradation_ok"] is True
    assert result["relative_degradation_ok"] is False
    assert not gate_scores(train, holdout).deployment_authorized


@requires_optuna
def test_optuna_exact_relative_mean_excess_cannot_use_rounded_half_for_approval():
    report = audit_optuna_study(study([0.0, 1.0, 2.0]), top_n=3,
                               holdout_evaluate=lambda p: [-1e-17, 0.5, 1.0][p["x"]])
    assert report.relative_degradation == 0.5
    assert report.kc_report.detail["relative_degradation_ok"] is False
    assert not report.deployment_authorized


@requires_optuna
@pytest.mark.parametrize("direction,values", [
    ("maximize", [0.5, 0.4, 0.3]), ("minimize", [1.5, 1.2, 0.9]),
])
def test_selected_candidate_exact_half_is_inclusive(direction, values):
    report = audit_optuna_study(study([1.0, 0.8, 0.6], direction=direction), top_n=3,
                               holdout_evaluate=lambda p: values[p["x"]])
    assert report.relative_degradation == report.selected_relative_degradation == 0.5
    assert report.deployment_authorized


@pytest.mark.parametrize("train,holdout", [
    ([-1.0, 0.0, 1.0], [-1.1, -0.1, 0.9]),
    ([-3.0, -2.0, -1.0], [-3.2, -2.2, -1.2]),
])
def test_nonpositive_original_train_mean_is_explicitly_undefined_with_absolute_defense(train, holdout):
    result = score_degradation(train, holdout)
    assert result["relative_degradation"] is None and result["relative_degradation_ok"] is None
    assert result["relative_degradation_status"] == "UNDEFINED_NONPOSITIVE_TRAIN_BASELINE"
    assert gate_scores(train, holdout).deployment_authorized
    assert not gate_scores(train, [value-2 for value in train]).deployment_authorized


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01, None, True, False])
def test_relative_threshold_is_finite_nonnegative_and_never_boolean(bad):
    with pytest.raises(ValueError, match="max_relative_degradation"):
        score_degradation([1, 2, 3], [1, 2, 3], max_relative_degradation=bad)
    with pytest.raises(ValueError, match="max_relative_degradation"):
        gate_scores([1, 2, 3], [1, 2, 3], max_relative_degradation=bad)
    with pytest.raises(ValueError, match="max_relative_degradation"):
        audit_optuna_study(None, max_relative_degradation=bad)


def test_raising_only_absolute_limit_cannot_disable_the_default_relative_policy():
    assert not gate_scores([1, 2, 3], [-100, -99, -98], max_degradation_gap=101).deployment_authorized
    relaxed = gate_scores([1, 2, 3], [-100, -99, -98],
                          max_degradation_gap=101, max_relative_degradation=50.5)
    assert relaxed.deployment_authorized
    assert not replace(relaxed, max_relative_degradation=0.5).deployment_authorized


@requires_optuna
def test_passing_mean_does_not_waive_selected_relative_policy():
    train, holdout = [0.5, 0.8, 1.0], [0.6, 0.9, 0.4]
    thresholds = KCThresholds.pure_objective(pearson_min=-1.0)
    assert gate_scores(train, holdout, thresholds=thresholds).deployment_authorized
    report = audit_optuna_study(study(train), holdout_evaluate=lambda p: holdout[p["x"]],
                               top_n=3, thresholds=thresholds)
    assert report.relative_degradation is not None and report.selected_degradation_gap is not None
    assert report.relative_degradation < 0.5 and report.selected_degradation_gap < 1.0
    assert report.selected_relative_degradation == 0.6
    assert not report.deployment_authorized and report.gated_best is None


@requires_optuna
def test_feasible_candidate_outside_top_n_keeps_its_own_positive_train_baseline():
    report = audit_optuna_study(study([9.0, 8.0, 1.0], feasible=[False, False, True]), top_n=2,
                               holdout_evaluate=lambda p: [9.0, 8.0, 0.4][p["x"]])
    assert report.relative_degradation == 0.0 and report.selected_relative_degradation == 0.6
    assert report.best_feasible is not None and report.selected_degradation_gap is not None
    assert report.best_feasible.number == 2 and report.selected_degradation_gap < 1.0
    assert report.kc_report.status == "FAIL" and not report.deployment_authorized


@requires_optuna
def test_selected_exact_excess_cannot_be_approved_by_its_rounded_half_display():
    report = audit_optuna_study(study([0.5, 1.0, 2.0]), top_n=3, max_degradation_gap=2.0,
                               holdout_evaluate=lambda p: [0.5, 1.0, 0.9999999999999999][p["x"]])
    assert report.relative_degradation is not None
    assert report.relative_degradation < 0.5
    assert report.selected_relative_degradation == 0.5  # exact ratio is 0.50000000000000005
    assert report.kc_report.detail["selected_relative_degradation_ok"] is False
    assert report.selected_degradation_gap is not None
    assert report.selected_degradation_gap < 2.0 and not report.deployment_authorized


@requires_optuna
def test_optuna_nonpositive_mean_and_selected_baselines_keep_absolute_defense():
    report = audit_optuna_study(study([-3.0, -2.0, -1.0]), top_n=3,
                               holdout_evaluate=lambda p: [-3.2, -2.2, -1.2][p["x"]])
    assert report.relative_degradation is None and report.selected_relative_degradation is None
    assert report.relative_degradation_status == "UNDEFINED_NONPOSITIVE_TRAIN_BASELINE"
    assert report.selected_relative_degradation_status == "UNDEFINED_NONPOSITIVE_TRAIN_BASELINE"
    assert report.deployment_authorized


def test_reconstructed_score_or_direction_cannot_keep_stale_degradation_approval():
    unchanged = gate_scores([0.7, 0.8, 0.9], [0.7, 0.8, 0.9])
    assert unchanged.deployment_authorized
    assert not replace(unchanged, holdout_scores=(0.01, 0.02, 0.03), deployment_authorized=True).deployment_authorized
    improved = gate_scores([1, 2, 3], [101, 102, 103])
    assert improved.deployment_authorized
    assert not replace(improved, direction="minimize", deployment_authorized=True).deployment_authorized
    assert not replace(unchanged, relative_degradation=0.99, deployment_authorized=True).deployment_authorized


@requires_optuna
def test_reconstructed_selected_candidate_cannot_keep_another_candidates_certificate():
    report = audit_optuna_study(study([1.0, 2.0, 3.0]), top_n=3,
                               holdout_evaluate=lambda p: [1.0, 2.0, 3.0][p["x"]])
    assert report.gated_best is not None and report.deployment_authorized
    collapsed = replace(report.gated_best, holdout_value=0.01)
    assert not replace(report, gated_best=collapsed, deployment_authorized=True).deployment_authorized
    assert not replace(report, gated_best=collapsed, best_any=collapsed,
                       deployment_authorized=True).deployment_authorized
