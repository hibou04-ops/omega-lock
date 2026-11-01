# SPDX-License-Identifier: Apache-2.0
"""Boundary tests for independent calibration evidence helpers."""

from math import log, sqrt

import pytest

from omega_lock.holdout_defense import (
    classify_calibration_verdict,
    compute_deflated_threshold,
)


def test_deflated_threshold_matches_requested_formula():
    expected = 0.2 + 0.05 * sqrt(1 + log(10))
    assert compute_deflated_threshold(0.2, 0.05, 10) == pytest.approx(expected)


@pytest.mark.parametrize("trial_count", [-5, 0, 1])
def test_deflated_threshold_clamps_trial_count_to_one(trial_count):
    assert compute_deflated_threshold(0.2, 0.05, trial_count) == pytest.approx(0.25)


def test_more_trials_never_reduce_threshold():
    thresholds = [compute_deflated_threshold(0.2, 0.05, n) for n in (1, 2, 10, 100)]
    assert all(left < right for left, right in zip(thresholds, thresholds[1:]))


def test_zero_noise_keeps_base_threshold():
    assert compute_deflated_threshold(-0.1, 0.0, 100) == -0.1


@pytest.mark.parametrize(
    ("delta", "noise_floor", "holdout", "expected"),
    [
        (0.2, 0.1, True, "ESTABLISHED"),
        (0.1, 0.1, True, "NOT_ESTABLISHED"),
        (0.05, 0.1, True, "NOT_ESTABLISHED"),
        (-0.2, 0.1, True, "NOT_ESTABLISHED"),
        (0.0, 0.0, True, "NOT_ESTABLISHED"),
        (0.01, 0.0, True, "ESTABLISHED"),
        (0.2, 0.1, False, "M0_HOLD"),
        (0.1, 0.1, False, "M0_HOLD"),
        (-0.2, 0.1, False, "M0_HOLD"),
    ],
)
def test_calibration_verdict_truth_table(delta, noise_floor, holdout, expected):
    assert classify_calibration_verdict(delta, noise_floor, holdout) == expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observation_is_rejected(invalid):
    with pytest.raises(ValueError, match="observed_delta"):
        classify_calibration_verdict(invalid, 0.1, True)


@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_noise_floor_is_rejected(invalid):
    with pytest.raises(ValueError, match="noise_floor_mde"):
        compute_deflated_threshold(0.1, invalid, 1)
    with pytest.raises(ValueError, match="noise_floor_mde"):
        classify_calibration_verdict(0.2, invalid, True)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_base_threshold_is_rejected(invalid):
    with pytest.raises(ValueError, match="base_threshold"):
        compute_deflated_threshold(invalid, 0.1, 1)
