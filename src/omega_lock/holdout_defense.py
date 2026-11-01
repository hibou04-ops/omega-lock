# SPDX-License-Identifier: Apache-2.0
"""Pure calibration evidence checks, independent of orchestration state."""

from __future__ import annotations

from math import isfinite, log, sqrt


def compute_deflated_threshold(
    base_threshold: float, noise_floor_mde: float, trial_count: int
) -> float:
    """Raise a threshold by the noise floor and the number of explored trials.

    Counts below one use one trial, as specified by the logarithmic penalty.
    The caller supplies the measured nonnegative MDE; this helper does not
    establish the independence or statistical power of those measurements.
    """
    if not isfinite(base_threshold):
        raise ValueError("base_threshold must be finite")
    if not isfinite(noise_floor_mde) or noise_floor_mde < 0:
        raise ValueError("noise_floor_mde must be finite and nonnegative")
    return base_threshold + noise_floor_mde * sqrt(1 + log(max(1, trial_count)))


def classify_calibration_verdict(
    observed_delta: float, noise_floor_mde: float, holdout_passed: bool
) -> str:
    """Classify the supplied evidence using a strict improvement boundary.

    A failed holdout takes precedence and yields M0_HOLD regardless of delta.
    With a successful holdout, only delta strictly above the nonnegative
    noise floor yields ESTABLISHED. These labels describe this local rule,
    not an independent proof of generalization.
    """
    if not isfinite(observed_delta):
        raise ValueError("observed_delta must be finite")
    if not isfinite(noise_floor_mde) or noise_floor_mde < 0:
        raise ValueError("noise_floor_mde must be finite and nonnegative")
    if not holdout_passed:
        return "M0_HOLD"
    if observed_delta <= noise_floor_mde:
        return "NOT_ESTABLISHED"
    return "ESTABLISHED"
