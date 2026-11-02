# SPDX-License-Identifier: Apache-2.0
"""Pure calibration evidence checks, independent of orchestration state."""

from __future__ import annotations

from math import isfinite, log, sqrt
from fractions import Fraction
from typing import Sequence, TypedDict


class ScoreDegradation(TypedDict):
    degradation_gap: float | None
    max_degradation_gap: float
    degradation_computable: bool
    degradation_status: str
    degradation_ok: bool
    absolute_degradation_ok: bool
    relative_degradation: float | None
    max_relative_degradation: float
    relative_degradation_status: str
    relative_degradation_ok: bool | None
    relative_degradation_scope: str


def validate_max_degradation_gap(value: float) -> float:
    """Require a finite nonnegative score-unit limit; None cannot disable it."""
    if isinstance(value, bool):
        raise ValueError("max_degradation_gap must be finite and nonnegative")
    try:
        limit = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_degradation_gap must be finite and nonnegative") from exc
    if not isfinite(limit) or limit < 0:
        raise ValueError("max_degradation_gap must be finite and nonnegative")
    return limit


def validate_max_relative_degradation(value: float) -> float:
    """Require a finite nonnegative ratio limit; bool/None cannot disable it."""
    if isinstance(value, bool):
        raise ValueError("max_relative_degradation must be finite and nonnegative")
    try:
        limit = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_relative_degradation must be finite and nonnegative") from exc
    if not isfinite(limit) or limit < 0:
        raise ValueError("max_relative_degradation must be finite and nonnegative")
    return limit


def score_degradation(
    train_scores: Sequence[float], holdout_scores: Sequence[float],
    max_degradation_gap: float = 1.0,
    max_relative_degradation: float = 0.50, *, direction: str = "maximize",
) -> ScoreDegradation:
    """Combine direction-adjusted absolute and positive-train relative bounds.

    Exact rational accumulation avoids overflowing means or losing a small
    represented drop through separately rounded large means. An unrepresentable
    overflow/underflow gap is unmeasured and cannot authorize deployment.
    Compare the exact rational gap to the exact supplied float limit; the
    reported float is a display value and cannot decide a boundary.
    Relative arithmetic uses Fraction(str(value)) on original score means;
    positive TRAIN is the denominator even for minimize objectives. A zero or
    negative TRAIN mean has no defined relative ratio: the absolute gate remains.
    """
    limit = validate_max_degradation_gap(max_degradation_gap)
    relative_limit = validate_max_relative_degradation(max_relative_degradation)
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    sign = -1 if direction == "minimize" else 1
    status = "OK"
    gap: float | None = None
    exact_within_limit = False
    relative: float | None = None
    relative_status = "NOT_COMPUTED"
    relative_ok: bool | None = False
    relative_allows = False
    if not train_scores:
        status = "EMPTY"
    elif len(train_scores) != len(holdout_scores):
        status = "LENGTH_MISMATCH"
    elif not all(isfinite(value) for value in (*train_scores, *holdout_scores)):
        status = "NON_FINITE"
    else:
        exact_gap = sign * (
            sum((Fraction.from_float(float(value)) for value in train_scores), Fraction())
            - sum((Fraction.from_float(float(value)) for value in holdout_scores), Fraction())
        ) / len(train_scores)
        exact_within_limit = exact_gap <= Fraction.from_float(limit)
        try:
            gap = float(exact_gap)
        except OverflowError:
            status = "NON_FINITE_GAP"
        if gap is not None and not isfinite(gap):
            gap, status = None, "NON_FINITE_GAP"
        if gap == 0.0 and exact_gap != 0:
            gap, status = None, "UNDERFLOW_GAP"
        decimal_train = sum((Fraction(str(value)) for value in train_scores), Fraction()) / len(train_scores)
        decimal_holdout = sum((Fraction(str(value)) for value in holdout_scores), Fraction()) / len(holdout_scores)
        if decimal_train <= 0:
            relative_status = "UNDEFINED_NONPOSITIVE_TRAIN_BASELINE"
            relative_ok = None
            relative_allows = True  # explicit non-applicability, not a measured zero
        else:
            exact_relative = sign * (decimal_train - decimal_holdout) / decimal_train
            relative_ok = exact_relative <= Fraction(str(relative_limit))
            relative_status = "OK"
            try:
                relative = float(exact_relative)
            except OverflowError:
                relative_status = "NON_FINITE_RELATIVE_DEGRADATION"
            if relative is not None and not isfinite(relative):
                relative, relative_status = None, "NON_FINITE_RELATIVE_DEGRADATION"
            if relative == 0.0 and exact_relative != 0:
                relative, relative_status = None, "UNDERFLOW_RELATIVE_DEGRADATION"
            relative_allows = relative_status == "OK" and relative_ok
    return {
        "degradation_gap": gap,
        "max_degradation_gap": limit,
        "degradation_computable": status == "OK",
        "degradation_status": status,
        "absolute_degradation_ok": status == "OK" and exact_within_limit,
        "relative_degradation": relative,
        "max_relative_degradation": relative_limit,
        "relative_degradation_status": relative_status,
        "relative_degradation_ok": relative_ok if relative_status in ("OK", "UNDEFINED_NONPOSITIVE_TRAIN_BASELINE") else False,
        "relative_degradation_scope": "DIRECTION_ADJUSTED_DECIMAL_MEAN_GAP_OVER_ORIGINAL_POSITIVE_TRAIN_MEAN",
        "degradation_ok": status == "OK" and exact_within_limit and relative_allows,
    }


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
