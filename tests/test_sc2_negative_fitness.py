# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P2: SC-2 ratio interpretation under negative fitness.

The SC-2 metric (grid top-quartile / random top-quartile) was
documented to assume non-negative fitness — but the code returned a
ratio without any signal that the assumption had been violated.

A trap case: both top-quartiles negative. ``-2 / -4 = 0.5``, which
fails the >= 1.5 threshold and SC-2 reports non-pass. Looks fine.
But the actual story is that random was *worse* (more negative) than
grid, i.e. grid IS doing better in absolute terms — yet the ratio
makes it look the other way. Conversely ``-6 / -4 = 1.5`` would pass
SC-2 even though grid is *more negative* (worse) than random.

The reviewer-recommended fix: surface ``sc2_warning`` when either
top-quartile is negative so the artifact reader knows the ratio's
interpretation is unstable. ``sc2_assumption`` is also recorded
unconditionally so the reader doesn't have to remember the contract.
"""
from __future__ import annotations

from omega_lock.grid import GridPoint
from omega_lock.random_search import compare_to_grid
from omega_lock.target import EvalResult


def _pts(values: list[float]) -> list[GridPoint]:
    return [
        GridPoint(idx=i, unlocked={}, params={},
                  result=EvalResult(fitness=v, n_trials=0))
        for i, v in enumerate(values)
    ]


def test_sc2_assumption_always_recorded():
    """The contract is documented unconditionally — readers don't have
    to remember it from the docstring."""
    report = compare_to_grid(_pts([1.0, 2.0, 3.0, 4.0]),
                             _pts([0.5, 1.0, 1.5, 2.0]))
    assert report["sc2_assumption"] == (
        "fitness_nonnegative_required_for_ratio_interpretation"
    )


def test_sc2_warning_absent_when_both_quartiles_nonneg():
    """Happy path — no negative fitness, no warning."""
    report = compare_to_grid(_pts([1.0, 2.0, 3.0, 4.0]),
                             _pts([0.5, 1.0, 1.5, 2.0]))
    assert "sc2_warning" not in report


def test_sc2_warning_when_grid_top_quartile_negative():
    report = compare_to_grid(_pts([-4.0, -3.0, -2.0, -1.0]),
                             _pts([1.0, 2.0, 3.0, 4.0]))
    assert report["sc2_warning"] == "negative_top_quartile_ratio_unstable"


def test_sc2_warning_when_random_top_quartile_negative():
    report = compare_to_grid(_pts([1.0, 2.0, 3.0, 4.0]),
                             _pts([-4.0, -3.0, -2.0, -1.0]))
    assert report["sc2_warning"] == "negative_top_quartile_ratio_unstable"


def test_sc2_warning_when_both_quartiles_negative():
    """The trap case: ratio looks reasonable (0.5) but the underlying
    comparison is meaningless because both are negative."""
    report = compare_to_grid(_pts([-2.0] * 4), _pts([-4.0] * 4))
    assert "sc2_warning" in report
    # Ratio computed normally (unchanged behaviour); warning is the
    # only NEW signal:
    assert report["ratio"] == 0.5


def test_sc2_warning_distinguishes_from_inf_path():
    """When random=0 and grid is positive, ratio=+inf — warning should
    NOT fire because the inputs aren't negative."""
    report = compare_to_grid(_pts([1.0] * 4), _pts([0.0] * 4))
    assert report["ratio"] == float("inf")
    assert "sc2_warning" not in report
