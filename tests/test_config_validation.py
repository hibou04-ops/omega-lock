# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P1: configs must reject invalid values at construction time.

The audit framework's whole point is that mistakes surface BEFORE they
corrupt an artifact. ``n_trials=0`` would have produced a P2 run with
no candidates evaluated yet still return a "PASS" status — silent
success on no work.

Each test passes a single invalid value and asserts ValueError. Default
constructions must still succeed unchanged (backward compat).
"""
from __future__ import annotations

import pytest

from omega_lock import IterativeConfig, P1Config
from omega_lock.p2_tpe import P2Config


# ---------------------------------------------------------------------------
# P1Config
# ---------------------------------------------------------------------------


def test_p1_config_default_construction_succeeds():
    P1Config()  # must not raise


@pytest.mark.parametrize(
    "kwargs,err",
    [
        ({"unlock_k": 0}, "unlock_k must be >= 1"),
        ({"unlock_k": -1}, "unlock_k must be >= 1"),
        ({"grid_points_per_axis": 1}, "grid_points_per_axis must be >= 2"),
        ({"grid_points_per_axis": 0}, "grid_points_per_axis must be >= 2"),
        ({"walk_forward_top_n": 1}, "walk_forward_top_n must be >= 2"),
        ({"walk_forward_top_n": 0}, "walk_forward_top_n must be >= 2"),
        ({"trade_ratio_scale": 0.0}, "trade_ratio_scale must be > 0"),
        ({"trade_ratio_scale": -1.0}, "trade_ratio_scale must be > 0"),
        ({"zoom_rounds": 0}, "zoom_rounds must be >= 1"),
        ({"zoom_factor": 0.0}, "zoom_factor must be in"),
        ({"zoom_factor": 1.0}, "zoom_factor must be in"),
        ({"zoom_factor": 1.5}, "zoom_factor must be in"),
        ({"constraint_policy": "made_up"}, "constraint_policy must be one of"),
        ({"holdout_mode": "block_everything"}, "holdout_mode must be"),
    ],
)
def test_p1_config_rejects_invalid(kwargs, err):
    with pytest.raises(ValueError, match=err):
        P1Config(**kwargs)


def test_p1_config_accepts_valid_constraint_policies():
    for policy in ("record", "prefer_feasible", "hard_fail"):
        P1Config(constraint_policy=policy)  # should not raise


def test_p1_config_accepts_valid_holdout_modes():
    for mode in ("evidence_only", "gate"):
        P1Config(holdout_mode=mode)


# ---------------------------------------------------------------------------
# P2Config
# ---------------------------------------------------------------------------


def test_p2_config_default_construction_succeeds():
    P2Config()


@pytest.mark.parametrize(
    "kwargs,err",
    [
        # n_trials=0 is the silent-success case the reviewer flagged: TPE
        # objective never runs, tpe_best_gp stays None, KC checks operate
        # on baseline-only state and could report PASS.
        ({"n_trials": 0}, "n_trials must be >= 1"),
        ({"n_trials": -5}, "n_trials must be >= 1"),
        ({"unlock_k": 0}, "unlock_k must be >= 1"),
        ({"walk_forward_top_n": 1}, "walk_forward_top_n must be >= 2"),
        ({"trade_ratio_scale": 0.0}, "trade_ratio_scale must be > 0"),
        ({"trade_ratio_scale": -0.01}, "trade_ratio_scale must be > 0"),
    ],
)
def test_p2_config_rejects_invalid(kwargs, err):
    with pytest.raises(ValueError, match=err):
        P2Config(**kwargs)


# ---------------------------------------------------------------------------
# IterativeConfig
# ---------------------------------------------------------------------------


def test_iterative_config_default_construction_succeeds():
    IterativeConfig()


@pytest.mark.parametrize(
    "kwargs,err",
    [
        ({"rounds": 0}, "rounds must be >= 1"),
        ({"per_round_unlock_k": 0}, "per_round_unlock_k must be >= 1"),
        ({"grid_points_per_axis": 1}, "grid_points_per_axis must be >= 2"),
        ({"walk_forward_top_n": 0}, "walk_forward_top_n must be >= 2"),
        ({"trade_ratio_scale": 0.0}, "trade_ratio_scale must be > 0"),
        ({"zoom_rounds": 0}, "zoom_rounds must be >= 1"),
        ({"zoom_factor": 0.0}, "zoom_factor must be in"),
        ({"zoom_factor": 1.0}, "zoom_factor must be in"),
    ],
)
def test_iterative_config_rejects_invalid(kwargs, err):
    with pytest.raises(ValueError, match=err):
        IterativeConfig(**kwargs)
