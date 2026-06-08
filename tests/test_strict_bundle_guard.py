# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Strict KC-4a bundle guard (threshold recalibration plan T3 / arch D2).

``KCThresholds.strict()`` raises the KC-4a Pearson floor to 0.5. That lower
false-PASS rate is only earned when the walk-forward gate is fed enough
points (``walk_forward_top_n >= DEFAULT_WALK_FORWARD_TOP_N``). The two-object
strict bundle (KCThresholds value + search input) MUST travel together;
``run_p1`` / ``run_p2_tpe`` raise ValueError at the call site if only one is
present — BEFORE any target evaluation.

These tests cover:
  * the ``assert_strict_bundle`` helper directly (firing / not-firing matrix),
  * the guard wired into ``run_p1`` (fires before any evaluate; safe bundle
    passes the guard), and
  * the guard wired into ``run_p2_tpe`` (fires before ``_require_optuna()``).

No network, no optuna dependency for the firing path.
"""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock import KCThresholds, P1Config, run_p1
from omega_lock.p2_tpe import P2Config, run_p2_tpe
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.walk_forward import (
    DEFAULT_WALK_FORWARD_TOP_N,
    assert_strict_bundle,
)


class _TripwireTarget:
    """Valid param_space, but evaluate() raises a sentinel. If the strict
    guard fires, no evaluate() happens and we never see the sentinel; if the
    guard passes (safe bundle), the pipeline reaches baseline evaluate and
    raises the sentinel — proving the guard let a safe bundle through.
    """

    class Sentinel(RuntimeError):
        pass

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec(name="b", dtype="float", low=0.0, high=1.0, neutral=0.5),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        raise _TripwireTarget.Sentinel("evaluate reached — guard did not fire")


# ---------------------------------------------------------------------------
# Unit: assert_strict_bundle helper
# ---------------------------------------------------------------------------


def test_helper_fires_on_strict_pearson_with_small_top_n():
    with pytest.raises(ValueError, match="strict KC-4a bundle is half-applied"):
        assert_strict_bundle(KCThresholds.strict(), walk_forward_top_n=10)


def test_helper_fires_at_top_n_just_below_default():
    with pytest.raises(ValueError):
        assert_strict_bundle(
            KCThresholds.strict(),
            walk_forward_top_n=DEFAULT_WALK_FORWARD_TOP_N - 1,
        )


def test_helper_silent_on_strict_pearson_with_default_top_n():
    # strict() + the default sample size is the intended bundle — no raise.
    assert_strict_bundle(
        KCThresholds.strict(), walk_forward_top_n=DEFAULT_WALK_FORWARD_TOP_N
    ) is None


def test_helper_silent_on_default_thresholds_even_with_small_top_n():
    # Non-strict thresholds (bare default pearson_min=0.3) never trip the
    # guard, regardless of walk_forward_top_n.
    assert_strict_bundle(KCThresholds(), walk_forward_top_n=2) is None


def test_helper_silent_at_pearson_just_below_strict():
    # 0.49 < 0.5 -> not "strict", guard stays silent even with tiny top_n.
    assert_strict_bundle(
        KCThresholds(pearson_min=0.49), walk_forward_top_n=2
    ) is None


def test_helper_fires_at_pearson_exactly_half():
    # Boundary: pearson_min == 0.5 counts as strict.
    with pytest.raises(ValueError):
        assert_strict_bundle(
            KCThresholds(pearson_min=0.5), walk_forward_top_n=10
        )


# ---------------------------------------------------------------------------
# run_p1: guard wired in, fires before any evaluation
# ---------------------------------------------------------------------------


def test_run_p1_raises_on_half_applied_strict_bundle():
    cfg = P1Config(
        kc_thresholds=KCThresholds.strict(),
        walk_forward_top_n=10,  # < DEFAULT_WALK_FORWARD_TOP_N
    )
    with pytest.raises(ValueError, match="strict KC-4a bundle is half-applied"):
        run_p1(_TripwireTarget(), config=cfg)


def test_run_p1_guard_fires_before_any_evaluate():
    # The guard must fire BEFORE baseline evaluate. If it didn't, the
    # tripwire's Sentinel would surface instead of the ValueError.
    cfg = P1Config(
        kc_thresholds=KCThresholds.strict(),
        walk_forward_top_n=10,
    )
    with pytest.raises(ValueError):
        run_p1(_TripwireTarget(), config=cfg)
    # (If evaluate had run, _TripwireTarget.Sentinel would have been raised.)


def test_run_p1_safe_strict_bundle_passes_guard():
    # strict() + default top_n is the legitimate bundle. The guard lets it
    # through, so the pipeline proceeds to baseline evaluate and hits the
    # tripwire Sentinel — proving the guard did NOT block the safe bundle.
    cfg = P1Config(
        kc_thresholds=KCThresholds.strict(),
        walk_forward_top_n=DEFAULT_WALK_FORWARD_TOP_N,
    )
    with pytest.raises(_TripwireTarget.Sentinel):
        run_p1(_TripwireTarget(), config=cfg)


def test_run_p1_default_config_passes_guard():
    # Bare default config (pearson 0.3, top_n=20) must never trip the guard.
    with pytest.raises(_TripwireTarget.Sentinel):
        run_p1(_TripwireTarget(), config=P1Config())


# ---------------------------------------------------------------------------
# run_p2_tpe: guard fires before _require_optuna() (independent of optuna)
# ---------------------------------------------------------------------------


def test_run_p2_raises_on_half_applied_strict_bundle():
    cfg = P2Config(
        kc_thresholds=KCThresholds.strict(),
        walk_forward_top_n=10,
    )
    # Must raise ValueError (the bundle guard), NOT ImportError — the guard
    # is placed before _require_optuna() so it fires whether or not optuna
    # is installed.
    with pytest.raises(ValueError, match="strict KC-4a bundle is half-applied"):
        run_p2_tpe(_TripwireTarget(), config=cfg)
