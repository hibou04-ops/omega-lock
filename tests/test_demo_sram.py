"""Smoke tests for the SRAM bitcell demo. Kept minimal — the physics is a
surrogate, not something to pin down numerically. We test shape, not numbers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make the examples/ directory importable so `omega_lock_demos.sram` resolves.
_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from omega_lock_demos.sram import (
    BitcellTarget, Corner, PVT_CORNERS, DEMO_CONSTRAINTS,
    eval_corner, read_snm,
)


def _neutral_params():
    return {"vdd": 0.90, "vth_n": 0.35, "vth_p": 0.35,
            "w_ratio_pd": 2.0, "w_ratio_pu": 1.0, "l_channel": 40.0}


def test_param_space_has_six_specs_with_valid_bounds():
    specs = BitcellTarget().param_space()
    assert len(specs) == 6
    for s in specs:
        assert s.low is not None and s.high is not None
        assert s.low <= s.neutral <= s.high
        assert s.low < s.high


def test_evaluate_returns_finite_fitness_at_neutral():
    t = BitcellTarget()
    r = t.evaluate(_neutral_params())
    assert math.isfinite(r.fitness)
    assert "read_snm_mv_worst" in r.metadata
    assert "write_margin_mv_worst" in r.metadata
    assert "leakage_na_worst" in r.metadata
    assert "per_corner" in r.metadata
    assert len(r.metadata["per_corner"]) == 5


def test_all_three_constraints_evaluable_without_raising():
    t = BitcellTarget()
    r = t.evaluate(_neutral_params())
    for c in DEMO_CONSTRAINTS:
        ok = c.fn(_neutral_params(), r)
        assert isinstance(ok, bool)


def test_hot_corner_leakage_exceeds_cold_corner_leakage():
    """SS (T=398K) must leak more than FF (T=233K)."""
    p = _neutral_params()
    ff = eval_corner(p, Corner("FF", 233.0, 1.0, 1.0))
    ss = eval_corner(p, Corner("SS", 398.0, 1.0, 1.0))
    assert ss["leakage_na"] > ff["leakage_na"]


def test_higher_vdd_improves_snm_when_vth_fixed():
    low  = read_snm(vdd=0.7, vth_n=0.35, beta_ratio=2.0)
    high = read_snm(vdd=1.1, vth_n=0.35, beta_ratio=2.0)
    assert high > low


def test_worst_case_aggregation_picks_min_for_margins_and_max_for_leakage():
    t = BitcellTarget()
    r = t.evaluate(_neutral_params())
    per_corner = r.metadata["per_corner"]
    assert r.metadata["read_snm_mv_worst"]     == min(pc["read_snm_mv"]     for pc in per_corner)
    assert r.metadata["write_margin_mv_worst"] == min(pc["write_margin_mv"] for pc in per_corner)
    assert r.metadata["leakage_na_worst"]      == max(pc["leakage_na"]      for pc in per_corner)


def test_demo_runs_end_to_end_quickly():
    """3 corners, 3-pt grid, unlock_k=2 — must finish in under 2 seconds."""
    import time
    from omega_lock import run_p1, P1Config
    from omega_lock.audit import AuditingTarget, make_report

    t = BitcellTarget(corners=PVT_CORNERS[:3])
    wrapped = AuditingTarget(t, constraints=DEMO_CONSTRAINTS)
    t0 = time.time()
    run_p1(
        train_target=wrapped,
        config=P1Config(unlock_k=2, grid_points_per_axis=3,
                        stress_verbose=False, grid_verbose=False),
    )
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"demo too slow: {elapsed:.2f}s"
    report = make_report(wrapped, method="test_demo")
    assert report.n_total > 0
