"""Tests for orchestrator auto phase tracking.

run_p1 should call set_phase() on each target as the pipeline transitions
between baseline / stress / grid / walk_forward / hybrid / holdout, so
the audit trail records the right phase per call without the user having
to wire it manually.
"""
from __future__ import annotations

from omega_lock.audit import AuditingTarget
from omega_lock.kill_criteria import KCThresholds
from omega_lock.orchestrator import P1Config, run_p1
from omega_lock.target import EvalResult, ParamSpec


class _T:
    """3-param target where `a` dominates so KC-2 stress passes."""

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="a", dtype="float", neutral=0.0, low=-2.0, high=2.0),
            ParamSpec(name="b", dtype="float", neutral=0.0, low=-2.0, high=2.0),
            ParamSpec(name="c", dtype="float", neutral=0.0, low=-1.0, high=1.0),
        ]

    def evaluate(self, p: dict) -> EvalResult:
        a, b, c = p["a"], p["b"], p["c"]
        fit = 1.0 - 5 * (a - 1.0) ** 2 / 10.0 - (b - 1.0) ** 2 / 10.0 - 0.001 * c**2
        return EvalResult(fitness=fit, n_trials=100, metadata={})


_LOOSE = KCThresholds(gini_min=0.05, top_bot_ratio_min=1.1, trade_count_min=1)
_CFG = P1Config(
    unlock_k=2,
    grid_points_per_axis=3,
    kc_thresholds=_LOOSE,
    stress_verbose=False,
    grid_verbose=False,
)


def _phases(trail) -> set[str]:
    return {run.phase for run in trail}


def test_train_only_records_baseline_stress_grid_phases():
    train = AuditingTarget(_T(), target_role="train")
    run_p1(train_target=train, config=_CFG)
    assert _phases(train.trail) == {"baseline", "stress", "grid"}


def test_baseline_phase_marks_only_first_call():
    """The very first evaluate() — neutral defaults — should be tagged 'baseline'."""
    train = AuditingTarget(_T(), target_role="train")
    run_p1(train_target=train, config=_CFG)
    baseline_runs = [r for r in train.trail if r.phase == "baseline"]
    assert len(baseline_runs) == 1


def test_test_target_gets_walk_forward_phase():
    train = AuditingTarget(_T(), target_role="train")
    test = AuditingTarget(_T(), target_role="test")
    run_p1(train_target=train, test_target=test, config=_CFG)
    assert _phases(test.trail) == {"walk_forward"}
    # Train trail should NOT have walk_forward — that's test's phase.
    assert "walk_forward" not in _phases(train.trail)


def test_holdout_target_gets_holdout_phase():
    train = AuditingTarget(_T(), target_role="train")
    holdout = AuditingTarget(_T(), target_role="holdout")
    run_p1(train_target=train, holdout_target=holdout, config=_CFG)
    assert _phases(holdout.trail) == {"holdout"}
    # Holdout is single-shot.
    assert len(holdout.trail) == 1


def test_validation_target_gets_hybrid_phase():
    train = AuditingTarget(_T(), target_role="train")
    val = AuditingTarget(_T(), target_role="validation")
    run_p1(train_target=train, validation_target=val, config=_CFG)
    assert "hybrid" in _phases(val.trail)


def test_bare_target_without_set_phase_does_not_crash():
    """Non-AuditingTarget targets just don't get tracked — no exception."""
    bare = _T()
    result = run_p1(train_target=bare, config=_CFG)
    assert result.status.startswith(("PASS", "FAIL"))  # pipeline ran
