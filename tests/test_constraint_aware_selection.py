"""Tests for P1Config.constraint_policy — constraint-aware grid_best selection.

Verifies that AuditingTarget surfaces constraint status into result.metadata
and that orchestrator.run_p1 honors the policy when picking grid_best.
"""
from __future__ import annotations

from omega_lock.audit import AuditingTarget, Constraint
from omega_lock.kill_criteria import KCThresholds
from omega_lock.orchestrator import P1Config, run_p1
from omega_lock.target import EvalResult, ParamSpec


class _BiasedTarget:
    """3-param target where the global fitness max is at (a=2, b=2, c=0)
    but a constraint forbids a > 1.0. `a` dominates fitness so KC-2 stress
    differentiation passes; `c` is a near-flat decoy axis.
    """

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="a", dtype="float", neutral=0.0, low=-2.0, high=2.0),
            ParamSpec(name="b", dtype="float", neutral=0.0, low=-2.0, high=4.0),
            ParamSpec(name="c", dtype="float", neutral=0.0, low=-1.0, high=1.0),
        ]

    def evaluate(self, params: dict) -> EvalResult:
        a, b, c = params["a"], params["b"], params["c"]
        # `a` dominates (factor 5x), `c` is near-flat decoy.
        fit = 1.0 - 5 * (a - 2.0) ** 2 / 10.0 - (b - 2.0) ** 2 / 10.0 - 0.001 * c**2
        return EvalResult(fitness=fit, n_trials=100, metadata={})


# Loose KC thresholds so toy targets don't trip KC-2/KC-3 unrelated to the
# constraint-policy behavior under test.
_LOOSE_KC = KCThresholds(gini_min=0.05, top_bot_ratio_min=1.1, trade_count_min=1)


def _make_train(constraints):
    return AuditingTarget(_BiasedTarget(), constraints=constraints)


def _cfg(policy: str = "record") -> P1Config:
    return P1Config(
        unlock_k=2,
        grid_points_per_axis=5,
        kc_thresholds=_LOOSE_KC,
        stress_verbose=False,
        grid_verbose=False,
        constraint_policy=policy,
    )


def test_metadata_surfaces_constraint_status():
    c_fail = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "a must be <=1")
    wrapped = _make_train([c_fail])
    result = wrapped.evaluate({"a": 1.5, "b": 2.0, "c": 0.0})
    assert result.metadata["_constraints_failed"] == ("a_le_1",)
    assert result.metadata["_constraints_passed"] == ()

    result_ok = wrapped.evaluate({"a": 0.5, "b": 2.0, "c": 0.0})
    assert result_ok.metadata["_constraints_failed"] == ()
    assert result_ok.metadata["_constraints_passed"] == ("a_le_1",)


def test_record_policy_picks_raw_max_default():
    """Default policy ignores constraints — raw fitness-max wins even when infeasible."""
    c = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "")
    train = _make_train([c])
    result = run_p1(train_target=train, config=_cfg("record"))
    # Global peak (a=2, b=2) violates constraint but record policy picks it.
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["a"] > 1.0


def test_prefer_feasible_picks_constraint_respecting_max():
    """prefer_feasible filters out constraint violators, picks max among rest."""
    c = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "")
    train = _make_train([c])
    result = run_p1(train_target=train, config=_cfg("prefer_feasible"))
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["a"] <= 1.0


def test_hard_fail_blocks_status_when_no_feasible_candidate():
    """All-violating constraint forces FAIL:CONSTRAINTS under hard_fail."""
    c_always_fail = Constraint("never_ok", lambda p, r: False, "")
    train = _make_train([c_always_fail])
    result = run_p1(train_target=train, config=_cfg("hard_fail"))
    assert "FAIL" in result.status
    assert "CONSTRAINTS" in result.status


def test_hard_fail_passes_when_feasible_candidate_exists():
    """hard_fail does NOT fail when at least one candidate is feasible."""
    c = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "")
    train = _make_train([c])
    result = run_p1(train_target=train, config=_cfg("hard_fail"))
    assert "CONSTRAINTS" not in result.status
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["a"] <= 1.0


def test_no_constraints_means_record_and_prefer_agree():
    """Bare AuditingTarget with no constraints — every policy yields same best."""
    r1 = run_p1(train_target=_make_train([]), config=_cfg("record"))
    r2 = run_p1(train_target=_make_train([]), config=_cfg("prefer_feasible"))
    assert r1.grid_best["unlocked"] == r2.grid_best["unlocked"]
