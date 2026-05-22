"""Tests for P1Config.constraint_policy — constraint-aware grid_best selection.

Verifies that AuditingTarget surfaces constraint status into result.metadata
and that orchestrator.run_p1 honors the policy when picking grid_best.
"""
from __future__ import annotations

from typing import Any

from omega_lock.audit import AuditingTarget, Constraint, make_report
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


def _cfg(policy: str = "record", **overrides: Any) -> P1Config:
    return P1Config(
        unlock_k=2,
        grid_points_per_axis=5,
        kc_thresholds=_LOOSE_KC,
        stress_verbose=False,
        grid_verbose=False,
        constraint_policy=policy,
        **overrides,
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
    assert any(
        "constraints were recorded but did not gate best-candidate selection" in msg
        for msg in result.warnings
    )


def test_prefer_feasible_picks_constraint_respecting_max():
    """prefer_feasible filters out constraint violators, picks max among rest."""
    c = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "")
    train = _make_train([c])
    result = run_p1(train_target=train, config=_cfg("prefer_feasible"))
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["a"] <= 1.0


def test_audit_report_preserves_raw_best_and_feasible_best_distinction():
    """The audit trail keeps raw optimizer output and feasible selection separate."""
    c = Constraint("a_le_1", lambda p, r: p["a"] <= 1.0, "")
    train = _make_train([c])
    result = run_p1(train_target=train, config=_cfg("prefer_feasible"))
    report = make_report(train, method="contract")

    best_any = report.best_any
    best_feasible = report.best_feasible

    assert best_any is not None
    assert best_feasible is not None
    assert best_any.fitness > best_feasible.fitness
    assert best_any.params["a"] > 1.0
    assert best_any.constraints_failed == ("a_le_1",)
    assert best_feasible.params["a"] <= 1.0
    assert best_feasible.constraints_failed == ()
    assert result.grid_best is not None
    assert result.grid_best["unlocked"]["a"] == best_feasible.params["a"]


def test_hard_fail_blocks_status_when_no_feasible_candidate():
    """All-violating constraint forces FAIL:CONSTRAINTS under hard_fail."""
    c_always_fail = Constraint("never_ok", lambda p, r: False, "")
    train = _make_train([c_always_fail])
    result = run_p1(train_target=train, config=_cfg("hard_fail"))
    assert "FAIL" in result.status
    assert "CONSTRAINTS" in result.status
    assert result.grid_best is not None

    constraints_report = next(k for k in result.kc_reports if k["name"] == "CONSTRAINTS")
    assert constraints_report["status"] == "FAIL"
    assert constraints_report["detail"]["n_candidates"] > 0

    report = make_report(train, method="contract")
    assert report.best_feasible is None
    assert report.best_any is not None
    assert all(run.constraints_failed == ("never_ok",) for run in report.runs)


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
    r1_grid_best = r1.grid_best
    r2_grid_best = r2.grid_best
    assert r1_grid_best is not None
    assert r2_grid_best is not None
    assert r1_grid_best["unlocked"] == r2_grid_best["unlocked"]


def test_sc2_advisory_is_visible_but_never_a_hard_constraint():
    """SC-2 is reported for review but must not enter release-blocking status."""
    result = run_p1(
        train_target=_make_train([]),
        config=_cfg("prefer_feasible", run_sc2_baseline=True),
    )

    sc2 = next(k for k in result.kc_reports if k["name"] == "SC-2")
    hard_fail_names = {k["name"] for k in result.kc_reports if k["status"] == "FAIL"}

    assert sc2["status"] == "ADVISORY"
    assert "sc2_pass" in sc2["detail"]
    assert "SC-2" not in hard_fail_names
    assert "SC-2" not in result.status
