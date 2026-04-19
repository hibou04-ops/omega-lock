"""Tests for omega_lock.audit — AuditingTarget + AuditReport + scorecard."""
from __future__ import annotations

import json
from itertools import count

import pytest

from omega_lock.audit import (
    AuditedRun,
    AuditingTarget,
    AuditReport,
    Constraint,
    make_report,
    render_scorecard,
)
from omega_lock.grid import GridSearch
from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec


# ── Minimal stub target ────────────────────────────────────────────────────

class StubTarget:
    """Deterministic 2-param target with a quadratic peak at (a=1, b=2)."""
    def __init__(self) -> None:
        self.n_calls = 0
        self.artifacts_to_return: dict = {}

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="a", dtype="float", neutral=0.0, low=-2.0, high=2.0),
            ParamSpec(name="b", dtype="float", neutral=0.0, low=-2.0, high=4.0),
        ]

    def evaluate(self, params: dict) -> EvalResult:
        self.n_calls += 1
        a = params["a"]; b = params["b"]
        fit = 1.0 - ((a - 1.0) ** 2 + (b - 2.0) ** 2) / 10.0
        return EvalResult(
            fitness=fit,
            n_trials=7,
            metadata={"call": self.n_calls, "a_plus_b": a + b},
            artifacts=dict(self.artifacts_to_return),
        )


# ── Constraint behavior ────────────────────────────────────────────────────

def test_constraint_records_pass_and_fail():
    t = StubTarget()
    c_pass = Constraint("always_pass", lambda p, r: True, "trivial")
    c_fail = Constraint("always_fail", lambda p, r: False, "trivial")
    wrapped = AuditingTarget(t, constraints=[c_pass, c_fail])
    wrapped.evaluate({"a": 1.0, "b": 2.0})
    run = wrapped.trail[0]
    assert run.constraints_passed == ("always_pass",)
    assert run.constraints_failed == ("always_fail",)
    assert run.is_feasible is False


def test_constraint_raising_predicate_is_counted_as_fail():
    t = StubTarget()
    def raises(p, r):
        raise ValueError("boom")
    wrapped = AuditingTarget(t, constraints=[Constraint("boomy", raises)])
    wrapped.evaluate({"a": 0.0, "b": 0.0})
    assert wrapped.trail[0].constraints_failed == ("boomy",)


def test_no_constraints_means_always_feasible():
    t = StubTarget()
    wrapped = AuditingTarget(t)
    wrapped.evaluate({"a": 0.0, "b": 0.0})
    assert wrapped.trail[0].is_feasible is True


# ── Protocol impersonation ─────────────────────────────────────────────────

def test_auditingtarget_is_a_calibrable_target():
    t = StubTarget()
    wrapped = AuditingTarget(t)
    assert isinstance(wrapped, CalibrableTarget)
    assert wrapped.param_space() == t.param_space()


def test_auditingtarget_delegates_evaluate():
    t = StubTarget()
    wrapped = AuditingTarget(t)
    r1 = wrapped.evaluate({"a": 1.0, "b": 2.0})
    r2 = t.evaluate({"a": 1.0, "b": 2.0})  # direct call = 2nd invocation
    # same params, same fitness (target is deterministic re: params)
    assert r1.fitness == pytest.approx(r2.fitness)
    # but n_calls counter increments via wrapper too
    assert t.n_calls == 2


# ── Phase / round / role tracking ──────────────────────────────────────────

def test_phase_and_round_stick_until_changed():
    t = StubTarget()
    w = AuditingTarget(t)
    w.set_phase("stress"); w.evaluate({"a": 0.0, "b": 0.0})
    w.set_phase("search"); w.evaluate({"a": 0.5, "b": 0.5})
    w.set_round(1);        w.evaluate({"a": 1.0, "b": 1.0})
    phases = [r.phase for r in w.trail]
    rounds = [r.round_index for r in w.trail]
    assert phases == ["stress", "search", "search"]
    assert rounds == [0, 0, 1]


def test_call_index_is_monotonic():
    t = StubTarget()
    w = AuditingTarget(t)
    for _ in range(5):
        w.evaluate({"a": 0.0, "b": 0.0})
    indices = [r.call_index for r in w.trail]
    assert indices == [0, 1, 2, 3, 4]


def test_target_role_is_recorded():
    t = StubTarget()
    w_train = AuditingTarget(t, target_role="train")
    w_test = AuditingTarget(t, target_role="test")
    w_train.evaluate({"a": 0.0, "b": 0.0})
    w_test.evaluate({"a": 0.0, "b": 0.0})
    assert w_train.trail[0].target_role == "train"
    assert w_test.trail[0].target_role == "test"


# ── Shared trail + counter ─────────────────────────────────────────────────

def test_shared_trail_keeps_global_order():
    t1 = StubTarget(); t2 = StubTarget()
    trail: list[AuditedRun] = []
    cnt = count(0)
    w1 = AuditingTarget(t1, target_role="train", shared_trail=trail, shared_counter=cnt)
    w2 = AuditingTarget(t2, target_role="test",  shared_trail=trail, shared_counter=cnt)
    w1.evaluate({"a": 0.0, "b": 0.0})
    w2.evaluate({"a": 0.5, "b": 0.5})
    w1.evaluate({"a": 1.0, "b": 1.0})
    assert len(trail) == 3
    assert [r.call_index for r in trail] == [0, 1, 2]
    assert [r.target_role for r in trail] == ["train", "test", "train"]


# ── Artifacts policy ───────────────────────────────────────────────────────

def test_artifacts_dropped_by_default():
    t = StubTarget()
    t.artifacts_to_return = {"big_blob": [0] * 1000}
    w = AuditingTarget(t)
    w.evaluate({"a": 0.0, "b": 0.0})
    assert "_artifacts" not in w.trail[0].metadata


def test_artifacts_retained_when_opted_in():
    t = StubTarget()
    t.artifacts_to_return = {"trace": [1, 2, 3]}
    w = AuditingTarget(t, retain_artifacts=True)
    w.evaluate({"a": 0.0, "b": 0.0})
    assert w.trail[0].metadata["_artifacts"] == {"trace": [1, 2, 3]}


# ── AuditReport summary ────────────────────────────────────────────────────

def test_report_summary_counts_feasibility():
    t = StubTarget()
    c = Constraint("a_positive", lambda p, r: p["a"] > 0, "a must be > 0")
    w = AuditingTarget(t, constraints=[c])
    for a in (-1.0, 0.5, 1.0):
        w.evaluate({"a": a, "b": 0.0})
    report = make_report(w, method="stub")
    assert report.n_total == 3
    assert report.n_feasible == 2
    assert report.feasibility_rate == pytest.approx(2 / 3)


def test_report_best_feasible_prefers_feasible_even_if_lower_fitness():
    t = StubTarget()
    c = Constraint("a_le_0", lambda p, r: p["a"] <= 0, "a must be non-positive")
    w = AuditingTarget(t, constraints=[c])
    # a=1 is the true peak; a=-1 is feasible but lower fitness.
    w.evaluate({"a": 1.0, "b": 2.0})   # infeasible, highest fitness
    w.evaluate({"a": -1.0, "b": 2.0})  # feasible, lower fitness
    report = make_report(w, method="stub")
    assert report.best_any.params["a"] == pytest.approx(1.0)
    assert report.best_feasible.params["a"] == pytest.approx(-1.0)


def test_report_by_phase_by_role_by_round_filters():
    t = StubTarget()
    w_train = AuditingTarget(t, target_role="train")
    w_train.set_phase("stress"); w_train.evaluate({"a": 0.0, "b": 0.0})
    w_train.set_phase("search"); w_train.set_round(1); w_train.evaluate({"a": 1.0, "b": 1.0})
    rep = make_report(w_train, method="stub")
    assert len(rep.by_phase("stress")) == 1
    assert len(rep.by_phase("search")) == 1
    assert len(rep.by_role("train")) == 2
    assert len(rep.by_round(1)) == 1
    assert len(rep.by_round(0)) == 1


# ── JSON roundtrip ─────────────────────────────────────────────────────────

def test_report_json_roundtrip_preserves_runs_and_structure():
    t = StubTarget()
    c = Constraint("pos", lambda p, r: r.fitness > 0, "positive fit")
    w = AuditingTarget(t, constraints=[c])
    w.set_phase("baseline"); w.evaluate({"a": 0.0, "b": 0.0})
    w.set_phase("search");   w.evaluate({"a": 1.0, "b": 2.0})
    report = make_report(w, method="stub", seed=42, stress_ranking=[("a", 0.9), ("b", 0.3)])

    js = report.to_json()
    d = json.loads(js)
    assert d["method"] == "stub"
    assert d["seed"] == 42
    assert len(d["runs"]) == 2
    assert d["stress_ranking"] == [["a", 0.9], ["b", 0.3]]

    rehydrated = AuditReport.from_json(js)
    assert rehydrated.method == "stub"
    assert rehydrated.seed == 42
    assert rehydrated.n_total == 2
    assert rehydrated.runs[0].phase == "baseline"
    assert rehydrated.runs[1].phase == "search"
    assert rehydrated.stress_ranking == (("a", 0.9), ("b", 0.3))


def test_rehydrated_constraint_predicate_raises_on_call():
    t = StubTarget()
    w = AuditingTarget(t, constraints=[Constraint("c", lambda p, r: True)])
    w.evaluate({"a": 0.0, "b": 0.0})
    report = make_report(w, method="stub")
    rehydrated = AuditReport.from_json(report.to_json())
    # Predicate sentinel should explicitly refuse execution.
    with pytest.raises(RuntimeError, match="not serialized"):
        rehydrated.constraints[0].fn({"a": 0.0, "b": 0.0}, EvalResult(fitness=0.0))


# ── Scorecard ──────────────────────────────────────────────────────────────

def test_render_scorecard_emits_key_sections():
    t = StubTarget()
    c = Constraint("fit_gt_half", lambda p, r: r.fitness > 0.5, "fitness > 0.5")
    w = AuditingTarget(t, constraints=[c])
    for a, b in [(0.0, 0.0), (1.0, 2.0), (-2.0, -2.0)]:
        w.evaluate({"a": a, "b": b})
    report = make_report(w, method="stub", seed=7,
                         stress_ranking=[("a", 1.2), ("b", 0.4)])
    s = render_scorecard(report)
    # spot-check required sections appear
    assert "omega-lock audit report" in s
    assert "Method:" in s and "stub" in s
    assert "Seed:" in s and "7" in s
    assert "Total runs:" in s and "3" in s
    assert "Feasible:" in s
    assert "fit_gt_half" in s
    assert "Best feasible" in s or "none — all runs violated" in s
    assert "Stress ranking" in s
    assert "Trail breakdown" in s


def test_render_scorecard_handles_empty_constraints_gracefully():
    t = StubTarget()
    w = AuditingTarget(t)
    w.evaluate({"a": 0.0, "b": 0.0})
    s = render_scorecard(make_report(w, method="stub"))
    assert "Total runs:" in s
    assert "Constraints:" not in s  # no constraints -> section omitted


# ── Integration with GridSearch ────────────────────────────────────────────

def test_audit_integrates_with_grid_search():
    t = StubTarget()
    c = Constraint("b_le_3", lambda p, r: p["b"] <= 3.0, "b <= 3")
    w = AuditingTarget(t, constraints=[c])
    w.set_phase("search")

    gs = GridSearch(target=w, unlocked=["a", "b"], grid_points_per_axis=3, verbose=False)
    results = gs.run(base_params={"a": 0.0, "b": 0.0})

    # 3 × 3 = 9 grid evaluations, each captured
    assert len(results) == 9
    assert len(w.trail) == 9

    # Best feasible != best any (peak at b=2.0 is feasible, but grid covers b up to 4)
    report = make_report(w, method="grid3x3")
    # With range [-2, 4] and 3 points per axis: b ∈ {-2, 1, 4}. So b=4 violates.
    infeasible = [r for r in report.runs if not r.is_feasible]
    assert len(infeasible) > 0
    feasible = [r for r in report.runs if r.is_feasible]
    assert len(feasible) > 0
    # best_feasible must actually satisfy constraint
    bf = report.best_feasible
    assert bf is not None
    assert bf.params["b"] <= 3.0


def test_audit_empty_trail_produces_valid_report():
    t = StubTarget()
    w = AuditingTarget(t)
    report = make_report(w, method="noop")
    assert report.n_total == 0
    assert report.best_feasible is None
    assert report.best_any is None
    # JSON roundtrip still works
    js = report.to_json()
    AuditReport.from_json(js)
