"""Tests for the objective benchmark scorecard (RAGAS-style for calibration)."""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock import (
    BenchmarkSpec, CalibrationMethod, run_benchmark,
    compute_effective_recall, compute_effective_precision,
    compute_param_L2_error, compute_fitness_gap_pct,
    compute_generalization_gap, compute_spearman,
)
from omega_lock.keyholes.phantom import PhantomKeyhole


# ── Metric unit tests ───────────────────────────────────────────────────

def test_effective_recall_full_match():
    assert compute_effective_recall({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_effective_recall_partial():
    assert compute_effective_recall({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)


def test_effective_recall_empty_true_is_1():
    assert compute_effective_recall({"a"}, set()) == 1.0


def test_effective_recall_empty_found_is_0():
    assert compute_effective_recall(set(), {"a", "b"}) == 0.0


def test_effective_precision_full_relevance():
    assert compute_effective_precision({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_effective_precision_with_decoy():
    # found 2 of 3 truly effective, 1 decoy included
    assert compute_effective_precision({"a", "b", "DECOY"}, {"a", "b", "c"}) == pytest.approx(2 / 3)


def test_effective_precision_empty_found_is_0():
    assert compute_effective_precision(set(), {"a"}) == 0.0


def test_param_L2_error_zero_on_exact_match():
    found = {"x": 0.5, "n": 10, "flag": True}
    true = {"x": 0.5, "n": 10, "flag": True}
    ranges = {"x": (0.0, 1.0), "n": (0, 100), "flag": (0, 1)}
    assert compute_param_L2_error(found, true, ranges) == 0.0


def test_param_L2_error_normalized_by_range():
    # x differs by 0.1 on range 1.0 → contributes 0.01 to squared; L2 = 0.1
    found = {"x": 0.6}
    true = {"x": 0.5}
    ranges = {"x": (0.0, 1.0)}
    assert compute_param_L2_error(found, true, ranges) == pytest.approx(0.1, abs=1e-9)


def test_param_L2_error_bool_mismatch_contributes_1():
    found = {"flag": False}
    true = {"flag": True}
    ranges = {"flag": (0, 1)}
    assert compute_param_L2_error(found, true, ranges) == pytest.approx(1.0, abs=1e-9)


def test_fitness_gap_pct_zero_when_found_equals_optimum():
    assert compute_fitness_gap_pct(found_fitness=10.0, true_optimum_fitness=10.0) == 0.0


def test_fitness_gap_pct_positive_when_below_optimum():
    # Found 7, optimum 10 → 30% gap remaining
    assert compute_fitness_gap_pct(found_fitness=7.0, true_optimum_fitness=10.0) == pytest.approx(30.0)


def test_fitness_gap_pct_negative_when_exceeds_optimum():
    # Found 12, optimum 10 → -20% (found BEAT the reference)
    assert compute_fitness_gap_pct(found_fitness=12.0, true_optimum_fitness=10.0) == pytest.approx(-20.0)


def test_generalization_gap_zero_when_no_test():
    assert compute_generalization_gap(train_fitness=5.0, test_fitness=None) == 0.0


def test_generalization_gap_normalized():
    # train=10, test=7 → gap = 0.3
    assert compute_generalization_gap(train_fitness=10.0, test_fitness=7.0) == pytest.approx(0.3)


def test_spearman_identical_is_1():
    assert compute_spearman(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_spearman_reversed_is_negative_1():
    assert compute_spearman(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)


def test_spearman_returns_none_on_insufficient_overlap():
    assert compute_spearman(["a", "b"], ["x", "y"]) is None


def test_spearman_uses_intersection_only():
    # Common: {a, b, c}. Order in both is a<b<c → ρ=1
    assert compute_spearman(["a", "b", "c", "x"], ["a", "b", "c", "y"]) == pytest.approx(1.0)


# ── End-to-end benchmark run ────────────────────────────────────────────

def _minimal_runner(target: Any, seed: int) -> dict[str, Any]:
    """Toy runner — returns the true optimum (perfect-score calibrator).
    Validates benchmark metrics with a known correct result."""
    from omega_lock.params import neutral_defaults
    specs = target.param_space()
    base = neutral_defaults(specs)
    base.update(target.true_optimum_params())
    result = target.evaluate(base)
    return {
        "found_params": base,
        "found_fitness": result.fitness,
        "train_fitness": result.fitness,
        "test_fitness": None,
        "unlocked": list(target.true_effective_params()),
        "stress_ranking": target.true_importance_ranking(),
        "status": "PASS",
        "n_evaluations": 1,
        "walltime_s": 0.0,
    }


def test_benchmark_perfect_runner_scores_perfect_metrics():
    spec = BenchmarkSpec(
        keyhole_name="PhantomKeyhole",
        keyhole_factory=PhantomKeyhole,
        seeds=[42, 7],
    )
    method = CalibrationMethod(name="oracle", runner=_minimal_runner)
    report = run_benchmark([spec], [method])

    assert len(report.rows) == 2
    for r in report.rows:
        assert r.effective_recall == 1.0
        assert r.effective_precision == 1.0
        # Exact match on true_optimum_params → zero L2 error
        assert r.param_L2_error == pytest.approx(0.0, abs=1e-9)
        # Fitness gap may be slightly negative (oracle with minimum decoy offset)
        # but must be ≤ 0 — the oracle doesn't leave optimum fitness on the table.
        assert r.fitness_gap_pct <= 0.0 + 1e-6
        assert r.stress_rank_spearman == pytest.approx(1.0)
        assert r.status_pass == 1


def test_benchmark_scorecard_aggregates_across_seeds():
    spec = BenchmarkSpec(
        keyhole_name="PhantomKeyhole",
        keyhole_factory=PhantomKeyhole,
        seeds=[42, 7, 100],
    )
    method = CalibrationMethod(name="oracle", runner=_minimal_runner)
    report = run_benchmark([spec], [method])
    cards = report.scorecard()
    assert len(cards) == 1
    assert cards[0].method == "oracle"
    assert cards[0].n_runs == 3
    assert cards[0].pass_rate == 1.0
    assert cards[0].effective_recall_mean == 1.0
    assert cards[0].effective_recall_std == 0.0


def test_benchmark_captures_crashing_method():
    """Crashing methods must produce a CRASH row, not abort the whole run."""
    spec = BenchmarkSpec(
        keyhole_name="PhantomKeyhole",
        keyhole_factory=PhantomKeyhole,
        seeds=[42],
    )

    def broken(target, seed):
        raise RuntimeError("boom")

    methods = [
        CalibrationMethod(name="oracle", runner=_minimal_runner),
        CalibrationMethod(name="broken", runner=broken),
    ]
    report = run_benchmark([spec], methods)
    assert len(report.rows) == 2
    statuses = {r.method: r.status_raw for r in report.rows}
    assert statuses["oracle"] == "PASS"
    assert statuses["broken"].startswith("CRASH")


def test_benchmark_render_scorecard_returns_text():
    spec = BenchmarkSpec("P", PhantomKeyhole, seeds=[42])
    report = run_benchmark([spec], [CalibrationMethod("oracle", _minimal_runner)])
    text = report.render_scorecard()
    assert "oracle" in text
    assert "method" in text
    assert "recall" in text


def test_benchmark_report_json_round_trip(tmp_path):
    import json as _json
    spec = BenchmarkSpec("P", PhantomKeyhole, seeds=[42, 7])
    report = run_benchmark([spec], [CalibrationMethod("oracle", _minimal_runner)])
    out = tmp_path / "bench.json"
    report.save(out)
    payload = _json.loads(out.read_text())
    assert "rows" in payload and "scorecard" in payload
    assert len(payload["rows"]) == 2
    assert payload["scorecard"][0]["method"] == "oracle"
