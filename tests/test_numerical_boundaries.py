"""Regressions for finite, degenerate and extreme-scale gate inputs."""
from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pytest

from omega_lock import gate_scores
from omega_lock.cli import main
from omega_lock.kill_criteria import KCThresholds, check_kc4
from omega_lock.walk_forward import pearson, pearson_result


@pytest.mark.parametrize("xs,ys", [
    ([1e200, 2e200, 3e200], [3e200, 2e200, 1e200]),
    ([1e-310, 2e-310, 3e-310], [3e-310, 2e-310, 1e-310]),
    ([-1e308, 0.0, 1e308], [1e308, 0.0, -1e308]),
    ([1e16, 1e16 + 2, 1e16 + 4], [3.0, 2.0, 1.0]),
    ([0.0, math.ulp(0.0), 2 * math.ulp(0.0)], [3.0, 2.0, 1.0]),
])
def test_extreme_nonconstant_inputs_have_measured_correlation(xs, ys):
    result = pearson_result(xs, ys)
    assert result.status == "OK"
    assert result.value == pytest.approx(-1.0)
    assert gate_scores(xs, ys).passed is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis,status", [(0, "NON_FINITE_X"), (1, "NON_FINITE_Y")])
def test_nonfinite_inputs_are_uncomputable_and_fail_closed(bad, axis, status):
    arrays = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    arrays[axis][1] = bad
    result = pearson_result(*arrays)
    assert result.value is None
    assert result.status == status
    assert pearson(*arrays) == 0.0  # legacy wrapper's degenerate fallback
    verdict = gate_scores(*arrays)
    assert verdict.passed is False
    assert verdict.pearson is None
    assert status in verdict.reasons[0]


@pytest.mark.parametrize("xs,ys,status", [
    ([], [], "EMPTY"),
    ([1.0], [1.0, 2.0], "LENGTH_MISMATCH"),
    ([1e308, 1e308], [1.0, 2.0], "ZERO_VARIANCE_X"),
    ([1.0, 2.0], [1e-310, 1e-310], "ZERO_VARIANCE_Y"),
])
def test_existing_degeneracy_categories_remain_explicit(xs, ys, status):
    assert pearson_result(xs, ys).status == status


def test_linear_association_does_not_guarantee_absolute_quality():
    verdict = gate_scores([1.0, 2.0, 3.0], [-99.0, -98.0, -97.0])
    assert verdict.passed is True
    assert verdict.pearson == pytest.approx(1.0)


@pytest.mark.parametrize("exponent", [-310, -150, 0, 150, 300])
def test_scaled_nontrivial_correlation_matches_decimal_oracle(exponent):
    scale = 10.0 ** exponent
    xs = [value * scale for value in [1, 4, -3, 6, 2]]
    ys = [value * scale for value in [3, -2, 1, 7, 4]]
    with localcontext() as context:
        context.prec = 100
        dx = [Decimal.from_float(value) for value in xs]
        dy = [Decimal.from_float(value) for value in ys]
        mx = sum(dx, Decimal(0)) / Decimal(len(dx))
        my = sum(dy, Decimal(0)) / Decimal(len(dy))
        numerator = sum(((x - mx) * (y - my) for x, y in zip(dx, dy)), Decimal(0))
        denominator = (sum(((x - mx) ** 2 for x in dx), Decimal(0))
                       * sum(((y - my) ** 2 for y in dy), Decimal(0))).sqrt()
        expected = float(numerator / denominator)
    assert pearson_result(xs, ys).value == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.01, 1.01])
def test_invalid_correlation_threshold_rejected(value):
    with pytest.raises(ValueError, match="pearson_min"):
        KCThresholds(pearson_min=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.01])
def test_invalid_ratio_threshold_rejected(value):
    with pytest.raises(ValueError, match="trade_ratio_min"):
        KCThresholds(trade_ratio_min=value)


@pytest.mark.parametrize("ratio", [math.nan, math.inf, -math.inf])
def test_enabled_ratio_gate_rejects_nonfinite_measurement(ratio):
    report = check_kc4([1, 2, 3], [1, 2, 3], ratio, KCThresholds())
    assert report.status == "FAIL"
    assert report.detail["trade_ratio_ok"] is False


def test_candidate_ids_validate_correspondence():
    assert gate_scores([1, 2, 3], [2, 3, 4],
                       train_ids=["A", "B", "C"], holdout_ids=["A", "B", "C"]).passed


@pytest.mark.parametrize("train_ids,holdout_ids", [
    (["A", "B", "C"], ["B", "A", "C"]),
    (["A", "B", "C"], ["A", "B", "D"]),
    (["A", "A", "C"], ["A", "A", "C"]),
    (["A", "B"], ["A", "B"]),
    (["A", "B", "C"], None),
    (["A", "", "C"], ["A", "", "C"]),
    (["A", "B", "C"], ["a", "B", "C"]),
])
def test_candidate_ids_fail_before_wrong_pairing(train_ids, holdout_ids):
    with pytest.raises(ValueError):
        gate_scores([1, 2, 3], [2, 3, 4], train_ids=train_ids, holdout_ids=holdout_ids)


@pytest.mark.parametrize("payload", ["[1, NaN, 3]", "[1, Infinity, 3]", "[1, 1e999, 3]"])
def test_cli_nonfinite_scores_are_input_errors(tmp_path, payload, capsys):
    scores = tmp_path / "scores.json"
    scores.write_text(payload)
    assert main(["gate", "--train", str(scores), "--holdout", str(scores)]) == 2
    assert "finite" in capsys.readouterr().err


def test_cli_invalid_threshold_is_input_error(tmp_path, capsys):
    scores = tmp_path / "scores.json"
    scores.write_text("[1, 2, 3]")
    assert main(["gate", "--train", str(scores), "--holdout", str(scores),
                 "--pearson-min", "nan"]) == 2
    assert "pearson_min" in capsys.readouterr().err
