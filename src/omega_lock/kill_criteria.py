# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Kill criteria (KC-1..4) — pre-declared, non-negotiable gates.

Philosophy: structural defense against Winchester re-entry. Each KC is
evaluated at a specific point in the pipeline; failing one stops the
run with an explicit write-up rather than letting the operator fudge
thresholds post-hoc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class KCThresholds:
    """Tunable kill-criteria thresholds.

    Defaults match research/omega_lock_p1/SPEC.md §3. Looser thresholds
    (e.g. for toy examples) can be passed to P1Config.
    """
    time_box_seconds: float = 3 * 24 * 3600      # KC-1: 3 days
    gini_min: float = 0.2                         # KC-2a: stress differentiation
    top_bot_ratio_min: float = 2.0                # KC-2b: head vs tail ratio
    trade_count_min: int = 50                     # KC-3: per best-config trade floor
    pearson_min: float = 0.3                      # KC-4a: walk-forward correlation
    trade_ratio_min: float = 0.5                  # KC-4b: test-trade / train-trade


KCStatus = Literal["PASS", "FAIL", "SKIP", "ADVISORY"]


@dataclass
class KCReport:
    name: str                                     # "KC-2", "KC-4", etc.
    status: KCStatus
    message: str
    detail: dict


def check_kc1(elapsed_seconds: float, thresholds: KCThresholds) -> KCReport:
    ok = elapsed_seconds <= thresholds.time_box_seconds
    return KCReport(
        name="KC-1",
        status="PASS" if ok else "FAIL",
        message=(
            f"{'PASS' if ok else 'FAIL'}: elapsed={elapsed_seconds:.0f}s "
            f"(budget={thresholds.time_box_seconds:.0f}s)"
        ),
        detail={"elapsed_s": elapsed_seconds, "budget_s": thresholds.time_box_seconds},
    )


def check_kc2(
    stresses: list[float],
    thresholds: KCThresholds,
) -> KCReport:
    """KC-2: stress distribution differentiation (Gini + top/bot ratio)."""
    from omega_lock.stress import gini_coefficient

    if not stresses:
        return KCReport(
            name="KC-2", status="FAIL",
            message="FAIL: no stress values",
            detail={},
        )
    sorted_desc = sorted(stresses, reverse=True)
    gini = gini_coefficient(sorted_desc)
    n = len(sorted_desc)

    # Pick k such that top-k and bot-k don't overlap; at least k=1.
    # For n < 2, the ratio check is meaningless (single param).
    if n >= 2:
        k = min(3, max(1, n // 2))
        top_k_mean = sum(sorted_desc[:k]) / k
        bot_k_mean = sum(sorted_desc[-k:]) / k
        ratio = (top_k_mean / bot_k_mean) if bot_k_mean > 0 else float("inf")
        ratio_ok = ratio >= thresholds.top_bot_ratio_min
    else:
        k = 1
        top_k_mean = sorted_desc[0]
        bot_k_mean = sorted_desc[0]
        ratio = 1.0
        ratio_ok = True    # skip meaningless check when only 1 param

    gini_ok = gini >= thresholds.gini_min
    passed = gini_ok and ratio_ok
    detail = {
        "gini": gini,
        "gini_ok": gini_ok,
        "top_k_mean": top_k_mean,
        "bot_k_mean": bot_k_mean,
        "ratio": ratio,
        "ratio_ok": ratio_ok,
        "k": k,
    }
    if passed:
        msg = f"PASS: Gini={gini:.3f}, top{k}/bot{k}={ratio:.2f}"
    else:
        fails = []
        if not gini_ok:
            fails.append(f"Gini={gini:.3f}<{thresholds.gini_min}")
        if not ratio_ok:
            fails.append(f"ratio={ratio:.2f}<{thresholds.top_bot_ratio_min}")
        msg = "FAIL: " + ", ".join(fails)
    return KCReport(name="KC-2", status="PASS" if passed else "FAIL", message=msg, detail=detail)


def check_kc3(
    trade_counts: dict[str, int],
    thresholds: KCThresholds,
) -> KCReport:
    """KC-3: best-config action count floor.

    `trade_counts` is a dict of labeled counts (e.g. {"baseline": 173,
    "train_best": 167, "test_best": 129}). All must meet floor.
    """
    if not trade_counts:
        return KCReport(
            name="KC-3", status="FAIL",
            message="FAIL: no trade counts",
            detail={},
        )
    failures = {k: v for k, v in trade_counts.items() if v < thresholds.trade_count_min}
    passed = not failures
    detail = {"counts": dict(trade_counts), "floor": thresholds.trade_count_min, "failures": failures}
    if passed:
        msg = f"PASS: all counts >= {thresholds.trade_count_min} ({trade_counts})"
    else:
        fails = ", ".join(f"{k}={v}" for k, v in failures.items())
        msg = f"FAIL: below floor {thresholds.trade_count_min}: {fails}"
    return KCReport(name="KC-3", status="PASS" if passed else "FAIL", message=msg, detail=detail)


def check_kc4(
    train_fitnesses: list[float],
    test_fitnesses: list[float],
    trade_ratio: float,
    thresholds: KCThresholds,
) -> KCReport:
    """KC-4: walk-forward consistency.

    - Pearson correlation between train and test fitness on top-N grid points
    - Trade ratio (test_best / train_best, scale-adjusted externally)

    The Pearson computation is done via ``pearson_result`` so the
    detail dict can distinguish "we measured a low correlation" from
    "the input was degenerate (empty / mismatched / zero variance)".
    A degenerate input is a FAIL on KC-4 — but the artifact reader
    sees ``pearson_status`` and knows why.
    """
    from omega_lock.walk_forward import pearson_result

    pr = pearson_result(train_fitnesses, test_fitnesses)
    # Backward-compat numeric value: None coerces to 0.0 so existing
    # consumers that only read detail['pearson'] still get a float they
    # can compare to the threshold.
    corr = pr.value if pr.value is not None else 0.0
    corr_ok = pr.computable and corr >= thresholds.pearson_min
    ratio_ok = trade_ratio >= thresholds.trade_ratio_min
    passed = corr_ok and ratio_ok
    detail = {
        "pearson": corr,
        "pearson_ok": corr_ok,
        "pearson_status": pr.status,
        "pearson_computable": pr.computable,
        "trade_ratio": trade_ratio,
        "trade_ratio_ok": ratio_ok,
        "n_points": len(train_fitnesses),
    }
    if passed:
        msg = f"PASS: pearson={corr:.3f}, trade_ratio={trade_ratio:.3f}"
    else:
        fails = []
        if not pr.computable:
            fails.append(f"pearson uncomputable: {pr.status}")
        elif not corr_ok:
            fails.append(f"pearson={corr:.3f}<{thresholds.pearson_min}")
        if not ratio_ok:
            fails.append(f"trade_ratio={trade_ratio:.3f}<{thresholds.trade_ratio_min}")
        msg = "FAIL: " + ", ".join(fails)
    return KCReport(name="KC-4", status="PASS" if passed else "FAIL", message=msg, detail=detail)
