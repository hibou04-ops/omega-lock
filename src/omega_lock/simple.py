# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""omega_lock.simple — plain-language facade over the audit gates.

Three entry points, no pipeline jargon required:

    * ``gate_scores(train_scores, holdout_scores)`` — "my tuner produced
      these scores in-sample and these on held-out data; does the ranking
      transfer?" Wraps the KC-4 Pearson gate over two plain number lists.
    * ``audit(target_fn, param_specs, holdout_fn=...)`` — "audit this
      scoring function over this parameter space." Wraps
      ``CallableAdapter`` + ``run_p1`` with sensible non-action defaults.
    * ``render_html(result, path)`` — re-exported from
      ``omega_lock.report_html`` for one-stop reporting.

These are NEW names, not renames: the full-control surface (``run_p1``,
``P1Config``, ``check_kc4``, ...) is unchanged and remains the contract
for integrators.

Note: ``audit()`` is importable as ``omega_lock.simple.audit`` only. It is
deliberately NOT re-exported at the package root because the name
``omega_lock.audit`` already belongs to the audit-trail subpackage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from omega_lock.adapters import CallableAdapter
from omega_lock.kill_criteria import KCReport, KCThresholds, check_kc4
from omega_lock.orchestrator import P1Config, P1Result, run_p1
from omega_lock.report_html import render_html
from omega_lock.target import ParamSpec

__all__ = ["GateVerdict", "audit", "gate_scores", "render_html"]


@dataclass(frozen=True)
class GateVerdict:
    """Outcome of ``gate_scores``.

    Attributes:
        passed: True when the train-to-holdout ranking transfer cleared
            the gate.
        pearson: measured Pearson correlation, or None when it could not
            be computed (empty input, length mismatch, zero variance —
            see ``reasons`` for which).
        reasons: human-readable failure reasons; empty when ``passed``.
        kc_report: the underlying KC-4 report (full detail dict).
        train_scores / holdout_scores: the gated inputs, kept so the
            verdict can be rendered to an HTML scorecard via
            ``render_html``.
    """

    passed: bool
    pearson: float | None
    reasons: tuple[str, ...]
    kc_report: KCReport
    train_scores: tuple[float, ...] = ()
    holdout_scores: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pearson": self.pearson,
            "reasons": list(self.reasons),
            "kc_report": {
                "name": self.kc_report.name,
                "status": self.kc_report.status,
                "message": self.kc_report.message,
                "detail": dict(self.kc_report.detail),
            },
            "train_scores": list(self.train_scores),
            "holdout_scores": list(self.holdout_scores),
        }


_PEARSON_STATUS_EXPLANATIONS = {
    "EMPTY": "no scores were provided",
    "LENGTH_MISMATCH": "train and holdout score lists have different lengths",
    "ZERO_VARIANCE_X": "all train scores are identical (zero variance)",
    "ZERO_VARIANCE_Y": "all holdout scores are identical (zero variance)",
}


def gate_scores(
    train_scores: Iterable[float],
    holdout_scores: Iterable[float],
    *,
    thresholds: KCThresholds | None = None,
) -> GateVerdict:
    """Gate a candidate ranking: do in-sample scores transfer out-of-sample?

    Pass the per-candidate scores measured on the data the search consumed
    (``train_scores``) and the same candidates re-scored on data it never
    saw (``holdout_scores``), index-aligned. The verdict applies the KC-4
    walk-forward Pearson gate (``check_kc4``) over the pair.

    Args:
        train_scores / holdout_scores: index-aligned numeric sequences.
        thresholds: KC thresholds; defaults to
            ``KCThresholds.pure_objective()`` because plain score lists
            carry no action counts. If you pass thresholds with
            ``trade_ratio_min`` set, that sub-gate is evaluated against a
            neutral ratio of 1.0 (there is no action-count concept here).

    Returns:
        GateVerdict — ``passed``, measured ``pearson`` (None when not
        computable), and explicit ``reasons`` on failure.
    """
    thr = thresholds if thresholds is not None else KCThresholds.pure_objective()
    train = [float(v) for v in train_scores]
    holdout = [float(v) for v in holdout_scores]

    kc = check_kc4(
        train_fitnesses=train,
        test_fitnesses=holdout,
        trade_ratio=1.0,
        thresholds=thr,
    )
    detail = kc.detail
    computable = bool(detail.get("pearson_computable"))
    pearson = float(detail["pearson"]) if computable else None

    reasons: list[str] = []
    if kc.status != "PASS":
        status = str(detail.get("pearson_status", "?"))
        if not computable:
            explanation = _PEARSON_STATUS_EXPLANATIONS.get(status, status)
            reasons.append(
                f"correlation not computable ({status}): {explanation}"
            )
        elif pearson is not None and pearson < thr.pearson_min:
            reasons.append(
                f"train->holdout correlation {pearson:.3f} is below the "
                f"gate threshold {thr.pearson_min} - the in-sample ranking "
                "did not survive out-of-sample"
            )
        if not detail.get("trade_ratio_ok", True):
            reasons.append(
                f"action ratio {detail.get('trade_ratio')} is below "
                f"trade_ratio_min={thr.trade_ratio_min} (note: gate_scores "
                "has no real action counts; ratio is fixed at 1.0)"
            )

    return GateVerdict(
        passed=kc.status == "PASS",
        pearson=pearson,
        reasons=tuple(reasons),
        kc_report=kc,
        train_scores=tuple(train),
        holdout_scores=tuple(holdout),
    )


ParamSpecsLike = Sequence[ParamSpec] | Mapping[str, Sequence[float]]


def _coerce_specs(param_specs: ParamSpecsLike) -> list[ParamSpec]:
    """Accept ParamSpec lists or a friendly ``{name: (low, high)}`` mapping."""
    if isinstance(param_specs, Mapping):
        specs: list[ParamSpec] = []
        for name, bounds in param_specs.items():
            values = list(bounds)
            if len(values) == 2:
                low, high = float(values[0]), float(values[1])
                neutral = (low + high) / 2.0
            elif len(values) == 3:
                low, high, neutral = (
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                )
            else:
                raise ValueError(
                    f"param {name!r}: expected (low, high) or "
                    f"(low, high, neutral), got {bounds!r}"
                )
            specs.append(
                ParamSpec(name=str(name), dtype="float", low=low, high=high, neutral=neutral)
            )
        if not specs:
            raise ValueError("param_specs mapping is empty")
        return specs

    specs = list(param_specs)
    if not specs:
        raise ValueError("param_specs is empty")
    for spec in specs:
        if not isinstance(spec, ParamSpec):
            raise TypeError(
                "param_specs sequence must contain ParamSpec items; "
                f"got {type(spec).__name__}"
            )
    return specs


def audit(
    target_fn: Callable[[dict[str, Any]], float],
    param_specs: ParamSpecsLike,
    *,
    holdout_fn: Callable[[dict[str, Any]], float] | None = None,
    output_path: str | Path | None = None,
    **cfg: Any,
) -> P1Result:
    """Audit a plain scoring function over a parameter space.

    A thin ``CallableAdapter`` + ``run_p1`` wrapper:

        result = audit(score, {"gain": (0.0, 10.0)}, holdout_fn=score_oos)
        print(result.status)            # "PASS" or "FAIL:KC-..."

    Args:
        target_fn: ``(params: dict) -> float`` score to MAXIMIZE, measured
            on the data the search may consume.
        param_specs: either a list of ``ParamSpec`` or a friendly mapping
            ``{name: (low, high)}`` / ``{name: (low, high, neutral)}``
            (floats; neutral defaults to the midpoint).
        holdout_fn: optional ``(params: dict) -> float`` scored on data
            the search never sees. When provided it becomes the
            walk-forward test target, enabling the KC-4 transfer gate.
        output_path: optional path for the ``P1Result`` JSON artifact.
        **cfg: forwarded to ``P1Config`` (e.g. ``unlock_k=2``,
            ``grid_points_per_axis=11``). Unknown keys raise ``TypeError``
            at construction. ``kc_thresholds`` defaults to
            ``KCThresholds.pure_objective()`` here because a bare callable
            reports no action counts (the default action floor of the full
            pipeline would fail it vacuously).

    Returns:
        The full ``P1Result`` — the same artifact ``run_p1`` emits.
    """
    specs = _coerce_specs(param_specs)
    train_target = CallableAdapter(fitness_fn=target_fn, specs=specs)
    test_target = (
        CallableAdapter(fitness_fn=holdout_fn, specs=specs)
        if holdout_fn is not None
        else None
    )
    if "kc_thresholds" not in cfg:
        cfg["kc_thresholds"] = KCThresholds.pure_objective()
    config = P1Config(**cfg)
    return run_p1(
        train_target=train_target,
        config=config,
        test_target=test_target,
        output_path=Path(output_path) if output_path is not None else None,
    )
