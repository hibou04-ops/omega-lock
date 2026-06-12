# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""render_html — single-file, stdlib-only HTML scorecard for audit results.

Accepts (object or its serialized dict form):
    * ``P1Result``           — the run_p1 pipeline artifact
    * ``AuditReport``        — the append-only audit trail report
    * ``StudyAuditReport``   — the Optuna bridge report
    * ``GateVerdict``        — the ``omega_lock.simple.gate_scores`` verdict

Output: one dark-theme HTML document with
    * a verdict banner (PASS / FAIL per the KC gates, or a neutral
      TRAIL banner for gate-less audit trails),
    * a best_any vs best_feasible table (train vs holdout columns),
    * a stress ranking table (when stress data exists), and
    * an inline SVG scatter of train vs holdout/test fitness per
      candidate with the identity line.

Determinism: pure string templating, no timestamps unless the caller
passes ``generated_at`` explicitly, no external assets, no JS. Rendering
the same input twice yields byte-identical HTML.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from omega_lock.audit._types import AUDIT_REPORT_SCHEMA_VERSION, AuditReport
from omega_lock.orchestrator import P1_RESULT_SCHEMA_VERSION, P1Result

__all__ = ["render_html"]


# ── Internal scorecard model ────────────────────────────────────────────────


@dataclass
class _BestRow:
    label: str
    train: float | None
    holdout: float | None
    note: str = ""


@dataclass
class _Scorecard:
    title: str
    verdict: str                 # banner text, e.g. "PASS" / "FAIL:KC-4" / "TRAIL"
    verdict_kind: str            # "pass" | "fail" | "neutral"
    meta: list[tuple[str, str]] = field(default_factory=list)
    kc_rows: list[tuple[str, str, str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    best_rows: list[_BestRow] = field(default_factory=list)
    stress_rows: list[tuple[str, str, str]] = field(default_factory=list)
    scatter: list[tuple[str, float, float]] = field(default_factory=list)
    scatter_x_label: str = "train fitness"
    scatter_y_label: str = "holdout fitness"


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return format(v, ".6g")
    return str(v)


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ── Coercers ────────────────────────────────────────────────────────────────


def _p1_to_dict(result: P1Result) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "kc_reports": result.kc_reports,
        "stress_results": result.stress_results,
        "grid_results": result.grid_results,
        "grid_best": result.grid_best,
        "walk_forward": result.walk_forward,
        "holdout_result": result.holdout_result,
        "constraint_policy": result.constraint_policy,
        "omega_lock_version": result.omega_lock_version,
    }


def _from_p1_payload(d: Mapping[str, Any]) -> _Scorecard:
    status = str(d.get("status", "?"))
    card = _Scorecard(
        title="omega-lock P1 run",
        verdict=status,
        verdict_kind="pass" if status == "PASS" else "fail",
    )
    card.meta.append(("schema", str(d.get("schema_version", "?"))))
    if d.get("omega_lock_version"):
        card.meta.append(("omega-lock version", str(d["omega_lock_version"])))
    if d.get("constraint_policy"):
        card.meta.append(("constraint policy", str(d["constraint_policy"])))

    for kc in d.get("kc_reports") or []:
        card.kc_rows.append(
            (str(kc.get("name", "?")), str(kc.get("status", "?")), str(kc.get("message", "")))
        )

    grid_results = d.get("grid_results") or []
    grid_best = d.get("grid_best")
    holdout = d.get("holdout_result")
    holdout_fitness = _as_float(holdout.get("fitness")) if holdout else None
    if grid_results:
        best_any = max(grid_results, key=lambda g: g.get("fitness", float("-inf")))
        best_any_is_selected = bool(grid_best) and best_any.get("idx") == grid_best.get("idx")
        card.best_rows.append(
            _BestRow(
                label="best_any",
                train=_as_float(best_any.get("fitness")),
                holdout=holdout_fitness if best_any_is_selected else None,
                note="highest train fitness, constraints ignored",
            )
        )
    if grid_best is not None:
        card.best_rows.append(
            _BestRow(
                label="grid_best (selected)",
                train=_as_float(grid_best.get("fitness")),
                holdout=holdout_fitness,
                note=f"selected under constraint_policy={d.get('constraint_policy', 'record')}",
            )
        )

    for s in sorted(
        d.get("stress_results") or [],
        key=lambda s: s.get("raw_stress", 0.0),
        reverse=True,
    ):
        card.stress_rows.append(
            (
                str(s.get("name", "?")),
                _fmt(_as_float(s.get("raw_stress"))),
                _fmt(_as_float(s.get("normalized_stress"))),
            )
        )

    wf = d.get("walk_forward")
    if wf and wf.get("train_fitnesses") and wf.get("test_fitnesses"):
        card.scatter_x_label = "train fitness"
        card.scatter_y_label = "test fitness (walk-forward slice)"
        for i, (tr, te) in enumerate(zip(wf["train_fitnesses"], wf["test_fitnesses"])):
            tr_f, te_f = _as_float(tr), _as_float(te)
            if tr_f is not None and te_f is not None:
                card.scatter.append((f"top-{i + 1}", tr_f, te_f))
    return card


def _from_audit_report_payload(d: Mapping[str, Any]) -> _Scorecard:
    summary = d.get("summary") or {}
    n_total = summary.get("n_total", len(d.get("runs") or []))
    n_feasible = summary.get("n_feasible")
    card = _Scorecard(
        title="omega-lock audit trail",
        verdict="TRAIL",
        verdict_kind="neutral",
    )
    card.meta.append(("schema", str(d.get("schema_version", "?"))))
    card.meta.append(("method", str(d.get("method", "?"))))
    if d.get("omega_lock_version"):
        card.meta.append(("omega-lock version", str(d["omega_lock_version"])))
    card.meta.append(("runs", _fmt(n_total)))
    if n_feasible is not None:
        card.meta.append(("feasible runs", _fmt(n_feasible)))

    runs = d.get("runs") or []
    best_any = max(runs, key=lambda r: r.get("fitness", float("-inf")), default=None)
    feasible_runs = [r for r in runs if not r.get("constraints_failed")]
    best_feasible = max(
        feasible_runs, key=lambda r: r.get("fitness", float("-inf")), default=None
    )
    if best_any is not None:
        card.best_rows.append(
            _BestRow(
                label="best_any",
                train=_as_float(best_any.get("fitness")),
                holdout=None,
                note="highest fitness in the trail, constraints ignored",
            )
        )
    card.best_rows.append(
        _BestRow(
            label="best_feasible",
            train=_as_float(best_feasible.get("fitness")) if best_feasible else None,
            holdout=None,
            note=(
                "highest fitness with zero constraint failures"
                if best_feasible is not None
                else "absent - every run violated a constraint"
            ),
        )
    )

    for pair in d.get("stress_ranking") or []:
        if isinstance(pair, Sequence) and len(pair) >= 2:
            card.stress_rows.append((str(pair[0]), _fmt(_as_float(pair[1])), "-"))
    return card


def _from_study_audit_payload(d: Mapping[str, Any]) -> _Scorecard:
    kc = d.get("kc_report") or {}
    status = str(kc.get("status", "?"))
    passed = bool(d.get("passed"))
    card = _Scorecard(
        title="omega-lock Optuna study audit",
        verdict=f"KC-4 {status}",
        verdict_kind="pass" if passed and status == "PASS" else (
            "neutral" if status == "SKIP" else ("pass" if passed else "fail")
        ),
    )
    card.meta.append(("schema", str(d.get("schema_version", "?"))))
    card.meta.append(("trials (total)", _fmt(d.get("n_trials_total"))))
    card.meta.append(("trials (completed)", _fmt(d.get("n_trials_completed"))))
    card.meta.append(("gated top-N", _fmt(d.get("top_n"))))
    card.meta.append(("feasibility source", str(d.get("feasibility_source", "?"))))
    card.meta.append(("pearson", _fmt(_as_float(d.get("pearson")))))
    card.kc_rows.append(
        (str(kc.get("name", "KC-4")), status, str(kc.get("message", "")))
    )

    def _row(label: str, cand: Mapping[str, Any] | None, note: str) -> _BestRow:
        if cand is None:
            return _BestRow(label=label, train=None, holdout=None, note=note)
        return _BestRow(
            label=label,
            train=_as_float(cand.get("train_value")),
            holdout=_as_float(cand.get("holdout_value")),
            note=note,
        )

    card.best_rows.append(
        _row("best_any", d.get("best_any"), "study winner, constraints ignored")
    )
    feas_note = (
        "highest-value trial flagged feasible"
        if d.get("best_feasible") is not None
        else (
            "absent - feasibility not inferable from user_attrs"
            if d.get("feasibility_source") == "absent"
            else "absent - no trial flagged feasible"
        )
    )
    card.best_rows.append(_row("best_feasible", d.get("best_feasible"), feas_note))
    gated_note = (
        "certified by the gate"
        if d.get("gated_best") is not None
        else "none - the gate refused to certify a candidate"
    )
    card.best_rows.append(_row("gated_best", d.get("gated_best"), gated_note))

    for cand in d.get("candidates") or []:
        tr = _as_float(cand.get("train_value"))
        ho = _as_float(cand.get("holdout_value"))
        if tr is not None and ho is not None:
            card.scatter.append((f"trial {cand.get('number')}", tr, ho))
    return card


def _from_gate_verdict(verdict: Any) -> _Scorecard:
    passed = bool(verdict.passed)
    kc = verdict.kc_report
    card = _Scorecard(
        title="omega-lock score gate",
        verdict="PASS" if passed else "FAIL",
        verdict_kind="pass" if passed else "fail",
    )
    card.meta.append(("pearson", _fmt(verdict.pearson)))
    card.meta.append(("n scores", _fmt(len(verdict.train_scores))))
    card.kc_rows.append((kc.name, kc.status, kc.message))
    card.reasons = [str(r) for r in verdict.reasons]
    for i, (tr, ho) in enumerate(zip(verdict.train_scores, verdict.holdout_scores)):
        card.scatter.append((f"candidate {i + 1}", float(tr), float(ho)))
    return card


def _coerce(obj: Any) -> _Scorecard:
    # Dataclass instances first (cheap isinstance / duck checks), then
    # serialized dict forms keyed by their embedded schema_version.
    from omega_lock.integrations.optuna_bridge import (  # local: avoid import cycle
        StudyAuditReport,
    )
    from omega_lock.simple import GateVerdict  # local: avoid import cycle

    if isinstance(obj, P1Result):
        return _from_p1_payload(_p1_to_dict(obj))
    if isinstance(obj, AuditReport):
        return _from_audit_report_payload(obj.to_dict())
    if isinstance(obj, StudyAuditReport):
        return _from_study_audit_payload(obj.to_dict())
    if isinstance(obj, GateVerdict):
        return _from_gate_verdict(obj)
    if isinstance(obj, Mapping):
        schema = str(obj.get("schema_version", ""))
        if schema.startswith("omega-lock.p1-result."):
            return _from_p1_payload(obj)
        if schema.startswith("omega-lock.audit-report."):
            return _from_audit_report_payload(obj)
        if schema.startswith("omega-lock.study-audit."):
            return _from_study_audit_payload(obj)
        raise ValueError(
            "render_html: unrecognized mapping — expected a serialized "
            f"P1Result ({P1_RESULT_SCHEMA_VERSION!r}), AuditReport "
            f"({AUDIT_REPORT_SCHEMA_VERSION!r}), or StudyAuditReport; "
            f"got schema_version={schema!r}"
        )
    raise TypeError(
        "render_html accepts P1Result, AuditReport, StudyAuditReport, "
        f"GateVerdict, or their serialized dict forms; got {type(obj).__name__}"
    )


# ── SVG scatter ─────────────────────────────────────────────────────────────

_SVG_W = 460
_SVG_H = 340
_SVG_PAD = 46


def _svg_scatter(card: _Scorecard) -> str:
    points = card.scatter
    if not points:
        return ""
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    span = hi - lo
    if span == 0:
        span = 1.0
    lo -= 0.08 * span
    hi += 0.08 * span
    span = hi - lo

    def sx(v: float) -> str:
        return format(_SVG_PAD + (v - lo) / span * (_SVG_W - 2 * _SVG_PAD), ".2f")

    def sy(v: float) -> str:
        return format(_SVG_H - _SVG_PAD - (v - lo) / span * (_SVG_H - 2 * _SVG_PAD), ".2f")

    parts: list[str] = []
    parts.append(
        f'<svg width="{_SVG_W}" height="{_SVG_H}" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        'role="img" aria-label="train vs holdout fitness scatter">'
    )
    # frame
    parts.append(
        f'<rect x="{_SVG_PAD}" y="{_SVG_PAD}" width="{_SVG_W - 2 * _SVG_PAD}" '
        f'height="{_SVG_H - 2 * _SVG_PAD}" class="frame"/>'
    )
    # identity line y = x
    parts.append(
        f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}" '
        'class="identity"/>'
    )
    # axis range labels
    parts.append(
        f'<text x="{_SVG_PAD}" y="{_SVG_H - _SVG_PAD + 16}" class="tick">{_fmt(lo)}</text>'
    )
    parts.append(
        f'<text x="{_SVG_W - _SVG_PAD}" y="{_SVG_H - _SVG_PAD + 16}" class="tick" '
        f'text-anchor="end">{_fmt(hi)}</text>'
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 8}" class="axis" '
        f'text-anchor="middle">{html.escape(card.scatter_x_label)}</text>'
    )
    parts.append(
        f'<text x="14" y="{_SVG_H / 2:.0f}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 14 {_SVG_H / 2:.0f})">'
        f"{html.escape(card.scatter_y_label)}</text>"
    )
    for label, x, y in points:
        parts.append(
            f'<circle cx="{sx(x)}" cy="{sy(y)}" r="4" class="pt">'
            f"<title>{html.escape(label)}: {_fmt(x)} -&gt; {_fmt(y)}</title></circle>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


# ── HTML assembly ───────────────────────────────────────────────────────────

_CSS = """
:root { color-scheme: dark; }
body { background: #0f1117; color: #d7dae0; font-family: 'Segoe UI', system-ui,
       -apple-system, sans-serif; margin: 0; padding: 2rem; }
.card { max-width: 880px; margin: 0 auto; }
h1 { font-size: 1.25rem; letter-spacing: 0.02em; color: #f0f2f5; }
h2 { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em;
     color: #8b93a3; margin-top: 2rem; border-bottom: 1px solid #262b36;
     padding-bottom: 0.4rem; }
.banner { padding: 0.9rem 1.2rem; border-radius: 8px; font-size: 1.05rem;
          font-weight: 600; letter-spacing: 0.03em; margin: 1rem 0; }
.banner.pass { background: #11331f; color: #4ade80; border: 1px solid #1f7a44; }
.banner.fail { background: #38151a; color: #f87171; border: 1px solid #a03040; }
.banner.neutral { background: #1c2230; color: #93a4c4; border: 1px solid #33405c; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid #232836; }
th { color: #8b93a3; font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.status-PASS { color: #4ade80; font-weight: 600; }
.status-FAIL { color: #f87171; font-weight: 600; }
.status-SKIP, .status-ADVISORY { color: #facc15; font-weight: 600; }
.meta { color: #8b93a3; font-size: 0.85rem; }
.meta span { margin-right: 1.2rem; }
ul.reasons { color: #f0b0b0; }
svg .frame { fill: none; stroke: #2a3040; }
svg .identity { stroke: #5b6478; stroke-dasharray: 5 4; }
svg .pt { fill: #60a5fa; fill-opacity: 0.85; }
svg .tick, svg .axis { fill: #8b93a3; font-size: 11px; }
footer { margin-top: 2.5rem; color: #5b6478; font-size: 0.78rem; }
""".strip()


def _render(card: _Scorecard, generated_at: str | None) -> str:
    out: list[str] = []
    out.append("<!DOCTYPE html>")
    out.append('<html lang="en">')
    out.append("<head>")
    out.append('<meta charset="utf-8">')
    out.append(f"<title>{html.escape(card.title)}</title>")
    out.append(f"<style>\n{_CSS}\n</style>")
    out.append("</head>")
    out.append("<body>")
    out.append('<div class="card">')
    out.append(f"<h1>{html.escape(card.title)}</h1>")
    out.append(
        f'<div class="banner {card.verdict_kind}">{html.escape(card.verdict)}</div>'
    )
    if card.meta:
        out.append('<p class="meta">')
        out.append(
            " ".join(
                f"<span>{html.escape(k)}: <b>{html.escape(v)}</b></span>"
                for k, v in card.meta
            )
        )
        out.append("</p>")

    if card.kc_rows:
        out.append("<h2>Gates</h2>")
        out.append("<table><tr><th>gate</th><th>status</th><th>message</th></tr>")
        for name, status, message in card.kc_rows:
            status_class = html.escape(status.split(":", 1)[0])
            out.append(
                f"<tr><td>{html.escape(name)}</td>"
                f'<td class="status-{status_class}">{html.escape(status)}</td>'
                f"<td>{html.escape(message)}</td></tr>"
            )
        out.append("</table>")

    if card.reasons:
        out.append("<h2>Reasons</h2>")
        out.append('<ul class="reasons">')
        for reason in card.reasons:
            out.append(f"<li>{html.escape(reason)}</li>")
        out.append("</ul>")

    if card.best_rows:
        out.append("<h2>Candidates</h2>")
        out.append(
            "<table><tr><th>candidate</th><th>train</th><th>holdout</th><th>note</th></tr>"
        )
        for row in card.best_rows:
            out.append(
                f"<tr><td>{html.escape(row.label)}</td>"
                f'<td class="num">{html.escape(_fmt(row.train))}</td>'
                f'<td class="num">{html.escape(_fmt(row.holdout))}</td>'
                f"<td>{html.escape(row.note)}</td></tr>"
            )
        out.append("</table>")

    if card.stress_rows:
        out.append("<h2>Stress ranking</h2>")
        out.append(
            "<table><tr><th>parameter</th><th>raw stress</th><th>normalized</th></tr>"
        )
        for name, raw, norm in card.stress_rows:
            out.append(
                f"<tr><td>{html.escape(name)}</td>"
                f'<td class="num">{html.escape(raw)}</td>'
                f'<td class="num">{html.escape(norm)}</td></tr>'
            )
        out.append("</table>")

    svg = _svg_scatter(card)
    if svg:
        out.append("<h2>Train vs holdout transfer</h2>")
        out.append(
            "<p class=\"meta\">Points on the dashed identity line transfer "
            "perfectly; points far below it scored high in-sample but did "
            "not carry over.</p>"
        )
        out.append(svg)

    footer_bits = ["omega-lock scorecard"]
    if generated_at is not None:
        footer_bits.append(f"generated at {generated_at}")
    out.append(f"<footer>{html.escape(' | '.join(footer_bits))}</footer>")
    out.append("</div>")
    out.append("</body>")
    out.append("</html>")
    return "\n".join(out) + "\n"


def render_html(
    obj: Any,
    path: str | Path | None = None,
    *,
    generated_at: str | None = None,
) -> str:
    """Render an audit result to a single-file dark-theme HTML scorecard.

    Args:
        obj: a ``P1Result``, ``AuditReport``, ``StudyAuditReport``,
            ``GateVerdict``, or the serialized dict form of the first three
            (e.g. a ``P1Result`` JSON artifact loaded with ``json.load``).
        path: optional output file; written UTF-8 with ``\\n`` newlines.
        generated_at: optional caller-supplied timestamp string. When
            omitted, the output embeds no timestamp at all so repeated
            renders of the same input are byte-identical.

    Returns:
        The HTML document as a string (also written to ``path`` if given).

    Raises:
        TypeError / ValueError: unsupported object or unrecognized dict
            schema.
    """
    card = _coerce(obj)
    document = _render(card, generated_at)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8", newline="\n")
    return document
