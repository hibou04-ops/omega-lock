# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""omega-lock console entry point (``[project.scripts]``).

Subcommands:
    demo    run the deterministic walk-forward gate case study and print
            the full narrative (offline, seeded, < 60 s).
    gate    apply the KC-4 transfer gate to two JSON arrays of scores
            (train vs holdout), optionally writing an HTML scorecard.
    report  render a saved result JSON artifact (``P1Result`` or audit
            report) to a single-file HTML scorecard.

Everything is offline and deterministic; argparse only, no third-party
CLI dependencies. There is still no ``omega-lock diff`` subcommand.

Exit codes: 0 = success / gate passed, 1 = gate failed, 2 = usage or
input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


class SystemExit2(Exception):
    """Input/usage error carrying a message; mapped to exit code 2."""


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit2(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit2(f"invalid JSON in {path}: {exc}")


def _load_score_array(path: str, label: str) -> list[float]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in payload
    ):
        raise SystemExit2(
            f"{label} file {path} must contain a JSON array of numbers, "
            f"e.g. [0.91, 0.84, 0.77]"
        )
    return [float(v) for v in payload]


def _cmd_demo(_args: argparse.Namespace) -> int:
    from omega_lock._demo import run_demo

    return run_demo()


def _cmd_gate(args: argparse.Namespace) -> int:
    from omega_lock.kill_criteria import KCThresholds
    from omega_lock.report_html import render_html
    from omega_lock.simple import gate_scores

    train = _load_score_array(args.train, "--train")
    holdout = _load_score_array(args.holdout, "--holdout")

    thresholds = (
        KCThresholds.pure_objective(pearson_min=args.pearson_min)
        if args.pearson_min is not None
        else None
    )
    verdict = gate_scores(train, holdout, thresholds=thresholds)

    print(f"gate     : {'PASS' if verdict.passed else 'FAIL'}")
    pearson_text = "n/a" if verdict.pearson is None else f"{verdict.pearson:.3f}"
    print(f"pearson  : {pearson_text}")
    print(f"scores   : {len(train)} candidate(s)")
    for reason in verdict.reasons:
        print(f"reason   : {reason}")

    if args.report is not None:
        render_html(verdict, args.report)
        print(f"report   : wrote {args.report}")

    return 0 if verdict.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    from omega_lock.report_html import render_html

    payload = _load_json(args.input)
    if not isinstance(payload, dict):
        raise SystemExit2(
            f"{args.input} must contain a JSON object (a saved P1Result or "
            "audit report artifact)"
        )
    try:
        render_html(payload, args.output)
    except (TypeError, ValueError) as exc:
        raise SystemExit2(str(exc))
    print(f"report   : wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from omega_lock import __version__

    parser = argparse.ArgumentParser(
        prog="omega-lock",
        description=(
            "Audit tuned candidates before they ship: walk-forward gate, "
            "hard constraints, feasible-best selection."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"omega-lock {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo",
        help="run the deterministic walk-forward gate case study",
        description=(
            "Runs the offline, seeded overfitting case study: a lucky-noise "
            "'winner' collapses out-of-sample, the KC-4 gate fails the run, "
            "and the constraint-gated feasible-best holds up."
        ),
    )
    demo.set_defaults(func=_cmd_demo)

    gate = sub.add_parser(
        "gate",
        help="gate train vs holdout score arrays (KC-4 Pearson transfer)",
        description=(
            "Reads two JSON arrays of numbers (index-aligned per candidate) "
            "and applies the KC-4 walk-forward Pearson gate. Exit code 0 = "
            "gate passed, 1 = gate failed."
        ),
    )
    gate.add_argument("--train", required=True, help="JSON array of in-sample scores")
    gate.add_argument(
        "--holdout", required=True, help="JSON array of held-out scores"
    )
    gate.add_argument(
        "--report", default=None, help="optional HTML scorecard output path"
    )
    gate.add_argument(
        "--pearson-min",
        type=float,
        default=None,
        help="override the Pearson gate threshold (default 0.3)",
    )
    gate.set_defaults(func=_cmd_gate)

    report = sub.add_parser(
        "report",
        help="render a saved result JSON artifact to an HTML scorecard",
        description=(
            "Accepts the JSON artifact written by run_p1(output_path=...) / "
            "P1Result.save(), or an AuditReport.to_json() payload, and writes "
            "a single-file dark-theme HTML scorecard."
        ),
    )
    report.add_argument(
        "--input", required=True, help="path to the result JSON artifact"
    )
    report.add_argument(
        "-o", "--output", required=True, help="HTML output path"
    )
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except SystemExit2 as exc:
        print(f"omega-lock: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
