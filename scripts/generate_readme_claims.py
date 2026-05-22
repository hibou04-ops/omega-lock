#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Generate README public-claim proof artifacts from an offline ledger.

The ledger is stored as JSON-compatible YAML so this script can run with the
Python standard library only. It does not use network access and does not run
the listed proof commands; commands are recorded as reproducible offline proof
hooks for reviewers and CI jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = Path("docs/claims/public_claims.yml")
GENERATED_MD = Path("docs/claims/generated_readme_claims.md")
GENERATED_JSON = Path("docs/claims/generated_readme_claims.json")

ClaimClass = Literal[
    "source_of_truth",
    "generated_doc",
    "reproducible_command",
    "deterministic_artifact",
    "qualitative_marker",
]

ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "source_of_truth",
        "generated_doc",
        "reproducible_command",
        "deterministic_artifact",
        "qualitative_marker",
    }
)
ALLOWED_STATUSES: frozenset[str] = frozenset({"validated", "qualitative", "todo"})
REQUIRED_CLAIM_IDS: tuple[str, ...] = (
    "walk_forward_validation",
    "hard_constraint_compliance",
    "feasible_best_vs_absolute_best",
    "append_only_audit_trail",
    "sha256_hash_chain_tamper_detection",
    "deterministic_offline_demos",
    "benchmark_scorecard",
    "stress_rank_spearman",
    "package_naming_install",
    "no_omega_lock_diff_cli",
)


@dataclass(frozen=True)
class Diagnostic:
    status: str
    message: str


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_ledger(root: Path, ledger_path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    path = ledger_path if ledger_path.is_absolute() else root / ledger_path
    return json.loads(path.read_text(encoding="utf-8"))


def _claims(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def validate_ledger(root: Path, ledger: dict[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    claims = _claims(ledger)
    readme_path = root / str(ledger.get("readme", "README.md"))
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    if not claims:
        diagnostics.append(Diagnostic("FAIL", "ledger has no claims list"))

    seen_ids: set[str] = set()
    for claim in claims:
        claim_id = claim.get("id")
        classification = claim.get("classification")
        status = claim.get("status", "validated")
        proof = claim.get("proof", [])
        markers = claim.get("readme_markers", [])

        if not isinstance(claim_id, str) or not claim_id:
            diagnostics.append(Diagnostic("FAIL", "claim is missing a non-empty id"))
            continue
        if claim_id in seen_ids:
            diagnostics.append(Diagnostic("FAIL", f"duplicate claim id: {claim_id}"))
        seen_ids.add(claim_id)

        if classification not in ALLOWED_CLASSIFICATIONS:
            diagnostics.append(
                Diagnostic("FAIL", f"{claim_id}: invalid classification {classification!r}")
            )
            continue

        if status not in ALLOWED_STATUSES:
            diagnostics.append(Diagnostic("FAIL", f"{claim_id}: invalid status {status!r}"))

        if not isinstance(markers, list) or not markers:
            diagnostics.append(Diagnostic("FAIL", f"{claim_id}: readme_markers must be non-empty"))
        else:
            for marker in markers:
                if not isinstance(marker, str) or not marker:
                    diagnostics.append(Diagnostic("FAIL", f"{claim_id}: invalid README marker"))
                elif marker not in readme_text:
                    diagnostics.append(
                        Diagnostic("FAIL", f"{claim_id}: README marker not found: {marker!r}")
                    )

        if not isinstance(proof, list):
            diagnostics.append(Diagnostic("FAIL", f"{claim_id}: proof must be a list"))
            proof = []

        if classification == "qualitative_marker":
            if status not in {"qualitative", "todo"}:
                diagnostics.append(
                    Diagnostic(
                        "FAIL",
                        f"{claim_id}: qualitative_marker claims must use qualitative or todo status",
                    )
                )
        elif not proof:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    f"{claim_id}: non-qualitative claim is missing proof entries",
                )
            )
        elif not any(isinstance(item, dict) and item.get("type") == classification for item in proof):
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    f"{claim_id}: proof must include at least one entry matching classification",
                )
            )

        for index, item in enumerate(proof):
            if not isinstance(item, dict):
                diagnostics.append(Diagnostic("FAIL", f"{claim_id}: proof[{index}] must be an object"))
                continue
            proof_type = item.get("type")
            if proof_type not in ALLOWED_CLASSIFICATIONS:
                diagnostics.append(
                    Diagnostic("FAIL", f"{claim_id}: proof[{index}] has invalid type {proof_type!r}")
                )
                continue
            if proof_type in {"source_of_truth", "generated_doc", "deterministic_artifact"}:
                path_value = item.get("path")
                if not isinstance(path_value, str) or not path_value:
                    diagnostics.append(
                        Diagnostic("FAIL", f"{claim_id}: proof[{index}] is missing path")
                    )
                elif not (root / path_value).exists():
                    diagnostics.append(
                        Diagnostic("FAIL", f"{claim_id}: proof path does not exist: {path_value}")
                    )
            if proof_type == "reproducible_command":
                if not isinstance(item.get("command"), str) or not item["command"]:
                    diagnostics.append(
                        Diagnostic("FAIL", f"{claim_id}: proof[{index}] is missing command")
                    )
                if item.get("network") is not False:
                    diagnostics.append(
                        Diagnostic(
                            "FAIL",
                            f"{claim_id}: reproducible_command proof must set network=false",
                        )
                    )

    missing = sorted(set(REQUIRED_CLAIM_IDS) - seen_ids)
    for claim_id in missing:
        diagnostics.append(Diagnostic("FAIL", f"required README claim is missing: {claim_id}"))

    return diagnostics


def _proof_summary(proof: list[Any]) -> str:
    parts: list[str] = []
    for item in proof:
        if not isinstance(item, dict):
            continue
        proof_type = item.get("type", "?")
        if "path" in item:
            parts.append(f"{proof_type}:{item['path']}")
        elif "command" in item:
            parts.append(f"{proof_type}:{item['command']}")
        else:
            parts.append(str(proof_type))
    return "; ".join(parts) if parts else "none"


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_payload(root: Path, ledger_path: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    claims = sorted(_claims(ledger), key=lambda item: item["id"])
    class_counts = {name: 0 for name in sorted(ALLOWED_CLASSIFICATIONS)}
    status_counts = {name: 0 for name in sorted(ALLOWED_STATUSES)}
    for claim in claims:
        class_counts[claim["classification"]] += 1
        status_counts[claim.get("status", "validated")] += 1

    return {
        "schema_version": 1,
        "source_ledger": _rel((root / ledger_path).resolve(), root),
        "readme": ledger.get("readme", "README.md"),
        "claim_count": len(claims),
        "classification_counts": class_counts,
        "status_counts": status_counts,
        "claims": claims,
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Generated README Claims",
        "",
        "Generated from `docs/claims/public_claims.yml` by "
        "`scripts/generate_readme_claims.py`. Do not edit this file by hand.",
        "",
        f"- README source: `{payload['readme']}`",
        f"- Claim count: {payload['claim_count']}",
        "- Classifications: "
        + ", ".join(
            f"{name}={count}"
            for name, count in payload["classification_counts"].items()
            if count
        ),
        "- Statuses: "
        + ", ".join(
            f"{name}={count}" for name, count in payload["status_counts"].items() if count
        ),
        "",
        "| Claim ID | Classification | Status | README markers | Proof summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in payload["claims"]:
        markers = "; ".join(claim.get("readme_markers", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(claim["id"]),
                    _escape_table(claim["classification"]),
                    _escape_table(claim.get("status", "validated")),
                    _escape_table(markers),
                    _escape_table(_proof_summary(claim.get("proof", []))),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def expected_outputs(root: Path, ledger_path: Path) -> tuple[str, str, list[Diagnostic]]:
    ledger = load_ledger(root, ledger_path)
    diagnostics = validate_ledger(root, ledger)
    payload = build_payload(root, ledger_path, ledger)
    return render_markdown(payload), render_json(payload), diagnostics


def write_outputs(root: Path, ledger_path: Path = DEFAULT_LEDGER) -> list[Diagnostic]:
    markdown, json_text, diagnostics = expected_outputs(root, ledger_path)
    if any(d.status == "FAIL" for d in diagnostics):
        return diagnostics

    md_path = root / GENERATED_MD
    json_path = root / GENERATED_JSON
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    return diagnostics


def check_outputs(root: Path, ledger_path: Path = DEFAULT_LEDGER) -> list[Diagnostic]:
    markdown, json_text, diagnostics = expected_outputs(root, ledger_path)
    if any(d.status == "FAIL" for d in diagnostics):
        return diagnostics

    expected = {
        GENERATED_MD: markdown,
        GENERATED_JSON: json_text,
    }
    for path, text in expected.items():
        full_path = root / path
        if not full_path.exists():
            diagnostics.append(Diagnostic("FAIL", f"generated file is missing: {path.as_posix()}"))
            continue
        actual = full_path.read_text(encoding="utf-8")
        if actual != text:
            diagnostics.append(Diagnostic("FAIL", f"generated file is stale: {path.as_posix()}"))
    return diagnostics


def _print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    if not diagnostics:
        print("PASS: README claim ledger is valid.")
        return
    for diagnostic in diagnostics:
        print(f"{diagnostic.status}: {diagnostic.message}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="claim ledger path relative to root",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    diagnostics = (
        check_outputs(root, args.ledger)
        if args.check
        else write_outputs(root, args.ledger)
    )
    _print_diagnostics(diagnostics)
    return 1 if any(d.status == "FAIL" for d in diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
