#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Generate a local GitHub release draft markdown file without creating a release.

The generator is offline and deterministic. It reads local release notes,
README claim artifacts, and an optional release-audit JSON file. It never calls
the GitHub API, creates tags, publishes to PyPI, or uploads assets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path("release_drafts")
CLAIM_LEDGER = Path("docs/claims/public_claims.yml")
GENERATED_CLAIMS_JSON = Path("docs/claims/generated_readme_claims.json")
GENERATED_CLAIMS_MD = Path("docs/claims/generated_readme_claims.md")
RELEASE_HISTORY = Path("RELEASE.md")
POST_RELEASE_COMMAND = "python scripts/post_release_verify.py --version {version} --distribution omega-lock --json"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+\Z")


@dataclass(frozen=True)
class DraftResult:
    status: str
    message: str
    output_path: Path | None = None
    details: tuple[str, ...] = ()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_script(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_version(version: str) -> DraftResult:
    if VERSION_RE.fullmatch(version):
        return DraftResult("PASS", f"Version is normalized: {version}.")
    return DraftResult("FAIL", "--version must use X.Y.Z numeric form.", details=(f"received: {version}",))


def check_generated_claims_current(root: Path) -> DraftResult:
    try:
        generator = _load_script(
            "omega_lock_readme_claims_release_draft",
            root / "scripts" / "generate_readme_claims.py",
        )
        diagnostics = generator.check_outputs(root)
    except Exception as exc:  # pragma: no cover - defensive for broken local tooling
        return DraftResult(
            "TOOLING_MISSING",
            "Could not run generated README claims check.",
            details=(str(exc),),
        )

    failures = tuple(
        str(getattr(diagnostic, "message", diagnostic))
        for diagnostic in diagnostics
        if getattr(diagnostic, "status", "") == "FAIL"
    )
    if failures:
        return DraftResult(
            "FAIL",
            "Claim ledger or generated README claim files are stale or invalid.",
            details=failures,
        )
    return DraftResult("PASS", "Claim ledger and generated README claim files are current.")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_claim_payload(root: Path) -> dict[str, Any]:
    payload_path = root / GENERATED_CLAIMS_JSON
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    return load_json(payload_path)


def load_claim_ledger(root: Path) -> dict[str, Any]:
    ledger_path = root / CLAIM_LEDGER
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    return load_json(ledger_path)


def extract_release_history(root: Path, version: str) -> tuple[str, tuple[str, ...]]:
    path = root / RELEASE_HISTORY
    if not path.exists():
        return "missing", ("RELEASE.md is absent.",)

    lines = path.read_text(encoding="utf-8").splitlines()
    exact_heading = f"### {version}"
    for index, line in enumerate(lines):
        if line.strip() == exact_heading:
            body: list[str] = []
            for next_line in lines[index + 1 :]:
                if next_line.startswith("### ") or next_line.startswith("## "):
                    break
                if next_line.strip():
                    body.append(next_line.rstrip())
            return "exact", tuple(body) if body else ("No body text under exact release-history heading.",)

    template_heading = "## Current Release Note Template"
    for index, line in enumerate(lines):
        if line.strip() == template_heading:
            body = []
            for next_line in lines[index + 1 :]:
                if next_line.startswith("## "):
                    break
                if next_line.strip():
                    body.append(next_line.rstrip())
            return "template", tuple(body) if body else ("Current release note template is empty.",)

    return "missing", (f"No release-history entry or current template found for {version}.",)


def _proof_command(proof: dict[str, Any]) -> str | None:
    if proof.get("type") == "reproducible_command":
        command = proof.get("command")
        if isinstance(command, str) and command:
            return command
    return None


def deterministic_commands(claim_payload: dict[str, Any], version: str) -> tuple[str, ...]:
    commands: set[str] = {
        "python scripts/generate_readme_claims.py --check",
        "python scripts/check_repo_consistency.py --check",
        "python scripts/run_golden_audit_cases.py --check",
        "python scripts/scope_freeze_check.py --check",
        f"python scripts/release_audit.py --intended-version {version} --offline --json",
    }
    for claim in claim_payload.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for proof in claim.get("proof", []):
            if isinstance(proof, dict):
                command = _proof_command(proof)
                if command is not None:
                    commands.add(command)
    return tuple(sorted(commands))


def find_release_audit_path(root: Path, version: str, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else root / explicit
        return path

    candidates = (
        root / "dist" / f"release_audit_v{version}.json",
        root / "dist" / f"release_audit_{version}.json",
        root / "dist" / "release_audit.json",
        root / f"release_audit_v{version}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_release_audit(root: Path, version: str, explicit: Path | None = None) -> tuple[Path | None, dict[str, Any] | None]:
    path = find_release_audit_path(root, version, explicit)
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_json(path)
    return path, payload


def list_golden_audit_artifacts(root: Path) -> tuple[str, ...]:
    fixture_dir = root / "tests" / "fixtures" / "golden_audits"
    if not fixture_dir.exists():
        return ()
    return tuple(_rel(path, root) for path in sorted(fixture_dir.glob("*.json")))


def list_dist_artifacts(root: Path, version: str) -> tuple[str, ...]:
    dist = root / "dist"
    if not dist.exists():
        return ()
    return tuple(
        _rel(path, root)
        for pattern in ("*.whl", "*.tar.gz")
        for path in sorted(dist.glob(pattern))
        if path.is_file()
        and path.name.startswith("omega_lock-")
        and version in path.name
    )


def _claim_rows(claim_payload: dict[str, Any], *, qualitative: bool) -> tuple[str, ...]:
    rows: list[str] = []
    for claim in sorted(claim_payload.get("claims", []), key=lambda item: str(item.get("id", ""))):
        if not isinstance(claim, dict):
            continue
        is_qualitative = claim.get("classification") == "qualitative_marker"
        if is_qualitative != qualitative:
            continue
        rows.append(
            f"- `{claim.get('id')}` ({claim.get('classification')}, {claim.get('status', 'validated')}): "
            f"{claim.get('claim')}"
        )
    return tuple(rows)


def _release_audit_lines(root: Path, path: Path | None, payload: dict[str, Any] | None) -> tuple[str, ...]:
    if payload is None or path is None:
        return (
            "- No release audit JSON was found. Run the offline release audit command before treating this draft as release evidence.",
        )

    lines = [f"- Source: `{_rel(path, root)}`"]
    mode = payload.get("mode")
    if mode:
        lines.append(f"- Mode: `{mode}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        counts = ", ".join(f"{key}={summary[key]}" for key in sorted(summary) if summary[key])
        lines.append(f"- Summary: {counts or 'no statuses recorded'}")
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        status = result.get("status", "?")
        name = result.get("name", "?")
        message = result.get("message", "")
        if status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED", "WARN"}:
            lines.append(f"- `{status}` `{name}`: {message}")
    return tuple(lines)


def render_draft(
    *,
    root: Path,
    version: str,
    claim_payload: dict[str, Any],
    claim_ledger: dict[str, Any],
    release_history_status: str,
    release_history_lines: Sequence[str],
    release_audit_path: Path | None,
    release_audit_payload: dict[str, Any] | None,
) -> str:
    generated_artifacts = (
        GENERATED_CLAIMS_MD.as_posix(),
        GENERATED_CLAIMS_JSON.as_posix(),
    )
    golden_artifacts = list_golden_audit_artifacts(root)
    dist_artifacts = list_dist_artifacts(root, version)
    commands = deterministic_commands(claim_payload, version)
    verified_rows = _claim_rows(claim_payload, qualitative=False)
    qualitative_rows = _claim_rows(claim_payload, qualitative=True)
    release_audit_lines = _release_audit_lines(root, release_audit_path, release_audit_payload)

    lines = [
        f"# omega-lock v{version} GitHub Release Draft",
        "",
        "This is a local draft generated for review. The generator did not publish to PyPI, create or push git tags, create a GitHub release, call the GitHub API, or upload assets.",
        "",
        "## Summary",
        "",
        "- GitHub repo: `hibou04-ops/omega-lock`",
        "- PyPI distribution: `omega-lock`",
        "- Python import package: `omega_lock`",
        "- CLI executable: none currently, unless a future explicit change adds one",
        f"- Version: `{version}`",
        f"- Claim ledger source: `{CLAIM_LEDGER.as_posix()}` with {claim_ledger.get('schema_version', '?')} schema version",
        "",
        "## Release History Source",
        "",
        f"- Status: `{release_history_status}`",
    ]
    lines.extend(f"- {line}" for line in release_history_lines)
    lines.extend(
        [
            "",
            "## Verified Changes",
            "",
        ]
    )
    lines.extend(verified_rows or ["- No non-qualitative generated README claims were available."])
    lines.extend(
        [
            "",
            "## Qualitative Or TODO Claim Boundaries",
            "",
        ]
    )
    lines.extend(qualitative_rows or ["- No qualitative/TODO claim boundaries were available."])
    lines.extend(
        [
            "",
            "## Audit Artifacts",
            "",
            "- Generated README claim artifacts:",
        ]
    )
    lines.extend(f"  - `{path}`" for path in generated_artifacts)
    lines.append("- Golden audit artifacts:")
    lines.extend(f"  - `{path}`" for path in golden_artifacts) if golden_artifacts else lines.append("  - none found")
    lines.append("- Local dist artifacts observed:")
    lines.extend(f"  - `{path}`" for path in dist_artifacts) if dist_artifacts else lines.append("  - none found")
    lines.extend(
        [
            "",
            "## Release Audit Output",
            "",
        ]
    )
    lines.extend(release_audit_lines)
    lines.extend(
        [
            "",
            "## Deterministic Commands",
            "",
        ]
    )
    lines.extend(f"- `{command}`" for command in commands)
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- This draft is offline documentation, not release approval.",
            "- PyPI publication status and GitHub release status are not asserted by this draft.",
            "- `TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are not approval states.",
            "- Downloads, stars, and badges are not correctness or trustworthiness proof.",
            "- Live provider/API behavior is not part of default CI evidence.",
            "",
            "## Post-Release Verification Command",
            "",
            "Run this only after a manual release has been created and published:",
            "",
            "```bash",
            POST_RELEASE_COMMAND.format(version=version),
            "```",
            "",
            "## Manual Release Boundary",
            "",
            "Publishing to PyPI, creating or pushing git tags, creating a GitHub release, and uploading release assets are manual actions. This script performs none of them.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_release_draft(
    root: Path,
    *,
    version: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    release_audit_json: Path | None = None,
) -> DraftResult:
    root = root.resolve()
    version_result = validate_version(version)
    if version_result.status != "PASS":
        return version_result

    claims_result = check_generated_claims_current(root)
    if claims_result.status != "PASS":
        return claims_result

    try:
        claim_payload = load_claim_payload(root)
        claim_ledger = load_claim_ledger(root)
        release_history_status, release_history_lines = extract_release_history(root, version)
        release_audit_path, release_audit_payload = load_release_audit(root, version, release_audit_json)
    except (OSError, json.JSONDecodeError) as exc:
        return DraftResult(
            "FAIL",
            "Release draft inputs could not be read.",
            details=(str(exc),),
        )

    draft = render_draft(
        root=root,
        version=version,
        claim_payload=claim_payload,
        claim_ledger=claim_ledger,
        release_history_status=release_history_status,
        release_history_lines=release_history_lines,
        release_audit_path=release_audit_path,
        release_audit_payload=release_audit_payload,
    )
    target_dir = output_dir if output_dir.is_absolute() else root / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"release_draft_v{version}.md"
    output_path.write_text(draft, encoding="utf-8", newline="\n")
    return DraftResult(
        "PASS",
        "GitHub release draft generated locally.",
        output_path=output_path,
        details=(_rel(output_path, root),),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="release version in X.Y.Z form")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--release-audit-json", type=Path, help="optional release audit JSON path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = generate_release_draft(
        args.root,
        version=args.version,
        output_dir=args.output_dir,
        release_audit_json=args.release_audit_json,
    )
    print(f"{result.status}: {result.message}")
    for detail in result.details:
        print(f"  - {detail}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
