#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Offline release-candidate scope-freeze checks for omega-lock.

The check is reversible and local: it reads a release-candidate marker from
docs/RELEASE_CANDIDATE.md, inspects git state when available, and verifies that
generated artifacts have not drifted. It never publishes, tags, creates GitHub
releases, or uses network access.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
RC_DOC = Path("docs/RELEASE_CANDIDATE.md")
UNSET_VALUES = frozenset({"", "UNSET", "RC_MARKER_UNSET", "RC_AUDIT_UNSET", "TODO", "TBD", "N/A"})
STATUSES = ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]

GENERATED_ARTIFACT_PATHS = frozenset(
    {
        "docs/claims/generated_readme_claims.md",
        "docs/claims/generated_readme_claims.json",
    }
)
GENERATED_ARTIFACT_PREFIXES = ("tests/fixtures/golden_audits/",)
CODE_PREFIXES = ("src/", "scripts/", "examples/", "tests/")
CODE_SUFFIXES = (".py",)
RELEVANT_PATHS = (
    "pyproject.toml",
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
    "RELEASE.md",
    "docs",
    "examples",
    "scripts",
    "src",
    "tests",
    ".github",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ReleaseCandidateConfig:
    marker: str | None
    release_audit_ref: str | None
    release_audit_command: str | None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_unset(value: str | None) -> bool:
    return value is None or value.strip() in UNSET_VALUES


def _normalize_config_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value in UNSET_VALUES else value


def _load_script(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_release_candidate_config(root: Path, doc_path: Path = RC_DOC) -> tuple[ReleaseCandidateConfig, CheckResult]:
    path = doc_path if doc_path.is_absolute() else root / doc_path
    if not path.exists():
        return (
            ReleaseCandidateConfig(None, None, None),
            CheckResult(
                "release-candidate-config",
                "WARN",
                "Release-candidate marker document is absent; freeze marker is not established.",
                (_rel(path, root),),
            ),
        )

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {
            "release_candidate_marker",
            "release_audit_after_commit",
            "release_audit_command",
        }:
            values[key] = value.strip().strip("`")

    config = ReleaseCandidateConfig(
        marker=_normalize_config_value(values.get("release_candidate_marker")),
        release_audit_ref=_normalize_config_value(values.get("release_audit_after_commit")),
        release_audit_command=_normalize_config_value(values.get("release_audit_command")),
    )
    status: Status = "PASS" if config.marker else "WARN"
    message = (
        "Release-candidate marker is configured."
        if config.marker
        else "Release-candidate marker is unset; freeze checks are advisory until a marker is recorded."
    )
    return (
        config,
        CheckResult(
            "release-candidate-config",
            status,
            message,
            (
                f"marker: {config.marker or 'RC_MARKER_UNSET'}",
                f"release_audit_after_commit: {config.release_audit_ref or 'RC_AUDIT_UNSET'}",
            ),
        ),
    )


def run_git(root: Path, args: Sequence[str], *, timeout_s: int = 30) -> GitResult:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return GitResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return GitResult(124, exc.stdout or "", exc.stderr or "git command timed out")
    return GitResult(completed.returncode, completed.stdout, completed.stderr)


def check_git_available(root: Path, git_runner=run_git) -> CheckResult:
    result = git_runner(root, ["rev-parse", "--is-inside-work-tree"])
    if result.returncode != 0 or result.stdout.strip() != "true":
        return CheckResult(
            "git-state",
            "TOOLING_MISSING",
            "Git metadata is unavailable; scope-freeze state cannot be inspected.",
            tuple(filter(None, (result.stderr.strip(), result.stdout.strip()))),
        )
    return CheckResult("git-state", "PASS", "Git metadata is available.")


def _git_lines(result: GitResult) -> tuple[str, ...]:
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def resolve_git_ref(root: Path, ref: str, git_runner=run_git) -> str | None:
    result = git_runner(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_changed_files_since_marker(root: Path, marker: str, git_runner=run_git) -> tuple[CheckResult, tuple[str, ...]]:
    resolved = resolve_git_ref(root, marker, git_runner)
    if resolved is None:
        return (
            CheckResult(
                "changed-since-rc-marker",
                "FAIL",
                "Release-candidate marker cannot be resolved as a local git commit.",
                (f"marker: {marker}",),
            ),
            (),
        )

    diff = git_runner(root, ["diff", "--name-only", resolved, "--"])
    if diff.returncode != 0:
        return (
            CheckResult(
                "changed-since-rc-marker",
                "FAIL",
                "Git diff from release-candidate marker failed.",
                tuple(filter(None, (diff.stderr.strip(), diff.stdout.strip()))),
            ),
            (),
        )
    untracked = git_runner(root, ["ls-files", "--others", "--exclude-standard"])
    if untracked.returncode != 0:
        return (
            CheckResult(
                "changed-since-rc-marker",
                "FAIL",
                "Git untracked-file inspection failed.",
                tuple(filter(None, (untracked.stderr.strip(), untracked.stdout.strip()))),
            ),
            (),
        )

    changed = tuple(sorted(set(_git_lines(diff) + _git_lines(untracked))))
    if not changed:
        return (
            CheckResult("changed-since-rc-marker", "PASS", "No files changed since the release-candidate marker."),
            changed,
        )
    return (
        CheckResult(
            "changed-since-rc-marker",
            "WARN",
            "Files changed since the release-candidate marker.",
            changed,
        ),
        changed,
    )


def is_code_file(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized == "pyproject.toml":
        return True
    return normalized.endswith(CODE_SUFFIXES) and normalized.startswith(CODE_PREFIXES)


def is_generated_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized in GENERATED_ARTIFACT_PATHS or normalized.startswith(GENERATED_ARTIFACT_PREFIXES)


def check_code_changes_after_generated_artifacts(marker: str | None, changed_files: Sequence[str]) -> CheckResult:
    if _is_unset(marker):
        return CheckResult(
            "code-after-generated-artifacts",
            "WARN",
            "Release-candidate marker is unset; code/artifact ordering is advisory only.",
        )

    code_changes = tuple(path for path in changed_files if is_code_file(path))
    generated_changes = tuple(path for path in changed_files if is_generated_artifact(path))
    if not code_changes:
        return CheckResult(
            "code-after-generated-artifacts",
            "PASS",
            "No code files changed since the release-candidate marker.",
        )
    if generated_changes:
        return CheckResult(
            "code-after-generated-artifacts",
            "PASS",
            "Code changes are accompanied by generated artifact changes; stale checks will verify content.",
            (
                f"code files: {len(code_changes)}",
                f"generated artifacts: {len(generated_changes)}",
            ),
        )
    return CheckResult(
        "code-after-generated-artifacts",
        "FAIL",
        "Code files changed after the release-candidate marker without generated artifact updates.",
        code_changes,
    )


def check_generated_claims(root: Path) -> CheckResult:
    try:
        generator = _load_script("omega_lock_readme_claims_scope_freeze", root / "scripts" / "generate_readme_claims.py")
        diagnostics = generator.check_outputs(root)
    except Exception as exc:  # pragma: no cover - defensive for broken local tooling
        return CheckResult(
            "generated-claims",
            "TOOLING_MISSING",
            "Generated README claim check could not run.",
            (str(exc),),
        )

    failures = tuple(
        str(getattr(diagnostic, "message", diagnostic))
        for diagnostic in diagnostics
        if getattr(diagnostic, "status", "") == "FAIL"
    )
    if failures:
        return CheckResult(
            "generated-claims",
            "FAIL",
            "Generated README claim files are stale or invalid.",
            failures,
        )
    return CheckResult("generated-claims", "PASS", "Generated README claim files are current.")


def check_golden_artifacts(root: Path) -> CheckResult:
    try:
        golden = _load_script("omega_lock_golden_scope_freeze", root / "scripts" / "run_golden_audit_cases.py")
        diagnostics = golden.check_golden_cases()
    except Exception as exc:  # pragma: no cover - defensive for broken local tooling
        return CheckResult(
            "golden-audit-artifacts",
            "TOOLING_MISSING",
            "Golden audit artifact check could not run.",
            (str(exc),),
        )

    failures = tuple(
        f"{getattr(diagnostic, 'name', 'unknown')}: {getattr(diagnostic, 'message', diagnostic)}"
        for diagnostic in diagnostics
        if getattr(diagnostic, "status", "") == "FAIL"
    )
    if failures:
        return CheckResult(
            "golden-audit-artifacts",
            "FAIL",
            "Golden audit artifacts are stale or invalid.",
            failures,
        )
    return CheckResult("golden-audit-artifacts", "PASS", "Golden audit artifacts are current.")


def _has_dirty_relevant_files(changed_files: Sequence[str]) -> bool:
    return any(
        path == "pyproject.toml"
        or path.startswith(("src/", "scripts/", "examples/", "tests/", "docs/", ".github/"))
        or path in {"README.md", "README_KR.md", "EASY_README.md", "EASY_README_KR.md", "RELEASE.md"}
        for path in changed_files
    )


def check_release_audit_freshness(
    root: Path,
    *,
    marker: str | None,
    release_audit_ref: str | None,
    changed_files: Sequence[str],
    git_runner=run_git,
) -> CheckResult:
    if _is_unset(marker):
        return CheckResult(
            "release-audit-freshness",
            "WARN",
            "Release-candidate marker is unset; release audit freshness is advisory only.",
        )
    if _is_unset(release_audit_ref):
        return CheckResult(
            "release-audit-freshness",
            "FAIL",
            "Release audit run is not recorded for this release-candidate marker.",
            ("set release_audit_after_commit after running scripts/release_audit.py offline",),
        )
    if _has_dirty_relevant_files(changed_files):
        return CheckResult(
            "release-audit-freshness",
            "FAIL",
            "Relevant files changed after the recorded release audit; rerun release audit after the latest change.",
            tuple(changed_files),
        )

    latest = git_runner(root, ["log", "-n", "1", "--format=%H", "--", *RELEVANT_PATHS])
    if latest.returncode != 0:
        return CheckResult(
            "release-audit-freshness",
            "FAIL",
            "Could not identify the latest relevant git change.",
            tuple(filter(None, (latest.stderr.strip(), latest.stdout.strip()))),
        )
    latest_ref = latest.stdout.strip()
    if not latest_ref:
        return CheckResult(
            "release-audit-freshness",
            "WARN",
            "No relevant git history was found for release audit freshness.",
        )

    audit_ref = resolve_git_ref(root, release_audit_ref, git_runner)
    if audit_ref is None:
        return CheckResult(
            "release-audit-freshness",
            "FAIL",
            "Recorded release audit ref cannot be resolved as a local git commit.",
            (f"release_audit_after_commit: {release_audit_ref}",),
        )

    ancestor = git_runner(root, ["merge-base", "--is-ancestor", latest_ref, audit_ref])
    if ancestor.returncode == 0:
        return CheckResult(
            "release-audit-freshness",
            "PASS",
            "Recorded release audit ref is at or after the latest relevant git change.",
            (f"latest_relevant_change: {latest_ref}", f"release_audit_after_commit: {audit_ref}"),
        )
    return CheckResult(
        "release-audit-freshness",
        "FAIL",
        "Recorded release audit ref is older than the latest relevant git change.",
        (f"latest_relevant_change: {latest_ref}", f"release_audit_after_commit: {audit_ref}"),
    )


def run_scope_freeze_check(
    root: Path,
    *,
    marker_override: str | None = None,
    release_audit_ref_override: str | None = None,
    git_runner=run_git,
) -> list[CheckResult]:
    root = root.resolve()
    config, config_result = load_release_candidate_config(root)
    marker = _normalize_config_value(marker_override) if marker_override is not None else config.marker
    release_audit_ref = (
        _normalize_config_value(release_audit_ref_override)
        if release_audit_ref_override is not None
        else config.release_audit_ref
    )

    results: list[CheckResult] = [config_result]
    if marker_override is not None or release_audit_ref_override is not None:
        results.append(
            CheckResult(
                "release-candidate-overrides",
                "WARN",
                "Command-line release-candidate override is active.",
                (
                    f"marker: {marker or 'RC_MARKER_UNSET'}",
                    f"release_audit_after_commit: {release_audit_ref or 'RC_AUDIT_UNSET'}",
                ),
            )
        )

    git_state = check_git_available(root, git_runner)
    results.append(git_state)

    changed_files: tuple[str, ...] = ()
    if git_state.status == "PASS" and marker:
        changed_result, changed_files = collect_changed_files_since_marker(root, marker, git_runner)
        results.append(changed_result)
    elif git_state.status == "PASS":
        results.append(
            CheckResult(
                "changed-since-rc-marker",
                "WARN",
                "Release-candidate marker is unset; changed files since marker were not inspected.",
            )
        )

    results.append(check_code_changes_after_generated_artifacts(marker, changed_files))
    results.append(check_generated_claims(root))
    results.append(check_golden_artifacts(root))
    if git_state.status == "PASS":
        results.append(
            check_release_audit_freshness(
                root,
                marker=marker,
                release_audit_ref=release_audit_ref,
                changed_files=changed_files,
                git_runner=git_runner,
            )
        )
    else:
        results.append(
            CheckResult(
                "release-audit-freshness",
                "TOOLING_MISSING",
                "Git metadata is unavailable; release audit freshness cannot be established.",
            )
        )
    return results


def has_blocking_status(results: Sequence[CheckResult]) -> bool:
    return any(result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"} for result in results)


def render_text(results: Sequence[CheckResult], *, root: Path) -> str:
    lines = [
        "Scope freeze check",
        f"Root: {root}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")
        for detail in result.details:
            lines.append(f"  - {detail}")
    counts = {status: sum(1 for result in results if result.status == status) for status in STATUSES}
    lines.append("")
    lines.append(
        "Summary: "
        + ", ".join(f"{status}={count}" for status, count in counts.items() if count)
    )
    return "\n".join(lines)


def to_payload(results: Sequence[CheckResult], *, root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "root": str(root),
        "statuses": {status: sum(1 for result in results if result.status == status) for status in STATUSES},
        "results": [
            {
                "name": result.name,
                "status": result.status,
                "message": result.message,
                "details": list(result.details),
            }
            for result in results
        ],
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the release-candidate freeze check")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--marker", help="override release_candidate_marker from docs/RELEASE_CANDIDATE.md")
    parser.add_argument(
        "--release-audit-ref",
        help="override release_audit_after_commit from docs/RELEASE_CANDIDATE.md",
    )
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    results = run_scope_freeze_check(
        root,
        marker_override=args.marker,
        release_audit_ref_override=args.release_audit_ref,
    )
    if args.json:
        print(json.dumps(to_payload(results, root=root), indent=2, sort_keys=True))
    else:
        print(render_text(results, root=root))
    return 1 if has_blocking_status(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
