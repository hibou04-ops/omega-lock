#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Offline-first release audit for omega-lock.

The default path is deterministic and local-only. PyPI and GitHub are queried
only when --network is passed explicitly. This script never publishes, tags, or
creates releases.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = "omega-lock"
IMPORT_PACKAGE = "omega_lock"
GITHUB_REPO = "hibou04-ops/omega-lock"
STATUSES = ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]


@dataclass(frozen=True)
class AuditResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


def _load_script(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_repo_consistency() -> ModuleType:
    return _load_script("omega_lock_repo_consistency", REPO_ROOT / "scripts" / "check_repo_consistency.py")


def _load_claim_generator() -> ModuleType:
    return _load_script("omega_lock_readme_claims", REPO_ROOT / "scripts" / "generate_readme_claims.py")


def _load_golden_cases() -> ModuleType:
    return _load_script("omega_lock_golden_audits", REPO_ROOT / "scripts" / "run_golden_audit_cases.py")


def _to_audit_result(result: Any, *, name: str | None = None) -> AuditResult:
    status = str(getattr(result, "status", "FAIL"))
    if status not in STATUSES:
        status = "FAIL"
    return AuditResult(
        name=name or str(getattr(result, "name", "unknown")),
        status=status,  # type: ignore[arg-type]
        message=str(getattr(result, "message", "")),
        details=tuple(str(item) for item in getattr(result, "details", ())),
    )


def _version_patterns(consistency: ModuleType) -> tuple[re.Pattern[str], ...]:
    patterns = getattr(consistency, "_current_version_patterns", None)
    if callable(patterns):
        return tuple(patterns())
    return (
        re.compile(r"(?:release|version)-(\d+\.\d+\.\d+)"),
        re.compile(r"pypi\.org/project/omega-lock/(\d+\.\d+\.\d+)/"),
        re.compile(r"omega-lock(?:\[[^\]]+\])?==(\d+\.\d+\.\d+)"),
        re.compile(r"omega_lock-(\d+\.\d+\.\d+)(?:-py3-none-any\.whl|\.tar\.gz)"),
        re.compile(r"git tag v(\d+\.\d+\.\d+)"),
        re.compile(r"git push origin v(\d+\.\d+\.\d+)"),
        re.compile(r"omega-lock/(\d+\.\d+\.\d+)/json"),
    )


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def check_intended_version(intended_version: str) -> AuditResult:
    if re.fullmatch(r"\d+\.\d+\.\d+", intended_version):
        return AuditResult(
            "intended-version",
            "PASS",
            f"Intended version is a normalized X.Y.Z value: {intended_version}.",
        )
    return AuditResult(
        "intended-version",
        "FAIL",
        "--intended-version must use X.Y.Z numeric form.",
        (f"received: {intended_version}",),
    )


def load_metadata(root: Path, consistency: ModuleType) -> tuple[Any, list[AuditResult]]:
    metadata, load_results = consistency.load_pyproject(root)
    return metadata, [_to_audit_result(result) for result in load_results]


def check_pyproject_version(metadata: Any, intended_version: str) -> AuditResult:
    found = getattr(metadata, "project_version", None)
    if found == intended_version:
        return AuditResult(
            "pyproject-version",
            "PASS",
            f"pyproject project.version is {intended_version}.",
        )
    return AuditResult(
        "pyproject-version",
        "FAIL",
        "pyproject project.version does not match the intended release version.",
        (f"expected: {intended_version}", f"found: {found!r}"),
    )


def check_init_version(root: Path, consistency: ModuleType, intended_version: str) -> AuditResult:
    init_version, init_error = consistency.read_init_version(root)
    if init_error is not None:
        return _to_audit_result(init_error)
    if init_version == intended_version:
        return AuditResult(
            "init-version",
            "PASS",
            f"{IMPORT_PACKAGE}.__version__ is {intended_version}.",
        )
    return AuditResult(
        "init-version",
        "FAIL",
        f"{IMPORT_PACKAGE}.__version__ does not match the intended release version.",
        (f"expected: {intended_version}", f"found: {init_version!r}"),
    )


def check_doc_versions(
    root: Path,
    consistency: ModuleType,
    docs: Sequence[str],
    intended_version: str,
    *,
    name: str,
    surface_label: str,
) -> AuditResult:
    missing = [doc for doc in docs if not (root / doc).exists()]
    if missing:
        return AuditResult(
            name,
            "FAIL",
            f"{surface_label} files are missing.",
            tuple(missing),
        )

    stale: list[str] = []
    patterns = _version_patterns(consistency)
    for doc in docs:
        path = root / doc
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in patterns:
                for match in pattern.finditer(line):
                    found = match.group(1)
                    if found != intended_version:
                        stale.append(
                            f"{doc}:{line_no}: expected {intended_version}, found {found}: "
                            f"{line.strip()[:140]}"
                        )

    if stale:
        return AuditResult(
            name,
            "FAIL",
            f"{surface_label} contain stale release-version references.",
            tuple(stale),
        )
    return AuditResult(
        name,
        "PASS",
        f"{surface_label} release-version references match {intended_version}.",
    )


def check_repo_naming(root: Path, consistency: ModuleType, metadata: Any) -> list[AuditResult]:
    results = [
        _to_audit_result(consistency.check_project_name(metadata)),
        _to_audit_result(consistency.check_import_package_dir(root)),
        _to_audit_result(consistency.check_naming_conflation(root)),
        _to_audit_result(consistency.check_cli_documentation(root, metadata.scripts)),
    ]
    return results


def check_version_match(consistency: ModuleType, metadata: Any, init_version: str | None) -> AuditResult:
    return _to_audit_result(consistency.check_version_match(metadata, init_version))


def check_changelog(root: Path, consistency: ModuleType) -> AuditResult:
    return _to_audit_result(consistency.check_changelog(root), name="changelog-status")


def check_generated_claims(root: Path) -> AuditResult:
    try:
        claims = _load_claim_generator()
        diagnostics = claims.check_outputs(root)
    except FileNotFoundError as exc:
        return AuditResult(
            "generated-claims",
            "FAIL",
            "Generated README claim files could not be checked because an input file is missing.",
            (str(exc),),
        )
    except ImportError as exc:
        return AuditResult(
            "generated-claims",
            "TOOLING_MISSING",
            "Generated README claim checker could not be loaded.",
            (str(exc),),
        )

    failures = [diagnostic.message for diagnostic in diagnostics if diagnostic.status == "FAIL"]
    if failures:
        return AuditResult(
            "generated-claims",
            "FAIL",
            "Generated README claim files are stale or invalid.",
            tuple(failures),
        )
    if diagnostics:
        details = tuple(f"{diagnostic.status}: {diagnostic.message}" for diagnostic in diagnostics)
        return AuditResult(
            "generated-claims",
            "PASS",
            "Generated README claim files are current.",
            details,
        )
    return AuditResult("generated-claims", "PASS", "Generated README claim files are current.")


def check_golden_artifacts(root: Path) -> AuditResult:
    try:
        golden = _load_golden_cases()
        fixture_dir = root / "tests" / "fixtures" / "golden_audits"
        diagnostics = golden.check_golden_cases(fixture_dir)
    except ModuleNotFoundError as exc:
        return AuditResult(
            "golden-audit-artifacts",
            "TOOLING_MISSING",
            "Golden audit artifact checker could not import a required local dependency.",
            (str(exc),),
        )
    except FileNotFoundError as exc:
        return AuditResult(
            "golden-audit-artifacts",
            "FAIL",
            "Golden audit artifacts could not be checked because a fixture is missing.",
            (str(exc),),
        )
    except ImportError as exc:
        return AuditResult(
            "golden-audit-artifacts",
            "TOOLING_MISSING",
            "Golden audit artifact checker could not be loaded.",
            (str(exc),),
        )

    failures = [f"{diagnostic.name}: {diagnostic.message}" for diagnostic in diagnostics if diagnostic.status == "FAIL"]
    if failures:
        return AuditResult(
            "golden-audit-artifacts",
            "FAIL",
            "Golden audit artifacts are stale, missing, or unexpected.",
            tuple(failures),
        )
    pass_count = sum(1 for diagnostic in diagnostics if diagnostic.status == "PASS")
    return AuditResult(
        "golden-audit-artifacts",
        "PASS",
        f"Golden audit artifacts are current ({pass_count} fixture(s)).",
    )


def check_dist_artifacts(root: Path, intended_version: str) -> AuditResult:
    dist = root / "dist"
    if not dist.exists():
        return AuditResult(
            "dist-artifacts",
            "WARN",
            "dist/ is absent; no local package artifacts were inspected.",
        )

    distribution_files = sorted(
        path.name
        for pattern in ("*.whl", "*.tar.gz")
        for path in dist.glob(pattern)
        if path.is_file()
    )
    if not distribution_files:
        return AuditResult(
            "dist-artifacts",
            "WARN",
            "dist/ exists but contains no local package artifacts.",
        )

    expected = {
        f"omega_lock-{intended_version}-py3-none-any.whl",
        f"omega_lock-{intended_version}.tar.gz",
    }
    stale: list[str] = []
    artifact_pattern = re.compile(r"^omega_lock-(\d+\.\d+\.\d+)(?:-py3-none-any\.whl|\.tar\.gz)$")
    for filename in distribution_files:
        match = artifact_pattern.fullmatch(filename)
        if match is None:
            stale.append(filename)
            continue
        if match.group(1) != intended_version:
            stale.append(filename)

    if stale:
        return AuditResult(
            "dist-artifacts",
            "FAIL",
            "dist/ contains omega-lock artifacts for a stale version.",
            tuple(stale),
        )

    missing_expected = sorted(expected - set(distribution_files))
    if missing_expected:
        return AuditResult(
            "dist-artifacts",
            "WARN",
            "dist/ artifacts are present, but the expected wheel/sdist pair is incomplete.",
            tuple(
                [f"missing: {name}" for name in missing_expected]
                + [f"present: {name}" for name in distribution_files]
            ),
        )

    details = tuple(f"present: {name}" for name in distribution_files if name in expected)
    return AuditResult(
        "dist-artifacts",
        "PASS",
        "dist/ contains current local wheel and sdist artifact filenames.",
        details,
    )


def check_git_tag_status(root: Path, intended_version: str) -> AuditResult:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "tag", "--list"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return AuditResult(
            "git-tags",
            "TOOLING_MISSING",
            "git is not available locally; local tag status could not be checked.",
        )
    except subprocess.TimeoutExpired:
        return AuditResult(
            "git-tags",
            "TOOLING_MISSING",
            "git tag --list timed out; local tag status could not be checked.",
        )

    if completed.returncode != 0:
        return AuditResult(
            "git-tags",
            "WARN",
            "Local git tag status is unavailable.",
            (completed.stderr.strip() or completed.stdout.strip() or "git tag --list failed",),
        )

    tags = frozenset(line.strip() for line in completed.stdout.splitlines() if line.strip())
    expected = f"v{intended_version}"
    if expected in tags:
        return AuditResult(
            "git-tags",
            "PASS",
            f"Local tag {expected} exists. This is not release approval.",
            (f"matching tag: {expected}",),
        )
    return AuditResult(
        "git-tags",
        "WARN",
        f"Local tag {expected} was not found. This is not release approval.",
        (f"local tag count: {len(tags)}",),
    )


def _http_json_status(url: str, *, success_message: str, not_found_message: str) -> AuditResult:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "omega-lock-release-audit"})
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                return AuditResult("network", "PASS", success_message)
            return AuditResult("network", "WARN", f"Unexpected HTTP status {response.status} for {url}.")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return AuditResult("network", "FAIL", not_found_message)
        return AuditResult(
            "network",
            "ENVIRONMENT_BLOCKED",
            f"Network check returned HTTP {exc.code}; status is not release approval.",
            (url,),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return AuditResult(
            "network",
            "ENVIRONMENT_BLOCKED",
            "Network check could not complete; status is not release approval.",
            (url, str(exc)),
        )


def check_pypi_status(intended_version: str, *, offline: bool) -> AuditResult:
    if offline:
        return AuditResult(
            "pypi-status",
            "WARN",
            "Offline mode: PyPI status was not checked and is not release approval.",
        )
    url = f"https://pypi.org/pypi/{PROJECT_NAME}/{intended_version}/json"
    result = _http_json_status(
        url,
        success_message=f"PyPI has {PROJECT_NAME} {intended_version}.",
        not_found_message=f"PyPI does not report {PROJECT_NAME} {intended_version}.",
    )
    return AuditResult("pypi-status", result.status, result.message, result.details)


def check_github_status(intended_version: str, *, offline: bool) -> AuditResult:
    if offline:
        return AuditResult(
            "github-release-status",
            "WARN",
            "Offline mode: GitHub release status was not checked and is not release approval.",
        )
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/v{intended_version}"
    result = _http_json_status(
        url,
        success_message=f"GitHub release v{intended_version} exists.",
        not_found_message=f"GitHub release v{intended_version} was not found.",
    )
    return AuditResult("github-release-status", result.status, result.message, result.details)


def run_release_audit(
    root: Path,
    *,
    intended_version: str,
    offline: bool = True,
) -> list[AuditResult]:
    root = root.resolve()
    consistency = _load_repo_consistency()
    metadata, load_results = load_metadata(root, consistency)
    results: list[AuditResult] = [check_intended_version(intended_version), *load_results]

    if load_results:
        results.extend(
            [
                check_generated_claims(root),
                check_golden_artifacts(root),
                check_dist_artifacts(root, intended_version),
                check_git_tag_status(root, intended_version),
                check_pypi_status(intended_version, offline=offline),
                check_github_status(intended_version, offline=offline),
            ]
        )
        return results

    init_version, init_error = consistency.read_init_version(root)
    results.append(check_pyproject_version(metadata, intended_version))
    results.append(check_init_version(root, consistency, intended_version))
    if init_error is None:
        results.append(check_version_match(consistency, metadata, init_version))
    results.append(
        check_doc_versions(
            root,
            consistency,
            ("README.md", "README_KR.md", "EASY_README.md", "EASY_README_KR.md"),
            intended_version,
            name="readme-family-versions",
            surface_label="README family",
        )
    )
    results.append(
        check_doc_versions(
            root,
            consistency,
            ("RELEASE.md",),
            intended_version,
            name="release-doc-versions",
            surface_label="RELEASE.md",
        )
    )
    results.extend(check_repo_naming(root, consistency, metadata))
    results.append(check_changelog(root, consistency))
    results.append(check_generated_claims(root))
    results.append(check_golden_artifacts(root))
    results.append(check_dist_artifacts(root, intended_version))
    results.append(check_git_tag_status(root, intended_version))
    results.append(check_pypi_status(intended_version, offline=offline))
    results.append(check_github_status(intended_version, offline=offline))
    return results


def summarize(results: Sequence[AuditResult]) -> dict[str, int]:
    return {status: sum(1 for result in results if result.status == status) for status in STATUSES}


def to_payload(
    results: Sequence[AuditResult],
    *,
    root: Path,
    intended_version: str,
    offline: bool,
) -> dict[str, Any]:
    return {
        "intended_version": intended_version,
        "mode": "offline" if offline else "network",
        "repository": GITHUB_REPO,
        "results": [
            {
                "details": list(result.details),
                "message": result.message,
                "name": result.name,
                "status": result.status,
            }
            for result in results
        ],
        "root": str(root.resolve()),
        "schema_version": 1,
        "summary": summarize(results),
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_text(results: Sequence[AuditResult], *, root: Path, intended_version: str, offline: bool) -> str:
    lines = [
        "Release audit",
        f"Root: {root.resolve()}",
        f"Intended version: {intended_version}",
        f"Mode: {'offline' if offline else 'network'}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")
        for detail in result.details:
            lines.append(f"  - {detail}")
    counts = summarize(results)
    lines.extend(
        [
            "",
            "Summary: " + ", ".join(f"{status}={count}" for status, count in counts.items() if count),
        ]
    )
    return "\n".join(lines)


def has_blocking_status(results: Sequence[AuditResult], *, strict: bool) -> bool:
    blocking = {"FAIL"}
    if strict:
        blocking.update({"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"})
    return any(result.status in blocking for result in results)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intended-version", required=True, help="intended release version in X.Y.Z form")
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    parser.add_argument("--strict", action="store_true", help="fail on TOOLING_MISSING or ENVIRONMENT_BLOCKED")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="force local-only checks")
    network.add_argument("--network", action="store_true", help="explicitly allow PyPI/GitHub status checks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    offline = not args.network
    results = run_release_audit(args.root, intended_version=args.intended_version, offline=offline)
    if args.json:
        print(
            render_json(
                to_payload(
                    results,
                    root=args.root,
                    intended_version=args.intended_version,
                    offline=offline,
                )
            ),
            end="",
        )
    else:
        print(render_text(results, root=args.root, intended_version=args.intended_version, offline=offline))
    return 1 if has_blocking_status(results, strict=args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
