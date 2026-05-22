#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Offline repository surface consistency checker.

This script performs only deterministic local checks. It does not query PyPI,
GitHub, package registries, or live provider APIs.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only.
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = (
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
    "RELEASE.md",
)
README_FILES = (
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
)
PROJECT_NAME = "omega-lock"
IMPORT_PACKAGE = "omega_lock"
IMPORT_PACKAGE_DIR = Path("src") / IMPORT_PACKAGE
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoMetadata:
    project_name: str | None
    project_version: str | None
    scripts: frozenset[str]
    pyproject_loaded: bool


NEGATIVE_CLI_CONTEXT = (
    "does not",
    "doesn't",
    "not currently",
    "no console",
    "no cli",
    "none currently",
    "currently none",
    "not installed",
    "is not installed",
    "do not install",
    "does not ship",
    "currently ship",
    "제공하지",
    "현재 없음",
    "없습니다",
    "아닙니다",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_pyproject(root: Path) -> tuple[RepoMetadata, list[CheckResult]]:
    if tomllib is None:
        return (
            RepoMetadata(None, None, frozenset(), False),
            [
                CheckResult(
                    "tomllib",
                    "TOOLING_MISSING",
                    "Python 3.11+ tomllib is required to read pyproject.toml.",
                )
            ],
        )

    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return (
            RepoMetadata(None, None, frozenset(), False),
            [CheckResult("pyproject", "FAIL", "pyproject.toml is missing.")],
        )

    try:
        data = tomllib.loads(_read_text(pyproject_path))
    except tomllib.TOMLDecodeError as exc:
        return (
            RepoMetadata(None, None, frozenset(), False),
            [CheckResult("pyproject", "FAIL", f"pyproject.toml is invalid TOML: {exc}")],
        )

    project = data.get("project", {})
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict):
        scripts = {}

    return (
        RepoMetadata(
            project_name=project.get("name") if isinstance(project.get("name"), str) else None,
            project_version=(
                project.get("version") if isinstance(project.get("version"), str) else None
            ),
            scripts=frozenset(str(key) for key in scripts),
            pyproject_loaded=True,
        ),
        [],
    )


def read_init_version(root: Path) -> tuple[str | None, CheckResult | None]:
    init_path = root / IMPORT_PACKAGE_DIR / "__init__.py"
    if not init_path.exists():
        return None, CheckResult(
            "package-init",
            "FAIL",
            f"{IMPORT_PACKAGE_DIR.as_posix()}/__init__.py is missing.",
        )

    try:
        tree = ast.parse(_read_text(init_path), filename=str(init_path))
    except SyntaxError as exc:
        return None, CheckResult("package-init", "FAIL", f"Cannot parse __init__.py: {exc}")

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value, None
        return None, CheckResult(
            "package-version",
            "FAIL",
            "__version__ exists but is not a string literal.",
        )

    return None, CheckResult(
        "package-version",
        "FAIL",
        f"{IMPORT_PACKAGE_DIR.as_posix()}/__init__.py does not define __version__.",
    )


def check_project_name(metadata: RepoMetadata) -> CheckResult:
    if metadata.project_name == PROJECT_NAME:
        return CheckResult(
            "project-name",
            "PASS",
            f"pyproject project.name is {PROJECT_NAME}.",
        )
    return CheckResult(
        "project-name",
        "FAIL",
        f"pyproject project.name must be {PROJECT_NAME!r}; found {metadata.project_name!r}.",
    )


def check_import_package_dir(root: Path) -> CheckResult:
    package_dir = root / IMPORT_PACKAGE_DIR
    if package_dir.is_dir():
        return CheckResult(
            "import-package-dir",
            "PASS",
            f"Import package directory exists at {IMPORT_PACKAGE_DIR.as_posix()}.",
        )
    return CheckResult(
        "import-package-dir",
        "FAIL",
        f"Import package directory {IMPORT_PACKAGE_DIR.as_posix()} is missing.",
    )


def check_version_match(metadata: RepoMetadata, init_version: str | None) -> CheckResult:
    if not metadata.project_version:
        return CheckResult("version-match", "FAIL", "pyproject project.version is missing.")
    if init_version is None:
        return CheckResult("version-match", "FAIL", "__version__ could not be read.")
    if metadata.project_version == init_version:
        return CheckResult(
            "version-match",
            "PASS",
            f"pyproject version and omega_lock.__version__ are {metadata.project_version}.",
        )
    return CheckResult(
        "version-match",
        "FAIL",
        "pyproject version and omega_lock.__version__ differ.",
        (f"pyproject: {metadata.project_version}", f"__version__: {init_version}"),
    )


def _doc_paths(root: Path) -> list[Path]:
    return [root / doc for doc in DOC_FILES]


def check_required_docs(root: Path) -> CheckResult:
    missing = [doc for doc in DOC_FILES if not (root / doc).exists()]
    if not missing:
        return CheckResult("required-docs", "PASS", "Required documentation files exist.")
    return CheckResult(
        "required-docs",
        "FAIL",
        "Required documentation files are missing.",
        tuple(missing),
    )


def _current_version_patterns() -> tuple[re.Pattern[str], ...]:
    return (
        re.compile(r"(?:release|version)-(\d+\.\d+\.\d+)"),
        re.compile(r"pypi\.org/project/omega-lock/(\d+\.\d+\.\d+)/"),
        re.compile(r"omega-lock(?:\[[^\]]+\])?==(\d+\.\d+\.\d+)"),
        re.compile(r"^\s*##\s+What Changed in (\d+\.\d+\.\d+)", re.MULTILINE),
        re.compile(r"^\s*##\s+(\d+\.\d+\.\d+)에서 바뀐 점", re.MULTILINE),
        re.compile(r"^\s*##\s+(\d+\.\d+\.\d+) Release Note", re.MULTILINE),
        re.compile(r"For (\d+\.\d+\.\d+), the expected files are:"),
        re.compile(r"omega_lock-(\d+\.\d+\.\d+)(?:-py3-none-any\.whl|\.tar\.gz)"),
        re.compile(r"Prepare release (\d+\.\d+\.\d+)"),
        re.compile(r"git tag v(\d+\.\d+\.\d+)"),
        re.compile(r"git push origin v(\d+\.\d+\.\d+)"),
        re.compile(r"omega-lock/(\d+\.\d+\.\d+)/json"),
    )


def check_stale_current_versions(root: Path, version: str | None) -> CheckResult:
    if not version:
        return CheckResult(
            "current-version-surfaces",
            "FAIL",
            "Cannot check documentation versions without pyproject project.version.",
        )

    stale: list[str] = []
    patterns = _current_version_patterns()
    for path in _doc_paths(root):
        if not path.exists():
            continue
        text = _read_text(path)
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in patterns:
                for match in pattern.finditer(line):
                    found = match.group(1)
                    if found != version:
                        stale.append(
                            f"{_rel(path, root)}:{line_no}: expected {version}, found {found}: "
                            f"{line.strip()[:140]}"
                        )

    if stale:
        return CheckResult(
            "current-version-surfaces",
            "FAIL",
            "Current release surfaces contain stale versions.",
            tuple(stale),
        )
    return CheckResult(
        "current-version-surfaces",
        "PASS",
        "Current release badges, install pins, release commands, and artifact names match pyproject version.",
    )


def check_readme_badges_and_installs(root: Path, version: str | None) -> CheckResult:
    if not version:
        return CheckResult(
            "readme-badges-installs",
            "FAIL",
            "Cannot check README version surfaces without pyproject project.version.",
        )

    issues: list[str] = []
    warnings: list[str] = []
    for doc in README_FILES:
        path = root / doc
        if not path.exists():
            continue
        text = _read_text(path)
        release_badges = re.findall(r"(?:release|version)-(\d+\.\d+\.\d+)", text)
        install_pins = re.findall(r"omega-lock(?:\[[^\]]+\])?==(\d+\.\d+\.\d+)", text)
        if not release_badges:
            warnings.append(f"{doc}: no static version badge found.")
        if not install_pins:
            warnings.append(f"{doc}: no exact omega-lock install pin found.")
        for found in release_badges:
            if found != version:
                issues.append(f"{doc}: release badge uses {found}, expected {version}.")
        for found in install_pins:
            if found != version:
                issues.append(f"{doc}: install pin uses {found}, expected {version}.")

    if issues:
        return CheckResult(
            "readme-badges-installs",
            "FAIL",
            "README badges or install snippets are version-inconsistent.",
            tuple(issues),
        )
    if warnings:
        return CheckResult(
            "readme-badges-installs",
            "WARN",
            "README version surfaces are consistent, but some expected surfaces are absent.",
            tuple(warnings),
        )
    return CheckResult(
        "readme-badges-installs",
        "PASS",
        "README release badges and omega-lock install pins match pyproject version.",
    )


def check_naming_conflation(root: Path) -> CheckResult:
    patterns = (
        (
            re.compile(r"pip\s+install\s+['\"]?omega_lock(?:\[|==|\s|$)"),
            "PyPI install command must use distribution name omega-lock, not omega_lock.",
        ),
        (
            re.compile(r"pypi\.org/project/omega_lock\b"),
            "PyPI project URLs must use distribution name omega-lock, not omega_lock.",
        ),
        (
            re.compile(r"github\.com/hibou04-ops/omega_lock\b"),
            "GitHub repo URL must use hibou04-ops/omega-lock, not omega_lock.",
        ),
        (
            re.compile(r"\b(?:from|import)\s+omega-lock\b"),
            "Python imports must use package name omega_lock, not omega-lock.",
        ),
        (
            re.compile(r"\bsrc/omega-lock\b"),
            "Source package path must be src/omega_lock, not src/omega-lock.",
        ),
    )
    issues: list[str] = []
    for path in _doc_paths(root):
        if not path.exists():
            continue
        for line_no, line in enumerate(_read_text(path).splitlines(), 1):
            for pattern, message in patterns:
                if pattern.search(line):
                    issues.append(
                        f"{_rel(path, root)}:{line_no}: {message} :: {line.strip()[:140]}"
                    )

    if issues:
        return CheckResult(
            "naming-conflation",
            "FAIL",
            "Documentation conflates repo, distribution, import package, or source path names.",
            tuple(issues),
        )
    return CheckResult(
        "naming-conflation",
        "PASS",
        "Documentation keeps repo, distribution, import package, and source path names distinct.",
    )


def _has_negative_cli_context(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in NEGATIVE_CLI_CONTEXT)


def _strip_prompt(line: str) -> str:
    return re.sub(r"^\s*(?:[$>]\s*)?", "", line).strip()


def _looks_like_cli_invocation(candidate: str) -> tuple[bool, str]:
    parts = candidate.split(maxsplit=1)
    if len(parts) != 2:
        return False, parts[0] if parts else ""

    command, rest = parts[0], parts[1].strip()
    if command not in {PROJECT_NAME, IMPORT_PACKAGE}:
        return False, command
    if not rest or rest.startswith(("=", ":", ",", ".")):
        return False, command

    first_arg = rest.split(maxsplit=1)[0].lower()
    prose_markers = {
        "does",
        "doesn't",
        "is",
        "isn't",
        "requires",
        "require",
        "required",
        "not",
    }
    if first_arg in prose_markers:
        return False, command
    return True, command


def _documented_cli_commands(text: str) -> list[tuple[int, str, str]]:
    commands: list[tuple[int, str, str]] = []
    in_fence = False

    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue

        if in_fence:
            command_line = _strip_prompt(line)
            is_cli, command = _looks_like_cli_invocation(command_line)
            if is_cli:
                commands.append((line_no, command, line.strip()))

        for inline in re.findall(r"`([^`]+)`", line):
            candidate = _strip_prompt(inline)
            is_cli, command = _looks_like_cli_invocation(candidate)
            if is_cli:
                commands.append((line_no, command, inline.strip()))

    return commands


def check_cli_documentation(root: Path, scripts: frozenset[str]) -> CheckResult:
    issues: list[str] = []
    for path in _doc_paths(root):
        if not path.exists():
            continue
        lines = _read_text(path).splitlines()
        for line_no, command, snippet in _documented_cli_commands("\n".join(lines)):
            start = max(0, line_no - 3)
            end = min(len(lines), line_no + 1)
            context = " ".join(lines[start:end]) if lines else snippet
            if _has_negative_cli_context(context):
                continue
            if command not in scripts:
                issues.append(
                    f"{_rel(path, root)}:{line_no}: `{snippet}` is documented as a command, "
                    f"but [project.scripts] does not define `{command}`."
                )

    if issues:
        return CheckResult(
            "cli-documentation",
            "FAIL",
            "Documentation references unsupported installed console commands.",
            tuple(issues),
        )
    if scripts:
        return CheckResult(
            "cli-documentation",
            "PASS",
            f"Documented package commands are backed by [project.scripts]: {', '.join(sorted(scripts))}.",
        )
    return CheckResult(
        "cli-documentation",
        "PASS",
        "No installed console command is documented; [project.scripts] is empty.",
    )


def check_changelog(root: Path) -> CheckResult:
    if (root / "CHANGELOG.md").exists():
        return CheckResult("changelog", "PASS", "Standalone CHANGELOG.md exists.")
    return CheckResult(
        "changelog",
        "WARN",
        "Standalone CHANGELOG.md is absent; release history appears to live in README/RELEASE surfaces.",
    )


def run_checks(root: Path) -> list[CheckResult]:
    root = root.resolve()
    metadata, load_results = load_pyproject(root)
    results: list[CheckResult] = list(load_results)

    if not metadata.pyproject_loaded:
        results.extend(
            [
                check_import_package_dir(root),
                check_required_docs(root),
                check_changelog(root),
            ]
        )
        return results

    init_version, init_error = read_init_version(root)
    results.append(check_project_name(metadata))
    results.append(check_import_package_dir(root))
    if init_error is not None:
        results.append(init_error)
    results.append(check_version_match(metadata, init_version))
    results.append(check_required_docs(root))
    results.append(check_stale_current_versions(root, metadata.project_version))
    results.append(check_naming_conflation(root))
    results.append(check_readme_badges_and_installs(root, metadata.project_version))
    results.append(check_cli_documentation(root, metadata.scripts))
    results.append(check_changelog(root))
    return results


def has_blocking_status(results: list[CheckResult], *, strict: bool) -> bool:
    blocking = {"FAIL"}
    if strict:
        blocking.update({"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"})
    return any(result.status in blocking for result in results)


def render_results(results: list[CheckResult], root: Path) -> str:
    counts = {status: 0 for status in ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")}
    for result in results:
        counts[result.status] += 1

    lines = [
        "Repository consistency check",
        f"Root: {root.resolve()}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")
        for detail in result.details:
            lines.append(f"  - {detail}")
    lines.extend(
        [
            "",
            "Summary: "
            + ", ".join(f"{status}={count}" for status, count in counts.items() if count),
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run offline consistency checks")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat TOOLING_MISSING and ENVIRONMENT_BLOCKED statuses as blocking",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to inspect; defaults to this script's repository",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.check:
        print("No action selected. Use --check.", file=sys.stderr)
        return 2

    results = run_checks(args.root)
    print(render_results(results, args.root))
    return 1 if has_blocking_status(results, strict=args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
