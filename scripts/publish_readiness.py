#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Aggregate local, non-publishing readiness checks for omega-lock."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
STATUSES = ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")
BLOCKING_STATUSES = frozenset({"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"})
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]
CommandFactory = Callable[[Path], tuple[str, ...]]


@dataclass(frozen=True)
class StepSpec:
    name: str
    command_factory: CommandFactory


@dataclass(frozen=True)
class StepResult:
    name: str
    status: Status
    message: str
    command: tuple[str, ...] = ()
    returncode: int | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def _script(path: str) -> str:
    return str(Path(path))


def _fixed_command(*parts: str) -> CommandFactory:
    command = tuple(parts)
    return lambda _root: command


def _twine_command(python_executable: str, intended_version: str) -> CommandFactory:
    def command(root: Path) -> tuple[str, ...]:
        dist = root / "dist"
        artifacts = tuple(
            str(path)
            for path in (
                dist / f"omega_lock-{intended_version}-py3-none-any.whl",
                dist / f"omega_lock-{intended_version}.tar.gz",
            )
            if path.is_file()
        )
        return (python_executable, "-m", "twine", "check", *artifacts)

    return command


def build_step_specs(
    *,
    intended_version: str,
    offline: bool,
    python_executable: str = sys.executable,
) -> list[StepSpec]:
    release_mode = "--offline" if offline else "--network"
    return [
        StepSpec(
            "repo-consistency",
            _fixed_command(
                python_executable,
                _script("scripts/check_repo_consistency.py"),
                "--check",
                "--strict",
            ),
        ),
        StepSpec(
            "readme-claims",
            _fixed_command(
                python_executable,
                _script("scripts/generate_readme_claims.py"),
                "--check",
            ),
        ),
        StepSpec(
            "golden-audits",
            _fixed_command(
                python_executable,
                _script("scripts/run_golden_audit_cases.py"),
                "--check",
            ),
        ),
        StepSpec(
            "demo-replay-check",
            _fixed_command(
                python_executable,
                _script("examples/demo_replay.py"),
                "--check",
            ),
        ),
        StepSpec(
            "demo-sram-check",
            _fixed_command(
                python_executable,
                _script("examples/demo_sram.py"),
                "--check",
            ),
        ),
        StepSpec("pytest", _fixed_command(python_executable, "-m", "pytest", "-q")),
        StepSpec("pyright", _fixed_command(python_executable, "-m", "pyright", "src", "tests")),
        StepSpec("ruff", _fixed_command(python_executable, "-m", "ruff", "check", "src", "tests")),
        StepSpec("build", _fixed_command(python_executable, "-m", "build", "--no-isolation")),
        StepSpec("twine-check", _twine_command(python_executable, intended_version)),
        StepSpec(
            "wheel-smoke-install",
            _fixed_command(
                python_executable,
                _script("scripts/wheel_smoke_install.py"),
                "--dist-dir",
                "dist",
                "--intended-version",
                intended_version,
            ),
        ),
        StepSpec(
            "release-audit",
            _fixed_command(
                python_executable,
                _script("scripts/release_audit.py"),
                "--intended-version",
                intended_version,
                release_mode,
                "--strict",
            ),
        ),
    ]


def _tail_lines(text: str, limit: int = 12) -> tuple[str, ...]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return tuple(lines[-limit:])


def run_command(command: Sequence[str], *, cwd: Path, timeout: int = 300) -> CommandResult | StepResult:
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        return StepResult(
            "subprocess",
            "TOOLING_MISSING",
            "Required executable is unavailable.",
            tuple(command),
            None,
            (str(exc),),
        )
    except subprocess.TimeoutExpired as exc:
        return StepResult(
            "subprocess",
            "FAIL",
            "Readiness step timed out.",
            tuple(command),
            None,
            (str(exc),),
        )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def classify_output(command_name: str, command: Sequence[str], completed: CommandResult) -> StepResult:
    output = completed.combined_output
    details = _tail_lines(output)
    command_tuple = tuple(command)
    if completed.returncode == 0:
        if re.search(r"\bWARN\b|\[WARN\]", output):
            return StepResult(
                command_name,
                "WARN",
                "Step completed with warnings.",
                command_tuple,
                completed.returncode,
                details,
            )
        return StepResult(
            command_name,
            "PASS",
            "Step completed.",
            command_tuple,
            completed.returncode,
            details,
        )

    if "ENVIRONMENT_BLOCKED" in output:
        return StepResult(
            command_name,
            "ENVIRONMENT_BLOCKED",
            "Step reported blocked network, registry, or environment access.",
            command_tuple,
            completed.returncode,
            details,
        )
    if "TOOLING_MISSING" in output or re.search(r"No module named ['\"]?[\w.]+['\"]?", output):
        return StepResult(
            command_name,
            "TOOLING_MISSING",
            "Step reported missing local tooling.",
            command_tuple,
            completed.returncode,
            details,
        )
    return StepResult(
        command_name,
        "FAIL",
        "Step failed.",
        command_tuple,
        completed.returncode,
        details,
    )


def run_step(step: StepSpec, *, root: Path) -> StepResult:
    command = step.command_factory(root)
    completed = run_command(command, cwd=root)
    if isinstance(completed, StepResult):
        return StepResult(
            step.name,
            completed.status,
            completed.message,
            command,
            completed.returncode,
            completed.details,
        )
    return classify_output(step.name, command, completed)


def run_publish_readiness(
    root: Path,
    *,
    intended_version: str,
    offline: bool = True,
    python_executable: str = sys.executable,
) -> list[StepResult]:
    root = root.resolve()
    return [
        run_step(step, root=root)
        for step in build_step_specs(
            intended_version=intended_version,
            offline=offline,
            python_executable=python_executable,
        )
    ]


def has_blocking_status(results: Sequence[StepResult]) -> bool:
    return any(result.status in BLOCKING_STATUSES for result in results)


def readiness_exit_code(results: Sequence[StepResult]) -> int:
    return 1 if has_blocking_status(results) else 0


def summarize(results: Sequence[StepResult]) -> dict[str, int]:
    return {status: sum(1 for result in results if result.status == status) for status in STATUSES}


def to_payload(
    results: Sequence[StepResult],
    *,
    root: Path,
    intended_version: str,
    offline: bool,
) -> dict[str, Any]:
    return {
        "approved": all(result.status == "PASS" for result in results),
        "intended_version": intended_version,
        "mode": "offline" if offline else "network",
        "results": [
            {
                "command": list(result.command),
                "details": list(result.details),
                "message": result.message,
                "name": result.name,
                "returncode": result.returncode,
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


def render_text(
    results: Sequence[StepResult],
    *,
    root: Path,
    intended_version: str,
    offline: bool,
) -> str:
    counts = summarize(results)
    if has_blocking_status(results):
        decision = "BLOCKED"
    elif any(result.status == "WARN" for result in results):
        decision = "PASS_WITH_WARNINGS"
    else:
        decision = "PASS"
    lines = [
        "Publish readiness (non-publishing)",
        f"Root: {root.resolve()}",
        f"Intended version: {intended_version}",
        f"Mode: {'offline' if offline else 'network'}",
        f"Decision: {decision}",
        "",
    ]
    for result in results:
        code = "" if result.returncode is None else f" (exit {result.returncode})"
        lines.append(f"[{result.status}] {result.name}{code}: {result.message}")
        for detail in result.details[:3]:
            lines.append(f"  - {detail}")
        if len(result.details) > 3:
            lines.append(f"  - ... {len(result.details) - 3} more line(s)")
    lines.extend(
        [
            "",
            "Summary: " + ", ".join(f"{status}={count}" for status, count in counts.items() if count),
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intended-version", required=True, help="intended release version")
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="run local-only checks")
    network.add_argument("--network", action="store_true", help="explicitly allow release-audit network checks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    offline = not args.network
    results = run_publish_readiness(
        args.root,
        intended_version=args.intended_version,
        offline=offline,
    )
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
        print(
            render_text(
                results,
                root=args.root,
                intended_version=args.intended_version,
                offline=offline,
            )
        )
    return readiness_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
