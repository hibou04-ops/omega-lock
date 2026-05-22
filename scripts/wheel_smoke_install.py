#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Build and smoke-install the local omega-lock wheel without PyPI access."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only.
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = "omega-lock"
IMPORT_PACKAGE = "omega_lock"
STATUSES = ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectInfo:
    name: str | None
    version: str | None
    scripts: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def load_pyproject(root: Path) -> tuple[ProjectInfo | None, CheckResult | None]:
    if tomllib is None:
        return None, CheckResult(
            "tomllib",
            "TOOLING_MISSING",
            "Python 3.11+ tomllib is required to read pyproject.toml.",
        )
    path = root / "pyproject.toml"
    if not path.exists():
        return None, CheckResult("pyproject", "FAIL", "pyproject.toml is missing.")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return None, CheckResult("pyproject", "FAIL", f"pyproject.toml is invalid TOML: {exc}")

    project = data.get("project", {})
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict):
        scripts = {}
    return (
        ProjectInfo(
            name=project.get("name") if isinstance(project.get("name"), str) else None,
            version=project.get("version") if isinstance(project.get("version"), str) else None,
            scripts=tuple(sorted(str(name) for name in scripts)),
        ),
        None,
    )


def validate_project_info(project: ProjectInfo, intended_version: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    if project.name == PROJECT_NAME:
        results.append(CheckResult("pyproject-name", "PASS", f"project.name is {PROJECT_NAME}."))
    else:
        results.append(
            CheckResult(
                "pyproject-name",
                "FAIL",
                "project.name does not match the PyPI distribution name.",
                (f"expected: {PROJECT_NAME}", f"found: {project.name!r}"),
            )
        )

    if project.version == intended_version:
        results.append(
            CheckResult("pyproject-version", "PASS", f"project.version is {intended_version}.")
        )
    else:
        results.append(
            CheckResult(
                "pyproject-version",
                "FAIL",
                "project.version does not match --intended-version.",
                (f"expected: {intended_version}", f"found: {project.version!r}"),
            )
        )
    return results


def build_command(root: Path, dist_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
        str(root),
    ]


def install_command(venv_python: Path, wheel_path: Path) -> list[str]:
    return [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        str(wheel_path),
    ]


def create_venv_command(venv_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(venv_dir),
    ]


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_command(args: Sequence[str], *, cwd: Path, timeout: int = 120) -> CommandResult | CheckResult:
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        return CheckResult(
            "subprocess",
            "TOOLING_MISSING",
            "Required executable is unavailable.",
            (str(exc),),
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            "subprocess",
            "FAIL",
            "Subprocess timed out.",
            (" ".join(args), str(exc)),
        )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def nonzero_result(
    name: str,
    completed: CommandResult,
    *,
    fail_message: str,
    tooling_markers: Sequence[str] = (),
) -> CheckResult:
    output = completed.combined_output
    status: Status = "FAIL"
    message = fail_message
    if any(marker in output for marker in tooling_markers):
        status = "TOOLING_MISSING"
        message = f"{name} tooling is unavailable."
    details = tuple(line for line in output.splitlines() if line.strip())[-20:]
    return CheckResult(name, status, message, details)


def build_wheel(root: Path, dist_dir: Path) -> tuple[CheckResult, Path | None]:
    dist_dir.mkdir(parents=True, exist_ok=True)
    completed = run_command(build_command(root, dist_dir), cwd=root, timeout=180)
    if isinstance(completed, CheckResult):
        return completed, None
    if completed.returncode != 0:
        return (
            nonzero_result(
                "build-wheel",
                completed,
                fail_message="Local wheel build failed.",
                tooling_markers=("No module named build", "No module named 'build'"),
            ),
            None,
        )
    wheel_result, wheel_path = find_wheel(dist_dir, "")
    if wheel_path is None:
        return (
            CheckResult(
                "build-wheel",
                "FAIL",
                "Local wheel build completed but no omega-lock wheel was found.",
                (str(dist_dir), wheel_result.message),
            ),
            None,
        )
    return CheckResult("build-wheel", "PASS", "Local wheel build completed.", (wheel_path.name,)), wheel_path


def find_wheel(dist_dir: Path, intended_version: str) -> tuple[CheckResult, Path | None]:
    if not dist_dir.exists():
        return CheckResult("wheel-artifact", "FAIL", "dist directory does not exist.", (str(dist_dir),)), None

    pattern = re.compile(r"^omega_lock-(\d+\.\d+\.\d+)-py3-none-any\.whl$")
    wheels = sorted(path for path in dist_dir.glob("omega_lock-*-py3-none-any.whl") if path.is_file())
    if not wheels:
        return CheckResult("wheel-artifact", "FAIL", "No omega-lock wheel found in dist directory."), None

    matching: list[Path] = []
    stale: list[str] = []
    for path in wheels:
        match = pattern.fullmatch(path.name)
        if match is None:
            stale.append(path.name)
            continue
        version = match.group(1)
        if intended_version and version == intended_version:
            matching.append(path)
        elif intended_version:
            stale.append(path.name)
        else:
            matching.append(path)

    if not matching:
        return (
            CheckResult(
                "wheel-artifact",
                "FAIL",
                "No wheel artifact matches the intended version.",
                tuple(stale),
            ),
            None,
        )
    if len(matching) > 1:
        newest = max(matching, key=lambda path: path.stat().st_mtime)
        return (
            CheckResult(
                "wheel-artifact",
                "WARN",
                "Multiple matching wheel artifacts found; using newest by mtime.",
                tuple(path.name for path in matching),
            ),
            newest,
        )
    return (
        CheckResult(
            "wheel-artifact",
            "PASS",
            "Wheel artifact matches intended version." if intended_version else "Wheel artifact found.",
            (matching[0].name,),
        ),
        matching[0],
    )


def create_virtualenv(venv_dir: Path, root: Path) -> CheckResult:
    completed = run_command(create_venv_command(venv_dir), cwd=root, timeout=120)
    if isinstance(completed, CheckResult):
        return completed
    if completed.returncode != 0:
        return nonzero_result(
            "create-venv",
            completed,
            fail_message="Temporary virtual environment creation failed.",
            tooling_markers=("No module named venv", "No module named 'venv'", "ensurepip"),
        )
    python_path = venv_python_path(venv_dir)
    if not python_path.exists():
        return CheckResult(
            "create-venv",
            "TOOLING_MISSING",
            "Virtual environment was created without a Python executable.",
            (str(python_path),),
        )
    return CheckResult("create-venv", "PASS", "Temporary virtual environment created.")


def install_local_wheel(venv_python: Path, wheel_path: Path, root: Path) -> CheckResult:
    completed = run_command(install_command(venv_python, wheel_path), cwd=root, timeout=120)
    if isinstance(completed, CheckResult):
        return completed
    if completed.returncode != 0:
        return nonzero_result(
            "install-wheel",
            completed,
            fail_message="Local wheel install failed.",
            tooling_markers=("No module named pip", "No module named 'pip'"),
        )
    return CheckResult(
        "install-wheel",
        "PASS",
        "Installed wheel from local artifact with --no-index and --no-deps.",
        (wheel_path.name,),
    )


def metadata_probe_code() -> str:
    return r'''
import json
from importlib import metadata

payload = {}
try:
    dist = metadata.distribution("omega-lock")
    payload["metadata_name"] = dist.metadata.get("Name")
    payload["metadata_version"] = dist.version
    payload["console_scripts"] = sorted(
        ep.name for ep in dist.entry_points if ep.group == "console_scripts"
    )
except Exception as exc:
    payload["metadata_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(payload, sort_keys=True))
'''


def runtime_probe_code() -> str:
    return r'''
import json

payload = {}
try:
    import omega_lock
    payload["import_package"] = "omega_lock"
    payload["import_version"] = getattr(omega_lock, "__version__", None)
except Exception as exc:
    payload["import_error"] = f"{type(exc).__name__}: {exc}"

try:
    from omega_lock import EvalResult, ParamSpec, pearson

    spec = ParamSpec("x", "float", 0.5, 0.0, 1.0)
    result = EvalResult(fitness=1.25, n_trials=3, metadata={"smoke": True})
    corr = pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    payload["minimal_api_ok"] = (
        spec.name == "x"
        and result.fitness == 1.25
        and result.n_trials == 3
        and abs(corr - 1.0) < 1e-12
    )
except Exception as exc:
    payload["minimal_api_ok"] = False
    payload["minimal_api_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(payload, sort_keys=True))
'''


def read_json_probe(
    *,
    name: str,
    venv_python: Path,
    root: Path,
    code: str,
    fail_message: str,
) -> tuple[CheckResult, dict[str, Any] | None]:
    completed = run_command([str(venv_python), "-c", code], cwd=root, timeout=60)
    if isinstance(completed, CheckResult):
        return completed, None
    if completed.returncode != 0:
        return (
            nonzero_result(
                name,
                completed,
                fail_message=fail_message,
            ),
            None,
        )
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        return (
            CheckResult(
                name,
                "FAIL",
                "Wheel probe did not emit valid JSON.",
                (str(exc), completed.stdout.strip()[:500]),
            ),
            None,
        )
    return CheckResult(name, "PASS", "Wheel probe completed."), payload


def read_metadata_payload(venv_python: Path, root: Path) -> tuple[CheckResult, dict[str, Any] | None]:
    return read_json_probe(
        name="metadata-smoke",
        venv_python=venv_python,
        root=root,
        code=metadata_probe_code(),
        fail_message="Wheel metadata probe failed.",
    )


def read_runtime_payload(venv_python: Path, root: Path) -> tuple[CheckResult, dict[str, Any] | None]:
    return read_json_probe(
        name="runtime-smoke",
        venv_python=venv_python,
        root=root,
        code=runtime_probe_code(),
        fail_message="Wheel runtime import/API probe failed.",
    )


def validate_metadata_payload(
    payload: dict[str, Any],
    *,
    intended_version: str,
    expected_scripts: Sequence[str],
) -> list[CheckResult]:
    results: list[CheckResult] = []

    metadata_name = payload.get("metadata_name")
    if metadata_name == PROJECT_NAME:
        results.append(CheckResult("wheel-metadata-name", "PASS", f"Wheel metadata Name is {PROJECT_NAME}."))
    else:
        results.append(
            CheckResult(
                "wheel-metadata-name",
                "FAIL",
                "Wheel metadata Name mismatch.",
                (f"expected: {PROJECT_NAME}", f"found: {metadata_name!r}", str(payload.get("metadata_error", ""))),
            )
        )

    metadata_version = payload.get("metadata_version")
    if metadata_version == intended_version:
        results.append(
            CheckResult("wheel-metadata-version", "PASS", f"Wheel metadata Version is {intended_version}.")
        )
    else:
        results.append(
            CheckResult(
                "wheel-metadata-version",
                "FAIL",
                "Wheel metadata Version mismatch.",
                (f"expected: {intended_version}", f"found: {metadata_version!r}"),
            )
        )

    actual_scripts = tuple(sorted(str(name) for name in payload.get("console_scripts", [])))
    expected = tuple(sorted(expected_scripts))
    if actual_scripts == expected:
        if expected:
            message = f"Wheel console scripts match pyproject: {', '.join(expected)}."
        else:
            message = "Wheel exposes no console scripts, matching pyproject."
        results.append(CheckResult("console-scripts", "PASS", message))
    else:
        results.append(
            CheckResult(
                "console-scripts",
                "FAIL",
                "Wheel console scripts do not match pyproject [project.scripts].",
                (f"expected: {list(expected)}", f"found: {list(actual_scripts)}"),
            )
        )
    return results


def _runtime_dependency_missing(error: str) -> bool:
    return "ModuleNotFoundError" in error and f"No module named '{IMPORT_PACKAGE}'" not in error


def validate_runtime_payload(payload: dict[str, Any], *, intended_version: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    import_error = str(payload.get("import_error", ""))
    if import_error:
        if _runtime_dependency_missing(import_error):
            return [
                CheckResult(
                    "runtime-import",
                    "WARN",
                    "Installed package import was skipped because runtime dependencies are unavailable in the offline no-deps environment.",
                    (import_error,),
                ),
                CheckResult(
                    "minimal-api",
                    "WARN",
                    "Minimal API smoke was skipped because runtime import dependencies are unavailable.",
                ),
            ]
        return [
            CheckResult(
                "runtime-import",
                "FAIL",
                "Installed wheel package import failed.",
                (import_error,),
            )
        ]

    if payload.get("import_package") == IMPORT_PACKAGE and payload.get("import_version") == intended_version:
        results.append(
            CheckResult("runtime-import", "PASS", f"{IMPORT_PACKAGE} imports with version {intended_version}.")
        )
    else:
        results.append(
            CheckResult(
                "runtime-import",
                "FAIL",
                "Installed wheel import package/version mismatch.",
                (
                    f"expected package: {IMPORT_PACKAGE}",
                    f"expected version: {intended_version}",
                    f"found package: {payload.get('import_package')!r}",
                    f"found version: {payload.get('import_version')!r}",
                ),
            )
        )

    if payload.get("minimal_api_ok") is True:
        results.append(CheckResult("minimal-api", "PASS", "Minimal installed API smoke test passed."))
    else:
        results.append(
            CheckResult(
                "minimal-api",
                "FAIL",
                "Minimal installed API smoke test failed.",
                (str(payload.get("minimal_api_error", "")),),
            )
        )
    return results


def validate_probe_payload(
    payload: dict[str, Any],
    *,
    intended_version: str,
    expected_scripts: Sequence[str],
) -> list[CheckResult]:
    return validate_metadata_payload(
        payload,
        intended_version=intended_version,
        expected_scripts=expected_scripts,
    ) + validate_runtime_payload(payload, intended_version=intended_version)


def run_demo_smoke(venv_python: Path, root: Path) -> CheckResult:
    demo = root / "examples" / "demo_replay.py"
    if not demo.exists():
        return CheckResult("demo-smoke", "WARN", "Cheap deterministic demo smoke is unavailable.")

    completed = run_command([str(venv_python), str(demo), "--check"], cwd=root, timeout=30)
    if isinstance(completed, CheckResult):
        return completed
    if completed.returncode != 0:
        return nonzero_result(
            "demo-smoke",
            completed,
            fail_message="Deterministic demo smoke failed.",
        )
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        return CheckResult(
            "demo-smoke",
            "FAIL",
            "Deterministic demo smoke emitted invalid JSON.",
            (str(exc), completed.stdout.strip()[:500]),
        )
    if payload.get("status") == "PASS":
        return CheckResult("demo-smoke", "PASS", "Deterministic demo replay smoke passed.")
    return CheckResult(
        "demo-smoke",
        "FAIL",
        "Deterministic demo replay smoke reported failure.",
        (json.dumps(payload, sort_keys=True),),
    )


def has_blocking_status(results: Sequence[CheckResult]) -> bool:
    return any(result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"} for result in results)


def summarize(results: Sequence[CheckResult]) -> dict[str, int]:
    return {status: sum(1 for result in results if result.status == status) for status in STATUSES}


def render_results(results: Sequence[CheckResult], *, root: Path, dist_dir: Path, intended_version: str) -> str:
    lines = [
        "Wheel smoke install",
        f"Root: {root.resolve()}",
        f"Dist dir: {dist_dir.resolve()}",
        f"Intended version: {intended_version}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")
        for detail in result.details:
            if detail:
                lines.append(f"  - {detail}")
    counts = summarize(results)
    lines.extend(
        [
            "",
            "Summary: " + ", ".join(f"{status}={count}" for status, count in counts.items() if count),
        ]
    )
    return "\n".join(lines)


def run_wheel_smoke(root: Path, dist_dir: Path, intended_version: str) -> list[CheckResult]:
    root = root.resolve()
    dist_dir = dist_dir if dist_dir.is_absolute() else root / dist_dir
    results: list[CheckResult] = []

    project, load_error = load_pyproject(root)
    if load_error is not None:
        return [load_error]
    assert project is not None
    results.extend(validate_project_info(project, intended_version))
    if has_blocking_status(results):
        return results

    build_result, _built_wheel = build_wheel(root, dist_dir)
    results.append(build_result)
    if build_result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        return results

    wheel_result, wheel_path = find_wheel(dist_dir, intended_version)
    results.append(wheel_result)
    if wheel_path is None or wheel_result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        return results

    with tempfile.TemporaryDirectory(prefix="omega-lock-wheel-smoke-") as temp:
        venv_dir = Path(temp) / "venv"
        venv_result = create_virtualenv(venv_dir, root)
        results.append(venv_result)
        if venv_result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
            return results

        venv_python = venv_python_path(venv_dir)
        install_result = install_local_wheel(venv_python, wheel_path, root)
        results.append(install_result)
        if install_result.status in {"FAIL", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
            return results

        metadata_result, metadata_payload = read_metadata_payload(venv_python, root)
        results.append(metadata_result)
        if metadata_payload is None:
            return results
        results.extend(
            validate_metadata_payload(
                metadata_payload,
                intended_version=intended_version,
                expected_scripts=project.scripts,
            )
        )
        if has_blocking_status(results):
            return results

        runtime_result, runtime_payload = read_runtime_payload(venv_python, root)
        results.append(runtime_result)
        if runtime_payload is None:
            return results
        runtime_results = validate_runtime_payload(
            runtime_payload,
            intended_version=intended_version,
        )
        results.extend(runtime_results)
        if has_blocking_status(results):
            return results

        if any(result.status == "WARN" for result in runtime_results):
            results.append(
                CheckResult(
                    "demo-smoke",
                    "WARN",
                    "Deterministic demo smoke skipped because runtime import dependencies are unavailable.",
                )
            )
        else:
            results.append(run_demo_smoke(venv_python, root))
    return results


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"), help="local dist directory")
    parser.add_argument("--intended-version", required=True, help="expected package version")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = run_wheel_smoke(args.root, args.dist_dir, args.intended_version)
    print(render_results(results, root=args.root, dist_dir=args.dist_dir, intended_version=args.intended_version))
    return 1 if has_blocking_status(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
