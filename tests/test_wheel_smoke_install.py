# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_wheel_smoke() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "wheel_smoke_install.py"
    spec = importlib.util.spec_from_file_location("wheel_smoke_install", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SMOKE = _load_wheel_smoke()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_build_and_install_commands_are_local_only(tmp_path: Path):
    build_cmd = SMOKE.build_command(tmp_path, tmp_path / "dist")
    install_cmd = SMOKE.install_command(tmp_path / "venv" / "Scripts" / "python.exe", tmp_path / "dist" / "omega_lock-1.2.3-py3-none-any.whl")

    assert "--no-isolation" in build_cmd
    assert "--no-index" in install_cmd
    assert "--no-deps" in install_cmd
    assert not any("pypi.org" in part or part.startswith("http") for part in build_cmd + install_cmd)


def test_load_pyproject_reads_expected_console_scripts(tmp_path: Path):
    _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "omega-lock"
        version = "1.2.3"

        [project.scripts]
        omega-lock = "omega_lock.cli:main"
        """,
    )

    project, error = SMOKE.load_pyproject(tmp_path)

    assert error is None
    assert project is not None
    assert project.name == "omega-lock"
    assert project.version == "1.2.3"
    assert project.scripts == ("omega-lock",)


def test_find_wheel_requires_intended_version(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    stale = dist / "omega_lock-1.2.2-py3-none-any.whl"
    current = dist / "omega_lock-1.2.3-py3-none-any.whl"
    stale.write_text("", encoding="utf-8")
    current.write_text("", encoding="utf-8")

    result, wheel = SMOKE.find_wheel(dist, "1.2.3")

    assert result.status == "PASS"
    assert wheel == current


def test_find_wheel_fails_when_only_stale_version_exists(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "omega_lock-1.2.2-py3-none-any.whl").write_text("", encoding="utf-8")

    result, wheel = SMOKE.find_wheel(dist, "1.2.3")

    assert result.status == "FAIL"
    assert wheel is None
    assert "omega_lock-1.2.2-py3-none-any.whl" in result.details


def test_validate_probe_payload_accepts_matching_metadata_import_and_no_cli():
    payload = {
        "metadata_name": "omega-lock",
        "metadata_version": "1.2.3",
        "console_scripts": [],
        "import_package": "omega_lock",
        "import_version": "1.2.3",
        "minimal_api_ok": True,
    }

    results = SMOKE.validate_probe_payload(
        payload,
        intended_version="1.2.3",
        expected_scripts=(),
    )

    assert not SMOKE.has_blocking_status(results)
    assert {result.name for result in results} == {
        "wheel-metadata-name",
        "wheel-metadata-version",
        "runtime-import",
        "minimal-api",
        "console-scripts",
    }


def test_validate_probe_payload_catches_metadata_and_import_mismatches():
    payload = {
        "metadata_name": "omega_lock",
        "metadata_version": "1.2.2",
        "console_scripts": [],
        "import_package": "omega_lock",
        "import_version": "1.2.2",
        "minimal_api_ok": False,
        "minimal_api_error": "boom",
    }

    results = SMOKE.validate_probe_payload(
        payload,
        intended_version="1.2.3",
        expected_scripts=(),
    )
    by_name = {result.name: result for result in results}

    assert by_name["wheel-metadata-name"].status == "FAIL"
    assert by_name["wheel-metadata-version"].status == "FAIL"
    assert by_name["runtime-import"].status == "FAIL"
    assert by_name["minimal-api"].status == "FAIL"
    assert SMOKE.has_blocking_status(results)


def test_validate_probe_payload_catches_unexpected_console_script():
    payload = {
        "metadata_name": "omega-lock",
        "metadata_version": "1.2.3",
        "console_scripts": ["omega-lock"],
        "import_package": "omega_lock",
        "import_version": "1.2.3",
        "minimal_api_ok": True,
    }

    results = SMOKE.validate_probe_payload(
        payload,
        intended_version="1.2.3",
        expected_scripts=(),
    )
    by_name = {result.name: result for result in results}

    assert by_name["console-scripts"].status == "FAIL"
    assert "omega-lock" in by_name["console-scripts"].details[1]


def test_validate_probe_payload_accepts_expected_console_script():
    payload = {
        "metadata_name": "omega-lock",
        "metadata_version": "1.2.3",
        "console_scripts": ["omega-lock"],
        "import_package": "omega_lock",
        "import_version": "1.2.3",
        "minimal_api_ok": True,
    }

    results = SMOKE.validate_probe_payload(
        payload,
        intended_version="1.2.3",
        expected_scripts=("omega-lock",),
    )
    by_name = {result.name: result for result in results}

    assert by_name["console-scripts"].status == "PASS"
    assert not SMOKE.has_blocking_status(results)


def test_runtime_missing_dependency_is_warn_not_tooling_missing():
    payload = {
        "import_error": "ModuleNotFoundError: No module named 'numpy'",
    }

    results = SMOKE.validate_runtime_payload(payload, intended_version="1.2.3")
    by_name = {result.name: result for result in results}

    assert by_name["runtime-import"].status == "WARN"
    assert by_name["minimal-api"].status == "WARN"
    assert not SMOKE.has_blocking_status(results)


def test_metadata_only_probe_preserves_console_script_check_without_import():
    payload = {
        "metadata_name": "omega-lock",
        "metadata_version": "1.2.3",
        "console_scripts": [],
        "import_error": "ModuleNotFoundError: No module named 'numpy'",
    }

    results = SMOKE.validate_metadata_payload(
        payload,
        intended_version="1.2.3",
        expected_scripts=(),
    )
    by_name = {result.name: result for result in results}

    assert by_name["wheel-metadata-name"].status == "PASS"
    assert by_name["wheel-metadata-version"].status == "PASS"
    assert by_name["console-scripts"].status == "PASS"
    assert not SMOKE.has_blocking_status(results)


def test_nonzero_build_missing_module_reports_tooling_missing():
    completed = SMOKE.CommandResult(
        returncode=1,
        stdout="",
        stderr="C:\\Python\\python.exe: No module named build",
    )

    result = SMOKE.nonzero_result(
        "build-wheel",
        completed,
        fail_message="Local wheel build failed.",
        tooling_markers=("No module named build", "No module named 'build'"),
    )

    assert result.status == "TOOLING_MISSING"


def test_nonzero_venv_missing_module_reports_tooling_missing():
    completed = SMOKE.CommandResult(
        returncode=1,
        stdout="",
        stderr="C:\\Python\\python.exe: No module named venv",
    )

    result = SMOKE.nonzero_result(
        "create-venv",
        completed,
        fail_message="Temporary virtual environment creation failed.",
        tooling_markers=("No module named venv", "No module named 'venv'", "ensurepip"),
    )

    assert result.status == "TOOLING_MISSING"
