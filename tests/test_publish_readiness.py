# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, Sequence


def _load_publish_readiness() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "publish_readiness.py"
    spec = importlib.util.spec_from_file_location("publish_readiness", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_publish_readiness()


class StepSpecLike(Protocol):
    name: str

    def command_factory(self, root: Path) -> tuple[str, ...]: ...


def _commands(specs: Sequence[StepSpecLike], root: Path) -> list[tuple[str, ...]]:
    return [tuple(spec.command_factory(root)) for spec in specs]


def test_step_plan_contains_all_required_non_publishing_checks(tmp_path: Path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "omega_lock-1.2.3-py3-none-any.whl").write_text("", encoding="utf-8")

    specs = GATE.build_step_specs(
        intended_version="1.2.3",
        offline=True,
        python_executable="python",
    )
    names = [spec.name for spec in specs]
    commands = _commands(specs, tmp_path)
    flattened = "\n".join(" ".join(command) for command in commands)

    assert names == [
        "repo-consistency",
        "readme-claims",
        "golden-audits",
        "demo-replay-check",
        "demo-sram-check",
        "pytest",
        "pyright",
        "ruff",
        "build",
        "twine-check",
        "wheel-smoke-install",
        "release-audit",
    ]
    assert "twine upload" not in flattened
    assert "git tag" not in flattened
    assert "gh release" not in flattened
    assert "pypi.org" not in flattened


def test_offline_plan_uses_offline_release_audit_and_local_build(tmp_path: Path):
    specs = GATE.build_step_specs(
        intended_version="1.2.3",
        offline=True,
        python_executable="python",
    )
    commands = {spec.name: spec.command_factory(tmp_path) for spec in specs}

    assert "--offline" in commands["release-audit"]
    assert "--network" not in commands["release-audit"]
    assert "--no-isolation" in commands["build"]
    assert "--intended-version" in commands["wheel-smoke-install"]
    assert "1.2.3" in commands["wheel-smoke-install"]


def test_network_plan_requires_explicit_network_flag_for_release_audit(tmp_path: Path):
    specs = GATE.build_step_specs(
        intended_version="1.2.3",
        offline=False,
        python_executable="python",
    )
    commands = {spec.name: spec.command_factory(tmp_path) for spec in specs}

    assert "--network" in commands["release-audit"]
    assert "--offline" not in commands["release-audit"]


def test_twine_command_includes_local_dist_artifacts(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "omega_lock-1.2.3-py3-none-any.whl"
    sdist = dist / "omega_lock-1.2.3.tar.gz"
    draft = dist / "release_draft_v1.2.3.md"
    zip_file = dist / "not-a-distribution.zip"
    wheel.write_text("", encoding="utf-8")
    sdist.write_text("", encoding="utf-8")
    draft.write_text("", encoding="utf-8")
    zip_file.write_text("", encoding="utf-8")

    spec = next(
        spec
        for spec in GATE.build_step_specs(
            intended_version="1.2.3",
            offline=True,
            python_executable="python",
        )
        if spec.name == "twine-check"
    )
    command = spec.command_factory(tmp_path)

    assert command[:4] == ("python", "-m", "twine", "check")
    assert str(wheel) in command
    assert str(sdist) in command
    assert str(draft) not in command
    assert str(zip_file) not in command


def test_classify_missing_modules_as_tooling_missing():
    for module in ("pytest", "pyright", "ruff", "build", "twine"):
        completed = GATE.CommandResult(
            returncode=1,
            stdout="",
            stderr=f"C:\\Python\\python.exe: No module named {module}",
        )

        result = GATE.classify_output(module, ("python", "-m", module), completed)

        assert result.status == "TOOLING_MISSING"


def test_classify_environment_blocked_is_release_blocker():
    completed = GATE.CommandResult(
        returncode=1,
        stdout="[ENVIRONMENT_BLOCKED] PyPI status could not be checked",
        stderr="",
    )

    result = GATE.classify_output("release-audit", ("python", "scripts/release_audit.py"), completed)

    assert result.status == "ENVIRONMENT_BLOCKED"
    assert GATE.has_blocking_status([result])
    assert GATE.readiness_exit_code([result]) == 1


def test_success_with_warn_is_not_tooling_or_environment_approval_blocker():
    completed = GATE.CommandResult(
        returncode=0,
        stdout="[WARN] changelog: absent\n[PASS] rest: ok",
        stderr="",
    )

    result = GATE.classify_output("release-audit", ("python", "scripts/release_audit.py"), completed)

    assert result.status == "WARN"
    assert not GATE.has_blocking_status([result])


def test_exit_code_zero_only_without_blocking_statuses():
    pass_result = GATE.StepResult("pytest", "PASS", "ok")
    warn_result = GATE.StepResult("release-audit", "WARN", "offline network status not checked")
    fail_result = GATE.StepResult("ruff", "FAIL", "lint failed")
    tooling_result = GATE.StepResult("pyright", "TOOLING_MISSING", "missing")
    env_result = GATE.StepResult("pypi", "ENVIRONMENT_BLOCKED", "blocked")

    assert GATE.readiness_exit_code([pass_result]) == 0
    assert GATE.readiness_exit_code([pass_result, warn_result]) == 0
    assert GATE.readiness_exit_code([fail_result]) == 1
    assert GATE.readiness_exit_code([tooling_result]) == 1
    assert GATE.readiness_exit_code([env_result]) == 1


def test_json_payload_is_stable_and_marks_unapproved_on_tooling_missing(tmp_path: Path):
    results = [
        GATE.StepResult("repo-consistency", "PASS", "ok", ("python", "check"), 0),
        GATE.StepResult("pyright", "TOOLING_MISSING", "missing", ("python", "-m", "pyright"), 1),
    ]

    payload = GATE.to_payload(
        results,
        root=tmp_path,
        intended_version="1.2.3",
        offline=True,
    )
    rendered_once = GATE.render_json(payload)
    rendered_twice = GATE.render_json(payload)
    parsed = json.loads(rendered_once)

    assert rendered_once == rendered_twice
    assert parsed["approved"] is False
    assert parsed["summary"]["TOOLING_MISSING"] == 1
    assert parsed["mode"] == "offline"


def test_json_payload_does_not_mark_warnings_as_approval(tmp_path: Path):
    results = [
        GATE.StepResult("repo-consistency", "PASS", "ok"),
        GATE.StepResult("release-audit", "WARN", "offline network status not checked"),
    ]

    payload = GATE.to_payload(
        results,
        root=tmp_path,
        intended_version="1.2.3",
        offline=True,
    )

    assert payload["approved"] is False
    assert GATE.readiness_exit_code(results) == 0
