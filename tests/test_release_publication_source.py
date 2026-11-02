"""Execute the actual workflow guards against committed source and altered archives."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _inline_guard(name: str) -> str:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    return textwrap.dedent(workflow.split("# BEGIN " + name + "\n", 1)[1].split("# END " + name, 1)[0])


def _execute_guard(tmp_path: Path, name: str, env: dict[str, str]):
    script = tmp_path / (name.lower() + ".py")
    script.write_text(_inline_guard(name), encoding="utf-8", newline="\n")
    return subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                          env={**os.environ, **env}, capture_output=True, check=False)


@pytest.mark.parametrize("tag,commit", [
    ("", "a" * 40), ("v0.3.11\nextra", "a" * 40), ("main", "a" * 40),
    ("v0.3.11", "a" * 39), ("v0.3.11", "a" * 41), ("v0.3.11", "z" * 40),
])
def test_dispatch_source_inputs_reject_missing_invalid_or_injected_values(tmp_path: Path, tag: str, commit: str):
    output = tmp_path / "github-output.txt"
    result = _execute_guard(tmp_path, "RELEASE_INPUT_VALIDATION", {
        "REQUESTED_RELEASE_TAG": tag, "REQUESTED_RELEASE_COMMIT": commit, "GITHUB_OUTPUT": str(output),
    })
    assert result.returncode != 0
    assert not output.exists()


def test_dispatch_source_input_accepts_only_exact_commit_and_normalizes_hex(tmp_path: Path):
    output = tmp_path / "github-output.txt"
    result = _execute_guard(tmp_path, "RELEASE_INPUT_VALIDATION", {
        "REQUESTED_RELEASE_TAG": "v0.3.11", "REQUESTED_RELEASE_COMMIT": "A" * 40, "GITHUB_OUTPUT": str(output),
    })
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "tag=v0.3.11\ncommit=" + "a" * 40 + "\n"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output([
        "git", "-c", "core.autocrlf=false", "-c", "user.name=Release Test",
        "-c", "user.email=release-test@example.invalid", *args,
    ], cwd=root, text=True, encoding="utf-8", stderr=subprocess.STDOUT).strip()


@pytest.mark.parametrize("case", [
    "valid", "wheel_bytes", "sdist_bytes", "extra_wheel_python", "missing_wheel_python",
    "worktree_changed", "commit_mismatch", "tag_mismatch", "version_mismatch",
])
def test_final_wheel_and_sdist_must_match_the_requested_git_commit(tmp_path: Path, case: str):
    package = tmp_path / "src/omega_lock"
    package.mkdir(parents=True)
    committed = {"src/omega_lock/__init__.py": b'__version__ = "0.3.11"\n',
                 "src/omega_lock/core.py": b"value = 42\n"}
    for name, data in committed.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="omega-lock"\nversion="0.3.11"\n', encoding="utf-8")
    _git(tmp_path, "init", "--quiet", "--object-format=sha1")
    _git(tmp_path, "add", "src", "pyproject.toml")
    _git(tmp_path, "commit", "--quiet", "-m", "immutable fixture source")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "tag", "v0.3.11")
    env = {"REQUESTED_RELEASE_TAG": "v0.3.11", "REQUESTED_RELEASE_COMMIT": commit,
           "PKG_VER": "0.3.11", "CONTROLLER_WORKFLOW_COMMIT": "c" * 40,
           "CONTROLLER_WORKFLOW_REF": "fixture/.github/workflows/publish.yml@refs/heads/main"}
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel_sources = {p.removeprefix("src/"): data for p, data in committed.items()}
    if case == "wheel_bytes":
        wheel_sources["omega_lock/core.py"] = b"value = 999\n"
    elif case == "extra_wheel_python":
        wheel_sources["omega_lock/extra.py"] = b"unexpected = True\n"
    elif case == "missing_wheel_python":
        del wheel_sources["omega_lock/core.py"]
    with zipfile.ZipFile(dist / "omega_lock-0.3.11-py3-none-any.whl", "w") as archive:
        for name, data in wheel_sources.items():
            archive.writestr(name, data)
        archive.writestr("omega_lock-0.3.11.dist-info/METADATA", "Metadata-Version: 2.1\nName: omega-lock\nVersion: 0.3.11\n")
    with tarfile.open(dist / "omega_lock-0.3.11.tar.gz", "w:gz") as archive:
        for name, original in committed.items():
            data = b"value = 999\n" if case == "sdist_bytes" and name.endswith("core.py") else original
            member = tarfile.TarInfo("omega_lock-0.3.11/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    if case == "worktree_changed":
        (package / "core.py").write_text("value = 999\n", encoding="utf-8")
    elif case == "commit_mismatch":
        env["REQUESTED_RELEASE_COMMIT"] = "f" * 40
    elif case == "tag_mismatch":
        (package / "core.py").write_text("value = 999\n", encoding="utf-8")
        _git(tmp_path, "add", "src")
        _git(tmp_path, "commit", "--quiet", "-m", "different source")
        env["REQUESTED_RELEASE_COMMIT"] = _git(tmp_path, "rev-parse", "HEAD")
    elif case == "version_mismatch":
        env["PKG_VER"] = "0.3.13"
    result = _execute_guard(tmp_path, "RELEASE_ARTIFACT_VERIFICATION", env)
    if case == "valid":
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["source_commit"] == commit
        assert receipt["controller_workflow_commit"] == "c" * 40
        assert receipt["verified_python_files"] == 2
        assert len(receipt["artifact_sha256"]) == 2
    else:
        assert result.returncode != 0, (case, result.stdout, result.stderr)
