"""Public help/demo stay usable with UTF-8 mode off and ASCII/cp949 pipes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("encoding", ["ascii", "cp949"])
def test_console_help_and_demo_are_ascii_outside_checkout(tmp_path: Path, encoding: str):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "0"
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = encoding + ":strict"
    for args in (["--help"], ["gate", "--help"], ["report", "--help"], ["demo"]):
        result = subprocess.run(
            [sys.executable, "-m", "omega_lock.cli", *args],
            cwd=tmp_path, env=env, capture_output=True, check=False,
        )
        assert result.returncode == 0, (args, result.stderr.decode(encoding, errors="replace"))
        assert result.stdout.isascii(), (args, result.stdout)
        assert result.stdout


@pytest.mark.parametrize("encoding", ["ascii", "cp949"])
def test_gate_and_input_errors_escape_display_paths_only(tmp_path: Path, encoding: str):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = encoding + ":strict"
    env.pop("PYTHONPATH", None)
    train = tmp_path / "train-\U0001f4a5.json"
    holdout = tmp_path / "holdout-\U0001f4a5.json"
    report = tmp_path / "report-\U0001f4a5.html"
    train.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    holdout.write_text(json.dumps([-100, -99, -98]), encoding="utf-8")

    def invoke(*args: str):
        result = subprocess.run([sys.executable, "-m", "omega_lock.cli", *args],
                                cwd=tmp_path, env=env, capture_output=True, check=False)
        assert result.stdout.isascii(), result.stdout
        assert result.stderr.isascii(), result.stderr
        assert b"UnicodeEncodeError" not in result.stderr
        return result

    failed = invoke("gate", "--train", str(train), "--holdout", str(holdout), "--report", str(report))
    assert failed.returncode == 1
    assert b"deployment_authorized: False" in failed.stdout
    assert b"\\U0001f4a5" in failed.stdout
    assert report.is_file()
    assert "<!DOCTYPE html>" in report.read_text(encoding="utf-8")
    train.write_text("invalid JSON", encoding="utf-8")
    invalid = invoke("gate", "--train", str(train), "--holdout", str(holdout))
    assert invalid.returncode == 2 and b"invalid JSON" in invalid.stderr
    parser_error = invoke("demo", "--\U0001f4a5")
    assert parser_error.returncode == 2 and b"unrecognized arguments" in parser_error.stderr
    assert b"\\U0001f4a5" in parser_error.stderr
