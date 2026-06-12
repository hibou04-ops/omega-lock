# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for the omega-lock console entry point (omega_lock.cli)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omega_lock import EvalResult, KCThresholds, P1Config, ParamSpec, __version__, run_p1
from omega_lock.cli import main


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# ── demo ───────────────────────────────────────────────────────────────────


def test_demo_prints_full_narrative_and_succeeds(capsys: pytest.CaptureFixture):
    rc = main(["demo"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "pipeline status      = FAIL:KC-4" in out
    assert "KC-4 verdict         = PASS" in out
    assert "Walk-forward gate demo PASSED." in out


def test_demo_matches_example_script_narrative(capsys: pytest.CaptureFixture):
    """`omega-lock demo` and the example file must print the same story."""
    import io
    from contextlib import redirect_stdout

    import walkforward_gate_demo

    rc = main(["demo"])
    cli_out = capsys.readouterr().out

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        example_rc = walkforward_gate_demo.main()

    assert rc == example_rc == 0
    assert cli_out == buffer.getvalue()


# ── gate ───────────────────────────────────────────────────────────────────


def test_gate_pass_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture):
    train = _write_json(tmp_path / "train.json", [1.0, 2.0, 3.0, 4.0, 5.0])
    holdout = _write_json(tmp_path / "holdout.json", [1.1, 2.2, 2.9, 4.2, 5.1])

    rc = main(["gate", "--train", train, "--holdout", holdout])

    out = capsys.readouterr().out
    assert rc == 0
    assert "gate     : PASS" in out
    assert "pearson  : 0.99" in out


def test_gate_fail_exit_one_with_reason(tmp_path: Path, capsys: pytest.CaptureFixture):
    train = _write_json(tmp_path / "train.json", [1, 2, 3, 4, 5])
    holdout = _write_json(tmp_path / "holdout.json", [5, 4, 3, 2, 1])

    rc = main(["gate", "--train", train, "--holdout", holdout])

    out = capsys.readouterr().out
    assert rc == 1
    assert "gate     : FAIL" in out
    assert "reason   :" in out


def test_gate_pearson_min_override_flips_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    # Pearson([1,2,3], [3,1,2]) = -0.5: fails the 0.3 default, passes -0.9.
    train = _write_json(tmp_path / "train.json", [1.0, 2.0, 3.0])
    holdout = _write_json(tmp_path / "holdout.json", [3.0, 1.0, 2.0])

    rc_default = main(["gate", "--train", train, "--holdout", holdout])
    capsys.readouterr()
    rc_loose = main(
        ["gate", "--train", train, "--holdout", holdout, "--pearson-min", "-0.9"]
    )

    assert rc_default == 1
    assert rc_loose == 0


def test_gate_writes_html_report(tmp_path: Path, capsys: pytest.CaptureFixture):
    train = _write_json(tmp_path / "train.json", [1, 2, 3, 4])
    holdout = _write_json(tmp_path / "holdout.json", [4, 3, 2, 1])
    report = tmp_path / "gate.html"

    rc = main(["gate", "--train", train, "--holdout", holdout, "--report", str(report)])

    out = capsys.readouterr().out
    assert rc == 1  # verdict still drives the exit code
    assert f"wrote {report}" in out
    html = report.read_text(encoding="utf-8")
    assert "omega-lock score gate" in html
    assert "<svg" in html


def test_gate_rejects_non_numeric_arrays(tmp_path: Path, capsys: pytest.CaptureFixture):
    train = _write_json(tmp_path / "train.json", ["a", "b"])
    holdout = _write_json(tmp_path / "holdout.json", [1, 2])

    rc = main(["gate", "--train", train, "--holdout", holdout])

    captured = capsys.readouterr()
    assert rc == 2
    assert "JSON array of numbers" in captured.err


def test_gate_rejects_non_array_payload(tmp_path: Path, capsys: pytest.CaptureFixture):
    train = _write_json(tmp_path / "train.json", {"scores": [1, 2]})
    holdout = _write_json(tmp_path / "holdout.json", [1, 2])

    rc = main(["gate", "--train", train, "--holdout", holdout])

    assert rc == 2
    assert "JSON array of numbers" in capsys.readouterr().err


def test_gate_missing_file_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture):
    holdout = _write_json(tmp_path / "holdout.json", [1, 2])

    rc = main(["gate", "--train", str(tmp_path / "absent.json"), "--holdout", holdout])

    assert rc == 2
    assert "file not found" in capsys.readouterr().err


def test_gate_invalid_json_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    holdout = _write_json(tmp_path / "holdout.json", [1, 2])

    rc = main(["gate", "--train", str(bad), "--holdout", holdout])

    assert rc == 2
    assert "invalid JSON" in capsys.readouterr().err


# ── report ─────────────────────────────────────────────────────────────────


class _TinyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", neutral=0.0, low=0.0, high=1.0),
            ParamSpec(name="y", dtype="float", neutral=0.0, low=0.0, high=1.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(
            fitness=10.0 * float(params["x"]) + float(params["y"]),
            sample_count=100,
        )


def test_report_renders_saved_p1_result_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    artifact = tmp_path / "p1_result.json"
    run_p1(
        train_target=_TinyTarget(),
        test_target=_TinyTarget(),
        config=P1Config(
            unlock_k=2,
            grid_points_per_axis=3,
            walk_forward_top_n=4,
            kc_thresholds=KCThresholds(
                gini_min=0.0, top_bot_ratio_min=1.0, trade_count_min=1,
                pearson_min=0.9, trade_ratio_min=0.0,
            ),
            stress_verbose=False,
            grid_verbose=False,
        ),
        output_path=artifact,
    )
    out_html = tmp_path / "p1.html"

    rc = main(["report", "--input", str(artifact), "-o", str(out_html)])

    assert rc == 0
    assert f"wrote {out_html}" in capsys.readouterr().out
    html = out_html.read_text(encoding="utf-8")
    assert "omega-lock P1 run" in html
    assert "<svg" in html


def test_report_rejects_unknown_schema(tmp_path: Path, capsys: pytest.CaptureFixture):
    artifact = _write_json(tmp_path / "weird.json", {"schema_version": "nope.v9"})

    rc = main(["report", "--input", artifact, "-o", str(tmp_path / "out.html")])

    assert rc == 2
    assert "unrecognized mapping" in capsys.readouterr().err


def test_report_rejects_non_object_payload(tmp_path: Path, capsys: pytest.CaptureFixture):
    artifact = _write_json(tmp_path / "list.json", [1, 2, 3])

    rc = main(["report", "--input", artifact, "-o", str(tmp_path / "out.html")])

    assert rc == 2
    assert "JSON object" in capsys.readouterr().err


# ── version / smoke ────────────────────────────────────────────────────────


def test_version_flag_via_module_subprocess():
    completed = subprocess.run(
        [sys.executable, "-m", "omega_lock.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == f"omega-lock {__version__}"


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
