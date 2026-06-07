# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Regression guard for the 0.3.2 sdist packaging fix.

The 0.3.1 sdist shipped ``tests/`` but NOT ``scripts/``, ``tests/fixtures/``,
or ``examples/``. Many tests load ``scripts/*.py`` at module-import time and
one (``test_demo_sram``) imports from ``examples/`` (via the shipped
``pythonpath``), so ``pytest --collect-only`` on the unpacked sdist threw
collection errors. MANIFEST.in now grafts those directories into the sdist
ONLY -- the wheel must stay clean (import package only).

This test builds the sdist + wheel once (module-scoped), then asserts
membership. It is a fast tarball/zip membership check rather than a nested
``pytest --collect-only`` subprocess (which is the documented manual VERIFY
step in RELEASE.md). If the build backend is unavailable the test SKIPs --
the same offline-tolerant posture as the wheel smoke test.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_dist(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build sdist + wheel into an isolated dist dir; return (sdist, wheel)."""
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "No module named build" in combined or "No module named 'build'" in combined:
            pytest.skip("python -m build is not available in this environment")
        pytest.skip(f"sdist/wheel build failed (treated as tooling-blocked):\n{combined[-1500:]}")

    sdists = list(out.glob("*.tar.gz"))
    wheels = list(out.glob("*.whl"))
    assert len(sdists) == 1, [p.name for p in sdists]
    assert len(wheels) == 1, [p.name for p in wheels]
    return sdists[0], wheels[0]


def _sdist_members(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as tf:
        # Strip the leading "omega_lock-<ver>/" prefix for readability.
        return ["/".join(n.split("/", 1)[1:]) if "/" in n else n for n in tf.getnames()]


def _wheel_members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_sdist_includes_scripts_tests_fixtures_examples(built_dist):
    sdist, _wheel = built_dist
    members = _sdist_members(sdist)
    member_set = set(members)

    # scripts/ -- loaded by module-level script-loader tests.
    assert "scripts/check_docking_presence.py" in member_set, "scripts/ missing from sdist"
    assert "scripts/run_golden_audit_cases.py" in member_set

    # tests/ + tests/fixtures/ -- the tests themselves and golden JSON.
    assert any(m.startswith("tests/") and m.endswith(".py") for m in members)
    assert any(
        m.startswith("tests/fixtures/golden_audits/") and m.endswith(".json")
        for m in members
    ), "tests/fixtures/golden_audits/*.json missing from sdist"

    # examples/ -- imported by test_demo_sram via the shipped pythonpath.
    assert "examples/demo_sram.py" in member_set, "examples/ missing from sdist"


def test_sdist_excludes_compiled_bytecode(built_dist):
    sdist, _wheel = built_dist
    members = _sdist_members(sdist)
    assert not any(m.endswith((".pyc", ".pyo")) for m in members)
    assert not any("__pycache__" in m for m in members)


def test_wheel_stays_clean_no_tests_scripts_examples(built_dist):
    _sdist, wheel = built_dist
    members = _wheel_members(wheel)
    # The wheel ships ONLY the import package (and dist-info metadata).
    assert not any(m.startswith("tests/") for m in members), "tests/ leaked into wheel"
    assert not any(m.startswith("scripts/") for m in members), "scripts/ leaked into wheel"
    assert not any(m.startswith("examples/") for m in members), "examples/ leaked into wheel"
    # The import package IS present.
    assert any(m.startswith("omega_lock/") for m in members)
