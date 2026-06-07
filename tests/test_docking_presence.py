# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tests for the Tier C / C4 tier-aware docking presence-lint.

Two layers, by design (the happy real-layout path can never exercise the
FAIL/nonzero-exit contract, so a synthetic negative layer always runs):

* **Real-layout** -- skips unless all three sibling repos are checked out next
  to omega-lock; when present, asserts a clean PASS and zero exit. This is what
  keeps omega-lock's single-repo CI green (the lint is cross-repo by nature).
* **Synthetic** -- fabricates repos in ``tmp_path`` to prove: a missing sibling
  SKIPs (no fail), a missing guard in a present repo FAILs (nonzero exit), an
  ``import omega_lock`` in antemortem ``src/`` is an asymmetry breach (nonzero),
  and -- the load-bearing tier-awareness -- a Tier B repo with NO omega-lock
  dependency still PASSes (requiring one would be the cardinal asymmetry sin).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "check_docking_presence.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_docking_presence", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# --------------------------------------------------------------------------- #
# Synthetic-layout builders (always available; no siblings needed)
# --------------------------------------------------------------------------- #
def _make_omega_lock(root: Path) -> Path:
    repo = root / "omega-lock"
    (repo / "src" / "omega_lock").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "src" / "omega_lock" / "contract.py").write_text("X = 1\n", "utf-8")
    (repo / "tests" / "test_contract_manifest.py").write_text("# t\n", "utf-8")
    return repo


def _make_omegaprompt(root: Path, *, pin: str | None = "omega-lock>=0.3.0,<0.4.0") -> Path:
    repo = root / "omegaprompt"
    (repo / "tests").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "tests" / "test_omega_lock_contract.py").write_text("# t\n", "utf-8")
    (repo / ".github" / "workflows" / "omega-lock-compat.yml").write_text(
        "name: compat\n", "utf-8"
    )
    deps = f'    "{pin}",\n' if pin else ""
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'omegaprompt'\ndependencies = [\n"
        '    "typer>=0.12.0",\n'
        f"{deps}"
        "]\n",
        "utf-8",
    )
    return repo


def _make_antemortem(
    root: Path,
    *,
    job_id: str = "omega-lock-citation-drift",
    import_omega_lock: bool = False,
    omega_lock_dep: bool = False,
) -> Path:
    repo = root / "antemortem-cli"
    (repo / "src" / "antemortem").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "scripts" / "check_omega_lock_citations.py").write_text("# s\n", "utf-8")
    (repo / "tests" / "test_omega_lock_citation_invariant.py").write_text(
        "# t\n", "utf-8"
    )
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        f"jobs:\n  {job_id}:\n    runs-on: ubuntu-latest\n", "utf-8"
    )
    body = (
        "import os\n"
        + ("import omega_lock\n" if import_omega_lock else "")
        + "VALUE = 1\n"
    )
    (repo / "src" / "antemortem" / "core.py").write_text(body, "utf-8")
    # A Tier B repo MUST NOT need an omega-lock dependency; this flag exists
    # only to prove the lint does not require/penalize either presence/absence.
    dep = '    "omega-lock>=0.3.0,<0.4.0",\n' if omega_lock_dep else ""
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'antemortem-cli'\ndependencies = [\n"
        '    "typer>=0.12.0",\n'
        f"{dep}"
        "]\n",
        "utf-8",
    )
    return repo


def _statuses(report) -> dict[str, str]:
    return {r.name: r.status for r in report.results}


# --------------------------------------------------------------------------- #
# Synthetic: the happy synthetic baseline
# --------------------------------------------------------------------------- #
def test_synthetic_all_present_passes(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path),
        _make_antemortem(tmp_path),
    )
    assert report.exit_code == 0, _statuses(report)
    assert not report.failures


# --------------------------------------------------------------------------- #
# Synthetic: missing sibling -> SKIP, not FAIL
# --------------------------------------------------------------------------- #
def test_missing_sibling_skips_not_fails(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        tmp_path / "does-not-exist-omegaprompt",
        _make_antemortem(tmp_path),
    )
    assert report.exit_code == 0
    statuses = _statuses(report)
    assert statuses["omegaprompt repo"] == "SKIP"
    # Per-repo, not per-tier: omega-lock's Tier A producer side still ran.
    assert statuses["tier-A producer manifest"] == "PASS"
    assert statuses["tier-A producer self-check"] == "PASS"


def test_all_siblings_missing_skips_clean(tmp_path: Path):
    report = mod.run_checks(
        tmp_path / "no-ol", tmp_path / "no-op", tmp_path / "no-am"
    )
    assert report.exit_code == 0
    assert all(r.status == "SKIP" for r in report.results)


# --------------------------------------------------------------------------- #
# Synthetic: missing guard inside a PRESENT repo -> FAIL (nonzero)
# --------------------------------------------------------------------------- #
def test_missing_producer_guard_fails(tmp_path: Path):
    repo = _make_omega_lock(tmp_path)
    (repo / "tests" / "test_contract_manifest.py").unlink()
    report = mod.run_checks(
        repo, _make_omegaprompt(tmp_path), _make_antemortem(tmp_path)
    )
    assert report.exit_code == 1
    assert any(
        r.name == "tier-A producer self-check" and r.status == "FAIL"
        for r in report.results
    )


def test_missing_consumer_canary_fails(tmp_path: Path):
    op = _make_omegaprompt(tmp_path)
    (op / ".github" / "workflows" / "omega-lock-compat.yml").unlink()
    report = mod.run_checks(
        _make_omega_lock(tmp_path), op, _make_antemortem(tmp_path)
    )
    assert report.exit_code == 1
    assert any(
        r.name == "tier-A consumer canary workflow" and r.status == "FAIL"
        for r in report.results
    )


def test_missing_tier_a_pin_fails(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path, pin=None),
        _make_antemortem(tmp_path),
    )
    assert report.exit_code == 1
    assert any(
        r.name.startswith("tier-A pin") and r.status == "FAIL"
        for r in report.results
    )


def test_tier_a_pin_whitespace_tolerant(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path, pin="omega-lock >= 0.3.0, < 0.4.0"),
        _make_antemortem(tmp_path),
    )
    assert report.exit_code == 0, _statuses(report)


def test_wrong_tier_a_pin_range_fails(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path, pin="omega-lock>=0.4.0,<0.5.0"),
        _make_antemortem(tmp_path),
    )
    assert report.exit_code == 1


def test_missing_tier_b_ci_job_fails(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path),
        _make_antemortem(tmp_path, job_id="some-other-job"),
    )
    assert report.exit_code == 1
    assert any(
        r.name.startswith("tier-B pin analog") and r.status == "FAIL"
        for r in report.results
    )


# --------------------------------------------------------------------------- #
# Synthetic: asymmetry breach -> FAIL (nonzero)
# --------------------------------------------------------------------------- #
def test_asymmetry_breach_import_fails(tmp_path: Path):
    report = mod.run_checks(
        _make_omega_lock(tmp_path),
        _make_omegaprompt(tmp_path),
        _make_antemortem(tmp_path, import_omega_lock=True),
    )
    assert report.exit_code == 1
    breach = [
        r
        for r in report.results
        if r.name.startswith("tier-B asymmetry") and r.status == "FAIL"
    ]
    assert breach
    assert any("core.py" in d for d in breach[0].details)


def test_lookalike_import_is_not_a_breach(tmp_path: Path):
    am = _make_antemortem(tmp_path)
    # `omega_lock_helpers` / `mini_omega_lock` must NOT trip the AST scan.
    (am / "src" / "antemortem" / "helpers.py").write_text(
        "import omega_lock_helpers\nimport mini_omega_lock\n", "utf-8"
    )
    report = mod.run_checks(
        _make_omega_lock(tmp_path), _make_omegaprompt(tmp_path), am
    )
    assert report.exit_code == 0, _statuses(report)


def test_omega_lock_in_docstring_is_not_a_breach(tmp_path: Path):
    am = _make_antemortem(tmp_path)
    (am / "src" / "antemortem" / "doc.py").write_text(
        '"""This mentions import omega_lock in prose only."""\nX = 1\n', "utf-8"
    )
    report = mod.run_checks(
        _make_omega_lock(tmp_path), _make_omegaprompt(tmp_path), am
    )
    assert report.exit_code == 0, _statuses(report)


# --------------------------------------------------------------------------- #
# Tier-awareness: the cardinal asymmetry sin must NOT be enforced
# --------------------------------------------------------------------------- #
def test_tier_b_does_not_require_pyproject_pin(tmp_path: Path):
    """A Tier B repo with NO omega-lock dependency must still PASS.

    Requiring a runtime pin on Tier B (or even rewarding one) would defeat the
    load-bearing asymmetry. The lint must only check the CI checkout job +
    zero-import, never a pyproject dependency, for antemortem.
    """
    am = _make_antemortem(tmp_path, omega_lock_dep=False)
    report = mod.run_checks(
        _make_omega_lock(tmp_path), _make_omegaprompt(tmp_path), am
    )
    assert report.exit_code == 0, _statuses(report)
    # No check name should reference an antemortem pyproject dependency.
    assert not any(
        "antemortem" in r.name and "pin" in r.name and "pyproject" in r.message
        for r in report.results
    )


def test_tier_b_pyproject_pin_is_neither_required_nor_a_breach(tmp_path: Path):
    """Even if someone DID add an omega-lock dep, the import-scan is the real
    asymmetry guard -- a dep without an import is not what this lint policing.

    (The dep-absence is reported as a possible extension, not enforced here.)
    """
    am = _make_antemortem(tmp_path, omega_lock_dep=True, import_omega_lock=False)
    report = mod.run_checks(
        _make_omega_lock(tmp_path), _make_omegaprompt(tmp_path), am
    )
    # No import -> no breach from this lint's perspective.
    assert report.exit_code == 0, _statuses(report)


# --------------------------------------------------------------------------- #
# Real sibling layout (skips unless all three are checked out)
# --------------------------------------------------------------------------- #
def test_real_sibling_layout_passes_if_present():
    ol = mod.OMEGA_LOCK_ROOT
    op = mod.DEFAULT_OMEGAPROMPT_ROOT
    am = mod.DEFAULT_ANTEMORTEM_ROOT
    missing = [str(p) for p in (ol, op, am) if not p.is_dir()]
    if missing:
        pytest.skip(f"sibling repo(s) not checked out: {missing}")
    report = mod.run_checks(ol, op, am)
    assert report.exit_code == 0, (
        "real-layout docking presence-lint FAILED:\n"
        + "\n".join(
            f"  [{r.status}] {r.name}: {r.message}" for r in report.failures
        )
    )
