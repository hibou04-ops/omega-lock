#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Tier-aware "born-docked" presence-lint for the omega 3-repo family.

This is the **Tier C / C4** deliverable of the omega docking hardlock (see
``DOCKING.md`` C4 and ``omega-docking-plan/FINAL_PLAN.md``). It is the optional
"teeth": a tiny, offline, deterministic lint that mechanically enforces that
every cross-repo coupling in the C2 registry carries its guard -- so the
"born docked" rule (C1) is checked by a machine, not just trusted as prose.

It performs only deterministic local filesystem checks. It does NOT query PyPI,
GitHub, package registries, or live provider APIs, and it does NOT import any of
the three repos.

Tier-awareness is load-bearing (DOCKING.md C1/C4)
-------------------------------------------------
The three repos are NOT symmetrically coupled, so the lint is NOT uniform:

* **Tier A (runtime seam, omega-lock <-> omegaprompt):** guarded by a producer
  self-check + a consumer fail-loud test + a consumer canary workflow, and
  **pinned by a pyproject dependency range** (``omega-lock>=0.3.0,<0.4.0`` in
  omegaprompt's ``pyproject.toml``).
* **Tier B (doc-citation seam, antemortem-cli -> omega-lock):** guarded by an
  offline citation-drift script + a namespace-invariant test + a CI checkout
  job, and pinned by a **CI checkout ``ref`` (an immutable SHA)** -- NEVER by a
  dependency. The load-bearing asymmetry is that antemortem-cli carries
  **zero** ``import omega_lock`` / zero omega-lock dependency.

A naive "every coupling's pin must be a pyproject dependency" check would
false-fail antemortem-cli, or worse pressure adding an omega-lock dependency to
it -- the cardinal asymmetry sin. So this lint NEVER looks for a runtime pin on
Tier B; for Tier B the tier-aware analog of "the pin exists" is "the CI checkout
job exists", plus the positive assertion of zero omega_lock imports.

Cross-repo scope / robustness
-----------------------------
The lint spans 3 sibling repos and can only fully run where all three are
checked out. Presence is evaluated **per-repo, not per-tier**:

* A **missing sibling repo** -> SKIP (clearly noted; does NOT hard-fail). This
  is what keeps omega-lock's single-repo CI green when run there alone.
* A **missing guard inside a present repo** -> FAIL (nonzero exit).
* An **asymmetry breach** (an ``import omega_lock`` in antemortem ``src/``) ->
  FAIL (nonzero exit).

Because Tier A spans two repos, if one is absent its side is SKIPped while the
other repo's side still runs (omega-lock's producer guards are still checked
even when omegaprompt is absent).

Exit status
-----------
* ``0`` -- no real missing-guard / asymmetry breach (skips are not failures).
* ``1`` -- at least one FAIL (a present repo is missing a required guard, or an
  asymmetry breach was found).
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only.
    tomllib = None  # type: ignore[assignment]


# omega-lock is the family anchor (DOCKING.md lives at its root). This script
# lives in omega-lock/scripts/, so the omega-lock root is two parents up.
OMEGA_LOCK_ROOT = Path(__file__).resolve().parent.parent
# Siblings default to dirs next to the omega-lock checkout.
DEFAULT_OMEGAPROMPT_ROOT = OMEGA_LOCK_ROOT.parent / "omegaprompt"
DEFAULT_ANTEMORTEM_ROOT = OMEGA_LOCK_ROOT.parent / "antemortem-cli"

# Tier A pin (DOCKING.md C2): omegaprompt pyproject depends on this range.
TIER_A_PIN_PACKAGE = "omega-lock"
TIER_A_PIN_SPEC = ">=0.3.0,<0.4.0"
# Tier B "pin" (DOCKING.md C2/C4): the CI checkout job id, NOT a dependency.
TIER_B_CI_JOB_ID = "omega-lock-citation-drift"
# Optional nice-to-have: the immutable SHA the Tier B checkout pins to.
TIER_B_EXPECTED_SHA = "c03b8ac3c97752f64796dee49f9f11ab90cbce7d"

Status = Literal["PASS", "FAIL", "SKIP"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


@dataclass
class LintReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(
        self,
        name: str,
        status: Status,
        message: str,
        details: tuple[str, ...] = (),
    ) -> None:
        self.results.append(CheckResult(name, status, message, details))

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "FAIL"]

    @property
    def exit_code(self) -> int:
        # SKIP never contributes to a nonzero exit; only a real FAIL does.
        return 1 if self.failures else 0


# --------------------------------------------------------------------------- #
# Primitive checks
# --------------------------------------------------------------------------- #
def _check_file_present(
    report: LintReport,
    label: str,
    repo_root: Path,
    rel: str,
) -> None:
    """FAIL if a required guard file is absent inside a PRESENT repo."""
    path = repo_root / rel
    if path.is_file():
        report.add(label, "PASS", f"present: {rel}")
    else:
        report.add(
            label,
            "FAIL",
            f"MISSING required guard: {rel}",
            (f"expected file at: {path}",),
        )


def _normalize_spec(spec: str) -> str:
    """Strip all whitespace so ``>= 0.3.0, < 0.4.0`` == ``>=0.3.0,<0.4.0``."""
    return "".join(spec.split())


def _check_tier_a_pin(report: LintReport, omegaprompt_root: Path) -> None:
    """Tier A pin: omegaprompt pyproject depends on omega-lock>=0.3.0,<0.4.0.

    Parsed via tomllib (not raw grep) and whitespace-normalized so formatting
    variants of the same constraint still pass.
    """
    label = "tier-A pin (omegaprompt pyproject dep)"
    pyproject = omegaprompt_root / "pyproject.toml"
    if not pyproject.is_file():
        report.add(label, "FAIL", f"MISSING pyproject.toml: {pyproject}")
        return
    if tomllib is None:  # pragma: no cover - Python < 3.11 only.
        report.add(
            label,
            "FAIL",
            "tomllib unavailable (needs Python >= 3.11) -- cannot verify pin",
        )
        return
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        report.add(label, "FAIL", f"could not parse pyproject.toml: {exc}")
        return

    deps = data.get("project", {}).get("dependencies", [])
    want = f"{TIER_A_PIN_PACKAGE}{_normalize_spec(TIER_A_PIN_SPEC)}"
    matches = [d for d in deps if _normalize_spec(str(d)) == want]
    if matches:
        report.add(label, "PASS", f"pinned: {matches[0]}")
        return

    # Surface what omega-lock dep (if any) was actually found, to aid triage.
    found = [d for d in deps if TIER_A_PIN_PACKAGE in _normalize_spec(str(d))]
    detail = (
        (f"found instead: {found}",)
        if found
        else ("no omega-lock dependency found in [project.dependencies]",)
    )
    report.add(
        label,
        "FAIL",
        f"Tier A pin not '{TIER_A_PIN_PACKAGE}{TIER_A_PIN_SPEC}'",
        detail,
    )


def _check_tier_b_ci_job(report: LintReport, antemortem_root: Path) -> None:
    """Tier B 'pin' analog: the CI checkout job exists in ci.yml.

    This is the tier-aware substitute for a runtime pin. We deliberately do NOT
    require -- and never look for -- an omega-lock dependency on Tier B. We
    match the stable job id (DOCKING.md C2) by simple text presence to avoid a
    PyYAML dependency; the SHA presence is reported as an advisory note only.
    """
    label = "tier-B pin analog (antemortem CI checkout job)"
    ci = antemortem_root / ".github" / "workflows" / "ci.yml"
    if not ci.is_file():
        report.add(label, "FAIL", f"MISSING ci.yml: {ci}")
        return
    text = ci.read_text(encoding="utf-8")
    if f"{TIER_B_CI_JOB_ID}:" not in text:
        report.add(
            label,
            "FAIL",
            f"MISSING CI job '{TIER_B_CI_JOB_ID}' in ci.yml",
            (f"checked: {ci}",),
        )
        return
    # Advisory: confirm the immutable SHA pin is the one the design froze.
    if TIER_B_EXPECTED_SHA in text:
        report.add(
            label,
            "PASS",
            f"CI job '{TIER_B_CI_JOB_ID}' present; pinned SHA {TIER_B_EXPECTED_SHA[:7]}",
        )
    else:
        report.add(
            label,
            "PASS",
            f"CI job '{TIER_B_CI_JOB_ID}' present (note: expected SHA "
            f"{TIER_B_EXPECTED_SHA[:7]} not found; checkout ref may have moved)",
        )


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        # Skip caches; rglob already excludes nothing by default.
        if "__pycache__" in path.parts:
            continue
        yield path


def _file_imports_omega_lock(path: Path) -> bool:
    """True if the file imports the ``omega_lock`` package (AST-accurate).

    AST avoids false-positives on prose/docstrings and on lookalike names such
    as ``omega_lock_foo`` or ``mini_omega_lock``. Falls back to an anchored
    text scan only if the file cannot be parsed.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Unparsable file: anchored line scan as a conservative fallback.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import omega_lock") or stripped.startswith(
                "from omega_lock"
            ):
                # Guard against `import omega_lock_foo` via boundary char.
                rest = stripped.split("omega_lock", 1)[1][:1]
                if rest in ("", " ", ".", ",", "\t"):
                    return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "omega_lock" or name.startswith("omega_lock."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "omega_lock" or mod.startswith("omega_lock."):
                return True
    return False


def _check_tier_b_asymmetry(report: LintReport, antemortem_root: Path) -> None:
    """Load-bearing asymmetry: antemortem src/ has ZERO omega_lock imports."""
    label = "tier-B asymmetry (antemortem src/: zero omega_lock imports)"
    src = antemortem_root / "src"
    if not src.is_dir():
        report.add(
            label,
            "FAIL",
            f"MISSING antemortem src/ dir: {src}",
        )
        return
    offenders = [
        str(p.relative_to(antemortem_root))
        for p in _iter_python_files(src)
        if _file_imports_omega_lock(p)
    ]
    if offenders:
        report.add(
            label,
            "FAIL",
            "ASYMMETRY BREACH: antemortem src/ imports omega_lock "
            "(Tier B must stay doc-only -- zero runtime dependency)",
            tuple(sorted(offenders)),
        )
    else:
        report.add(label, "PASS", "no omega_lock import in antemortem src/")


# --------------------------------------------------------------------------- #
# Per-repo guard runners (presence is per-repo, not per-tier)
# --------------------------------------------------------------------------- #
def _run_omega_lock_checks(report: LintReport, root: Path) -> None:
    """Tier A producer-side guards (live in omega-lock)."""
    if not root.is_dir():
        report.add(
            "omega-lock repo",
            "SKIP",
            f"sibling not present: {root}",
        )
        return
    report.add("omega-lock repo", "PASS", f"present: {root}")
    # Tier A producer guards.
    _check_file_present(
        report, "tier-A producer manifest", root, "src/omega_lock/contract.py"
    )
    _check_file_present(
        report,
        "tier-A producer self-check",
        root,
        "tests/test_contract_manifest.py",
    )


def _run_omegaprompt_checks(report: LintReport, root: Path) -> None:
    """Tier A consumer-side guards + pin (live in omegaprompt)."""
    if not root.is_dir():
        report.add(
            "omegaprompt repo",
            "SKIP",
            f"sibling not present: {root}",
        )
        return
    report.add("omegaprompt repo", "PASS", f"present: {root}")
    _check_file_present(
        report,
        "tier-A consumer test",
        root,
        "tests/test_omega_lock_contract.py",
    )
    _check_file_present(
        report,
        "tier-A consumer canary workflow",
        root,
        ".github/workflows/omega-lock-compat.yml",
    )
    _check_tier_a_pin(report, root)


def _run_antemortem_checks(report: LintReport, root: Path) -> None:
    """Tier B doc-citation guards + asymmetry (live in antemortem-cli)."""
    if not root.is_dir():
        report.add(
            "antemortem-cli repo",
            "SKIP",
            f"sibling not present: {root}",
        )
        return
    report.add("antemortem-cli repo", "PASS", f"present: {root}")
    _check_file_present(
        report,
        "tier-B citation script",
        root,
        "scripts/check_omega_lock_citations.py",
    )
    _check_file_present(
        report,
        "tier-B citation-invariant test",
        root,
        "tests/test_omega_lock_citation_invariant.py",
    )
    # Tier-aware: CI checkout job stands in for the (forbidden) runtime pin.
    _check_tier_b_ci_job(report, root)
    # Load-bearing asymmetry: positive assertion of zero omega_lock imports.
    _check_tier_b_asymmetry(report, root)


def run_checks(
    omega_lock_root: Path | None = None,
    omegaprompt_root: Path | None = None,
    antemortem_root: Path | None = None,
) -> LintReport:
    """Run the full tier-aware presence-lint and return a structured report.

    Any repo path left ``None`` falls back to its sibling-layout default. A
    repo path that does not exist is SKIPped (not failed). This function never
    raises on a missing sibling and never imports any of the three repos.
    """
    omega_lock_root = omega_lock_root or OMEGA_LOCK_ROOT
    omegaprompt_root = omegaprompt_root or DEFAULT_OMEGAPROMPT_ROOT
    antemortem_root = antemortem_root or DEFAULT_ANTEMORTEM_ROOT

    report = LintReport()
    _run_omega_lock_checks(report, Path(omega_lock_root))
    _run_omegaprompt_checks(report, Path(omegaprompt_root))
    _run_antemortem_checks(report, Path(antemortem_root))
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _format_report(report: LintReport) -> str:
    lines: list[str] = []
    lines.append("omega docking presence-lint (Tier C / C4, tier-aware)")
    lines.append("=" * 56)
    for r in report.results:
        lines.append(f"[{r.status:<4}] {r.name}: {r.message}")
        for d in r.details:
            lines.append(f"         - {d}")
    lines.append("-" * 56)
    n_pass = sum(1 for r in report.results if r.status == "PASS")
    n_fail = len(report.failures)
    n_skip = sum(1 for r in report.results if r.status == "SKIP")
    lines.append(f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    if n_fail:
        lines.append("RESULT: FAIL (missing guard or asymmetry breach)")
    elif n_skip:
        lines.append(
            "RESULT: OK (no breach; one or more siblings SKIPped -- "
            "run where all three repos are checked out for full coverage)"
        )
    else:
        lines.append("RESULT: OK (all three repos present, all guards intact)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tier-aware 'born-docked' presence-lint for the omega 3-repo "
            "family (Tier C / C4). Asserts each cross-repo coupling carries "
            "its guard. Missing sibling repos are SKIPped; a present repo "
            "missing a guard, or an asymmetry breach, exits nonzero."
        )
    )
    parser.add_argument(
        "--omega-lock-root",
        type=Path,
        default=None,
        help=f"omega-lock checkout (default: {OMEGA_LOCK_ROOT})",
    )
    parser.add_argument(
        "--omegaprompt-root",
        type=Path,
        default=None,
        help=f"omegaprompt checkout (default sibling: {DEFAULT_OMEGAPROMPT_ROOT})",
    )
    parser.add_argument(
        "--antemortem-root",
        type=Path,
        default=None,
        help=(
            "antemortem-cli checkout "
            f"(default sibling: {DEFAULT_ANTEMORTEM_ROOT})"
        ),
    )
    args = parser.parse_args(argv)

    report = run_checks(
        omega_lock_root=args.omega_lock_root,
        omegaprompt_root=args.omegaprompt_root,
        antemortem_root=args.antemortem_root,
    )
    print(_format_report(report))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
