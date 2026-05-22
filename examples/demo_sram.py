"""SRAM bitcell calibration demo — audit-surface showcase.

Runs `omega_lock.run_p1` on a 6T SRAM bitcell analytical surrogate across 5
PVT corners with 3 hard constraints. Wraps the target with
`omega_lock.audit.AuditingTarget` to produce a reviewable scorecard + JSON
audit trail.

Usage (from repo root):
    python examples/demo_sram.py

Output:
    - stdout: the audit scorecard
    - output/audit_sram.json: full audit trail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the examples/ directory importable so `omega_lock_demos.sram` resolves.
_EXAMPLES_DIR = Path(__file__).parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
_SRC_DIR = _EXAMPLES_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from omega_lock import run_p1, P1Config
from omega_lock.audit import AuditingTarget, make_report, render_scorecard
from omega_lock.audit._types import AUDIT_REPORT_SCHEMA_VERSION, AuditReport
from omega_lock_demos.sram import BitcellTarget, PVT_CORNERS, DEMO_CONSTRAINTS


def _build_demo():
    target = BitcellTarget(corners=PVT_CORNERS, seed=42)

    wrapped = AuditingTarget(target, constraints=DEMO_CONSTRAINTS, target_role="train")
    wrapped.set_phase("search")

    result = run_p1(
        train_target=wrapped,
        config=P1Config(
            unlock_k=3,
            grid_points_per_axis=5,
            stress_verbose=False,
            grid_verbose=False,
        ),
    )

    report = make_report(wrapped, method="run_p1", seed=42)
    return result, report


def _run_ref(run) -> dict[str, object] | None:
    if run is None:
        return None
    return {
        "call_index": run.call_index,
        "fitness": round(float(run.fitness), 12),
        "constraints_failed": list(run.constraints_failed),
        "params": {
            key: round(float(value), 12)
            for key, value in sorted(run.params.items())
        },
    }


def build_check_summary() -> dict[str, object]:
    result, report = _build_demo()
    signed = report.to_dict(with_hash_chain=True)
    rehydrated = AuditReport.from_json(json.dumps(signed, sort_keys=True))
    hash_chain_valid = rehydrated.verify_hash_chain(signed["hash_chain"])

    pass_counts = report.constraint_pass_counts()
    fail_counts = {
        constraint.name: report.n_total - pass_counts[constraint.name]
        for constraint in report.constraints
    }
    best_any = report.best_any
    best_feasible = report.best_feasible
    feasible_best_differs = (
        best_any is not None
        and best_feasible is not None
        and best_any.call_index != best_feasible.call_index
    )

    markers = {
        "best_feasible_present": best_feasible is not None,
        "best_feasible_has_no_failed_constraints": (
            best_feasible is not None and not best_feasible.constraints_failed
        ),
        "hard_constraint_failures_visible": any(count > 0 for count in fail_counts.values()),
        "feasible_best_differs_from_absolute_best": feasible_best_differs,
        "absolute_best_is_infeasible": (
            best_any is not None and bool(best_any.constraints_failed)
        ),
        "hash_chain_valid": hash_chain_valid,
        "schema_roundtrip_valid": signed["schema_version"] == AUDIT_REPORT_SCHEMA_VERSION,
    }
    summary = {
        "demo": "demo_sram",
        "status": "PASS" if all(markers.values()) else "FAIL",
        "pipeline_status": result.status,
        "audit_schema_version": signed["schema_version"],
        "n_total": report.n_total,
        "n_feasible": report.n_feasible,
        "constraint_pass_counts": pass_counts,
        "constraint_fail_counts": fail_counts,
        "best_any": _run_ref(best_any),
        "best_feasible": _run_ref(best_feasible),
        "walk_forward_gate_result": "NOT_APPLICABLE_NO_TEST_TARGET",
        "hash_chain_length": len(signed["hash_chain"]),
        "markers": markers,
    }
    return summary


def _run_check() -> int:
    summary = build_check_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


def _run_demo() -> int:
    result, report = _build_demo()
    print(render_scorecard(report))

    if report.best_feasible is not None:
        bf = report.best_feasible
        print("Best feasible params:")
        for k, v in bf.params.items():
            print(f"  {k:12s} = {v:.4g}")
        print(f"Best feasible fitness: {bf.fitness:.4g}")
        print(f"  SNM worst:     {bf.metadata['read_snm_mv_worst']:.1f} mV")
        print(f"  WM worst:      {bf.metadata['write_margin_mv_worst']:.1f} mV")
        print(f"  leakage worst: {bf.metadata['leakage_na_worst']:.3f} nA")
    else:
        print("No feasible design found in the search grid.")
        if report.best_any is not None:
            ba = report.best_any
            print(f"Closest (infeasible): fitness={ba.fitness:.4g}")
            print(f"  Failed: {', '.join(ba.constraints_failed)}")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_sram.json"
    out_path.write_text(report.to_json(), encoding="utf-8")
    print(f"\nFull audit trail: {out_path}")
    print(f"Pipeline status:  {result.status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run deterministic self-check")
    args = parser.parse_args(argv)
    return _run_check() if args.check else _run_demo()


if __name__ == "__main__":
    sys.exit(main())
