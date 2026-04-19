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

import sys
from pathlib import Path

# Make the examples/ directory importable so `omega_lock_demos.sram` resolves.
_EXAMPLES_DIR = Path(__file__).parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from omega_lock import run_p1, P1Config
from omega_lock.audit import AuditingTarget, make_report, render_scorecard
from omega_lock_demos.sram import BitcellTarget, PVT_CORNERS, DEMO_CONSTRAINTS


def main() -> int:
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


if __name__ == "__main__":
    sys.exit(main())
