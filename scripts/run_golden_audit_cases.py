"""Generate and check deterministic offline golden audit cases.

The fixtures compare stable semantic fields, not wall-clock timings. Cases are
pure Python and use no network, provider, registry, or live API access.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock import EvalResult, KCThresholds, P1Config, ParamSpec, __version__, run_p1
from omega_lock.audit._types import (
    AUDIT_REPORT_SCHEMA_VERSION,
    AuditedRun,
    AuditReport,
    Constraint,
)


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "golden_audits"
GOLDEN_SCHEMA_VERSION = 1
FIXED_STARTED = "2026-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class Diagnostic:
    status: str
    name: str
    message: str


class _LinearTrainTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="x", dtype="float", neutral=0.0, low=0.0, high=1.0),
            ParamSpec(name="y", dtype="float", neutral=0.0, low=0.0, high=1.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(
            fitness=10.0 * float(params["x"]) + float(params["y"]),
            n_trials=100,
        )


class _MatchingWalkForwardTarget(_LinearTrainTarget):
    pass


class _AntiCorrelatedWalkForwardTarget(_LinearTrainTarget):
    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(
            fitness=-(10.0 * float(params["x"]) + float(params["y"])),
            n_trials=100,
        )


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def load_golden_case(case_id: str, fixture_dir: Path = FIXTURE_DIR) -> dict[str, Any]:
    return json.loads((fixture_dir / f"{case_id}.json").read_text(encoding="utf-8"))


def verify_signed_report(payload: Mapping[str, Any]) -> bool:
    report = AuditReport.from_json(json.dumps(payload, sort_keys=True))
    chain = payload.get("hash_chain")
    if not isinstance(chain, list):
        return False
    return report.verify_hash_chain(chain)


def _constraint(name: str, description: str) -> Constraint:
    return Constraint(name=name, fn=lambda _params, _result: True, description=description)


def _run(
    call_index: int,
    *,
    x: float,
    fitness: float,
    passed: tuple[str, ...] = (),
    failed: tuple[str, ...] = (),
    phase: str = "grid",
    role: str = "train",
    n_trials: int = 100,
) -> AuditedRun:
    return AuditedRun(
        params={"x": x},
        fitness=fitness,
        n_trials=n_trials,
        metadata={
            "_constraints_passed": passed,
            "_constraints_failed": failed,
            "case_call": call_index,
        },
        timestamp_iso=f"2026-01-01T00:00:{call_index:02d}+00:00",
        constraints_passed=passed,
        constraints_failed=failed,
        phase=phase,
        call_index=call_index,
        target_role=role,
        round_index=0,
    )


def _report(case_id: str, runs: tuple[AuditedRun, ...], constraints: tuple[Constraint, ...]) -> AuditReport:
    ended = f"2026-01-01T00:00:{max(len(runs) - 1, 0):02d}+00:00"
    return AuditReport(
        method=f"golden:{case_id}",
        omega_lock_version=__version__,
        seed=101,
        started_iso=FIXED_STARTED,
        ended_iso=ended,
        constraints=constraints,
        runs=runs,
    )


def _stable_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 12)
        return str(value)
    if isinstance(value, list):
        return [_stable_float(v) for v in value]
    if isinstance(value, tuple):
        return [_stable_float(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable_float(v) for k, v in value.items()}
    return value


def _stable_run(run: AuditedRun) -> dict[str, Any]:
    return {
        "call_index": run.call_index,
        "params": _stable_float(run.params),
        "fitness": _stable_float(run.fitness),
        "n_trials": run.n_trials,
        "metadata": _stable_float(run.metadata),
        "constraints_passed": list(run.constraints_passed),
        "constraints_failed": list(run.constraints_failed),
        "phase": run.phase,
        "target_role": run.target_role,
        "round_index": run.round_index,
    }


def _run_ref(run: AuditedRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "call_index": run.call_index,
        "params": _stable_float(run.params),
        "fitness": _stable_float(run.fitness),
        "constraints_failed": list(run.constraints_failed),
    }


def _semantic_report(report: AuditReport, *, with_hash_chain: bool = False) -> dict[str, Any]:
    report_payload = report.to_dict(with_hash_chain=with_hash_chain)
    out: dict[str, Any] = {
        "audit_schema_version": report_payload["schema_version"],
        "method": report.method,
        "omega_lock_version": report.omega_lock_version,
        "seed": report.seed,
        "constraints": report_payload["constraints"],
        "runs": [_stable_run(run) for run in report.runs],
        "summary": _stable_float(report_payload["summary"]),
        "best_any": _run_ref(report.best_any),
        "best_feasible": _run_ref(report.best_feasible),
    }
    if with_hash_chain:
        chain = report.hash_chain()
        out["hash_chain"] = chain
        out["hash_chain_valid"] = report.verify_hash_chain(chain)
    return out


def _golden_case(
    case_id: str,
    *,
    description: str,
    semantic: Mapping[str, Any],
    signed_report: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "golden_schema_version": GOLDEN_SCHEMA_VERSION,
        "case_id": case_id,
        "description": description,
        "offline": True,
        "semantic": _stable_float(dict(semantic)),
    }
    if signed_report is not None:
        data["signed_report"] = _stable_float(dict(signed_report))
    if extra is not None:
        data["extra"] = _stable_float(dict(extra))
    return data


def _all_constraints_pass() -> dict[str, Any]:
    constraints = (_constraint("x_nonnegative", "x must be >= 0"),)
    report = _report(
        "all_constraints_pass",
        (
            _run(0, x=0.0, fitness=1.0, passed=("x_nonnegative",)),
            _run(1, x=0.5, fitness=2.0, passed=("x_nonnegative",)),
            _run(2, x=1.0, fitness=3.0, passed=("x_nonnegative",)),
        ),
        constraints,
    )
    return _golden_case(
        "all_constraints_pass",
        description="All evaluated audit runs satisfy the declared hard constraint.",
        semantic=_semantic_report(report),
    )


def _hard_constraint_failure() -> dict[str, Any]:
    constraints = (_constraint("x_lte_one", "x must be <= 1"),)
    report = _report(
        "hard_constraint_failure",
        (
            _run(0, x=0.0, fitness=1.0, passed=("x_lte_one",)),
            _run(1, x=1.5, fitness=2.0, failed=("x_lte_one",)),
            _run(2, x=0.5, fitness=1.5, passed=("x_lte_one",)),
        ),
        constraints,
    )
    return _golden_case(
        "hard_constraint_failure",
        description="At least one raw candidate violates a declared hard constraint.",
        semantic=_semantic_report(report),
        extra={"failed_call_indexes": [1]},
    )


def _feasible_best_differs_from_absolute_best() -> dict[str, Any]:
    constraints = (_constraint("x_lte_one", "x must be <= 1"),)
    report = _report(
        "feasible_best_differs_from_absolute_best",
        (
            _run(0, x=0.0, fitness=1.0, passed=("x_lte_one",)),
            _run(1, x=2.0, fitness=10.0, failed=("x_lte_one",)),
            _run(2, x=1.0, fitness=4.0, passed=("x_lte_one",)),
        ),
        constraints,
    )
    return _golden_case(
        "feasible_best_differs_from_absolute_best",
        description="The raw best score violates constraints, so best_feasible differs.",
        semantic=_semantic_report(report),
        extra={
            "best_any_call_index": 1,
            "best_feasible_call_index": 2,
        },
    )


def _no_feasible_candidates() -> dict[str, Any]:
    constraints = (_constraint("never_ok", "fixture constraint that every run violates"),)
    report = _report(
        "no_feasible_candidates",
        (
            _run(0, x=0.0, fitness=1.0, failed=("never_ok",)),
            _run(1, x=1.0, fitness=2.0, failed=("never_ok",)),
            _run(2, x=2.0, fitness=3.0, failed=("never_ok",)),
        ),
        constraints,
    )
    return _golden_case(
        "no_feasible_candidates",
        description="Every candidate violates the hard constraint; best_feasible is absent.",
        semantic=_semantic_report(report),
        extra={"release_status": "FAIL:CONSTRAINTS"},
    )


def _p1_config() -> P1Config:
    return P1Config(
        unlock_k=2,
        grid_points_per_axis=3,
        walk_forward_top_n=4,
        kc_thresholds=KCThresholds(
            gini_min=0.0,
            top_bot_ratio_min=1.0,
            trade_count_min=1,
            pearson_min=0.9,
            trade_ratio_min=0.0,
        ),
        stress_verbose=False,
        grid_verbose=False,
        constraint_policy="prefer_feasible",
    )


def _stable_p1_result(result: Any) -> dict[str, Any]:
    grid_best = result.grid_best
    kc_reports = [
        {
            "name": report["name"],
            "status": report["status"],
            "detail": _stable_float(_stable_kc_detail(report)),
        }
        for report in result.kc_reports
    ]
    walk_forward = result.walk_forward
    return {
        "status": result.status,
        "grid_best": _stable_float(
            {
                "idx": grid_best["idx"],
                "unlocked": grid_best["unlocked"],
                "fitness": grid_best["fitness"],
                "n_trials": grid_best["n_trials"],
            }
            if grid_best is not None
            else None
        ),
        "walk_forward": _stable_float(
            {
                "top_n": walk_forward["top_n"],
                "pearson": walk_forward["pearson"],
                "pearson_status": walk_forward["pearson_status"],
                "pearson_computable": walk_forward["pearson_computable"],
                "trade_ratio_scaled": walk_forward["trade_ratio_scaled"],
                "test_best_trades": walk_forward["test_best_trades"],
            }
            if walk_forward is not None
            else None
        ),
        "kc_reports": kc_reports,
        "hard_failures": [r["name"] for r in kc_reports if r["status"] == "FAIL"],
    }


def _stable_kc_detail(report: Mapping[str, Any]) -> dict[str, Any]:
    detail = dict(report["detail"])
    if report["name"] == "KC-1":
        detail.pop("elapsed_s", None)
    return detail


def _walk_forward_gate_pass() -> dict[str, Any]:
    result = run_p1(
        train_target=_LinearTrainTarget(),
        test_target=_MatchingWalkForwardTarget(),
        config=_p1_config(),
    )
    return _golden_case(
        "walk_forward_gate_pass",
        description="Walk-forward preserves the train ranking and KC-4 passes.",
        semantic=_stable_p1_result(result),
    )


def _walk_forward_gate_fail() -> dict[str, Any]:
    result = run_p1(
        train_target=_LinearTrainTarget(),
        test_target=_AntiCorrelatedWalkForwardTarget(),
        config=_p1_config(),
    )
    return _golden_case(
        "walk_forward_gate_fail",
        description="Walk-forward rejects the train ranking and KC-4 blocks status.",
        semantic=_stable_p1_result(result),
    )


def _append_only_hash_chain() -> dict[str, Any]:
    constraints = (_constraint("x_nonnegative", "x must be >= 0"),)
    base = _report(
        "append_only_hash_chain",
        (
            _run(0, x=0.0, fitness=1.0, passed=("x_nonnegative",)),
            _run(1, x=0.5, fitness=2.0, passed=("x_nonnegative",)),
            _run(2, x=1.0, fitness=3.0, passed=("x_nonnegative",)),
        ),
        constraints,
    )
    extended = _report(
        "append_only_hash_chain",
        base.runs + (_run(3, x=1.5, fitness=4.0, passed=("x_nonnegative",)),),
        constraints,
    )
    base_chain = base.hash_chain()
    extended_chain = extended.hash_chain()
    return _golden_case(
        "append_only_hash_chain",
        description="Hash-chain enabled audit report preserves append-only prefix.",
        semantic={
            "base": _semantic_report(base, with_hash_chain=True),
            "extended": _semantic_report(extended, with_hash_chain=True),
            "append_only_prefix_preserved": (
                extended_chain[: len(base_chain)] == base_chain
            ),
            "non_append_mutation_would_change_prefix": True,
        },
        signed_report=extended.to_dict(with_hash_chain=True),
    )


def _schema_validation_roundtrip() -> dict[str, Any]:
    constraints = (_constraint("x_nonnegative", "x must be >= 0"),)
    report = _report(
        "schema_validation_roundtrip",
        (
            _run(0, x=0.0, fitness=1.0, passed=("x_nonnegative",)),
            _run(1, x=1.0, fitness=2.0, passed=("x_nonnegative",)),
        ),
        constraints,
    )
    signed = report.to_dict(with_hash_chain=True)
    rehydrated = AuditReport.from_json(json.dumps(signed, sort_keys=True))
    schema_mismatch_rejected = False
    bad = dict(signed)
    bad["schema_version"] = "omega-lock.audit-report.v999"
    try:
        AuditReport.from_json(json.dumps(bad, sort_keys=True))
    except ValueError:
        schema_mismatch_rejected = True
    return _golden_case(
        "schema_validation_roundtrip",
        description="Current audit schema round-trips and mismatched schema is rejected.",
        semantic={
            "original": _semantic_report(report, with_hash_chain=True),
            "roundtrip": _semantic_report(rehydrated, with_hash_chain=True),
            "roundtrip_semantics_equal": (
                _semantic_report(report, with_hash_chain=True)
                == _semantic_report(rehydrated, with_hash_chain=True)
            ),
            "schema_mismatch_rejected": schema_mismatch_rejected,
        },
        signed_report=signed,
    )


def build_golden_cases() -> dict[str, dict[str, Any]]:
    builders = [
        _all_constraints_pass,
        _hard_constraint_failure,
        _feasible_best_differs_from_absolute_best,
        _no_feasible_candidates,
        _walk_forward_gate_pass,
        _walk_forward_gate_fail,
        _append_only_hash_chain,
        _schema_validation_roundtrip,
    ]
    return {case["case_id"]: case for case in (builder() for builder in builders)}


def update_golden_cases(fixture_dir: Path = FIXTURE_DIR) -> list[Diagnostic]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[Diagnostic] = []
    for case_id, case in build_golden_cases().items():
        path = fixture_dir / f"{case_id}.json"
        path.write_text(canonical_json(case), encoding="utf-8")
        diagnostics.append(Diagnostic("PASS", case_id, f"updated {path.as_posix()}"))
    return diagnostics


def check_golden_cases(fixture_dir: Path = FIXTURE_DIR) -> list[Diagnostic]:
    expected_cases = build_golden_cases()
    diagnostics: list[Diagnostic] = []
    for case_id, case in expected_cases.items():
        path = fixture_dir / f"{case_id}.json"
        expected = canonical_json(case)
        if not path.exists():
            diagnostics.append(Diagnostic("FAIL", case_id, f"missing fixture {path}"))
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            diagnostics.append(Diagnostic("FAIL", case_id, f"fixture drift: {path}"))
            continue
        diagnostics.append(Diagnostic("PASS", case_id, "fixture matches"))

    expected_names = {f"{case_id}.json" for case_id in expected_cases}
    if fixture_dir.exists():
        for path in sorted(fixture_dir.glob("*.json")):
            if path.name not in expected_names:
                diagnostics.append(
                    Diagnostic("FAIL", path.stem, f"unexpected fixture {path}")
                )
    return diagnostics


def has_failures(diagnostics: Sequence[Diagnostic]) -> bool:
    return any(d.status == "FAIL" for d in diagnostics)


def format_diagnostics(diagnostics: Sequence[Diagnostic]) -> str:
    lines = [
        "Golden audit cases",
        f"Root: {ROOT}",
        "",
    ]
    for diagnostic in diagnostics:
        lines.append(f"[{diagnostic.status}] {diagnostic.name}: {diagnostic.message}")
    pass_count = sum(1 for d in diagnostics if d.status == "PASS")
    fail_count = sum(1 for d in diagnostics if d.status == "FAIL")
    lines.append("")
    lines.append(f"Summary: PASS={pass_count}, FAIL={fail_count}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if fixtures drift")
    mode.add_argument("--update", action="store_true", help="regenerate fixtures")
    args = parser.parse_args(argv)

    diagnostics = (
        update_golden_cases() if args.update else check_golden_cases()
    )
    print(format_diagnostics(diagnostics))
    return 1 if has_failures(diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
