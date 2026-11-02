# SPDX-License-Identifier: Apache-2.0
"""Four offline, keyless deployment-gate cases; requires omega-lock[p2].

Run from a checkout: python examples/reproduce_optuna_traps.py
The script asserts the unsafe cases are blocked and exits zero on success.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lock import audit_optuna_study, gate_scores  # noqa: E402


def main() -> int:
    try:
        optuna = cast(Any, import_module("optuna"))
    except ImportError as exc:
        raise SystemExit(
            'Install the optional dependency first: python -m pip install ".[p2]"'
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="maximize")
    for index, score in enumerate([1.0, 2.0, 3.0]):
        study.add_trial(optuna.trial.create_trial(
            params={"x": index},
            distributions={"x": optuna.distributions.IntDistribution(0, 2)},
            value=score,
        ))
    skipped = audit_optuna_study(study, top_n=3)
    assert skipped.passed is True and skipped.kc_report.status == "SKIP"
    assert skipped.holdout_evaluated is False and skipped.deployment_authorized is False
    assert replace(skipped, deployment_authorized=True).deployment_authorized is False
    shifted = audit_optuna_study(
        study, holdout_evaluate=lambda params: float(params["x"]) - 100.0, top_n=3,
    )
    assert shifted.pearson == 1.0 and shifted.degradation_gap == 101.0
    assert shifted.passed is False and shifted.deployment_authorized is False
    assert gate_scores([1, 2, 3], [-100, -99, -98]).deployment_authorized is False
    normal = audit_optuna_study(
        study, holdout_evaluate=lambda params: float(params["x"]) + 1.0, top_n=3,
    )
    assert normal.passed and normal.deployment_authorized and normal.degradation_gap == 0.0
    normalized = optuna.create_study(direction="maximize")
    for index, score in enumerate([0.7, 0.8, 0.9]):
        normalized.add_trial(optuna.trial.create_trial(
            params={"x": index},
            distributions={"x": optuna.distributions.IntDistribution(0, 2)}, value=score,
        ))
    collapsed = audit_optuna_study(
        normalized, holdout_evaluate=lambda params: [0.01, 0.02, 0.03][int(params["x"])], top_n=3,
    )
    assert collapsed.pearson == 1.0 and collapsed.degradation_gap is not None
    assert collapsed.degradation_gap < 1.0 and collapsed.relative_degradation == 0.975
    assert not collapsed.passed and not collapsed.deployment_authorized
    assert not gate_scores([0.7, 0.8, 0.9], [0.01, 0.02, 0.03]).deployment_authorized
    rows: list[dict[str, Any]] = []
    for name, report in (("no_holdout", skipped), ("location_shift", shifted),
                         ("normalized_collapse", collapsed), ("normal", normal)):
        rows.append({
            "case": name, "passed": report.passed, "status": report.kc_report.status,
            "holdout_evaluated": report.holdout_evaluated,
            "deployment_authorized": report.deployment_authorized,
            "pearson": report.pearson, "degradation_gap": report.degradation_gap,
            "relative_degradation": report.relative_degradation,
            "relative_degradation_status": report.relative_degradation_status,
        })
    print(json.dumps(rows, indent=2, allow_nan=False))
    print("All four deployment-gate cases reproduced without API keys or network calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
