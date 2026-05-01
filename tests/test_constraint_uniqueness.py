# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P2: AuditingTarget rejects duplicate constraint names.

AuditReport.constraint_pass_counts() keys by constraint name. Pre-fix,
two constraints with the same name silently merged their pass counts
in the report — a copy-paste mistake produced a wrong-but-not-flagged
audit artifact.

The validation runs at wrap time so the failure is at the point of
configuration, not 100 evaluations into a run.
"""
from __future__ import annotations

from typing import Any

import pytest

from omega_lock.audit import AuditingTarget, Constraint
from omega_lock.target import EvalResult, ParamSpec


class _Stub:
    def param_space(self) -> list[ParamSpec]:
        return [ParamSpec(name="x", dtype="float", neutral=0.0, low=-1.0, high=1.0)]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        return EvalResult(fitness=1.0, n_trials=1)


def _c(name: str) -> Constraint:
    return Constraint(name=name, fn=lambda params, result: True)


def test_unique_constraint_names_accepted():
    AuditingTarget(_Stub(), constraints=[_c("a"), _c("b"), _c("c")])


def test_no_constraints_accepted():
    AuditingTarget(_Stub(), constraints=[])


def test_duplicate_constraint_names_rejected():
    with pytest.raises(ValueError, match="duplicate constraint names"):
        AuditingTarget(_Stub(), constraints=[_c("dup"), _c("dup")])


def test_duplicate_message_lists_dupes():
    with pytest.raises(ValueError, match=r"\['x', 'y'\]"):
        AuditingTarget(
            _Stub(),
            constraints=[
                _c("x"), _c("x"),
                _c("y"), _c("y"),
                _c("z"),
            ],
        )


def test_duplicate_validation_at_construction_not_evaluation():
    """Mistakes surface before any evaluate() call — not buried 100
    rounds deep into a run."""
    with pytest.raises(ValueError):
        AuditingTarget(_Stub(), constraints=[_c("same"), _c("same")])
