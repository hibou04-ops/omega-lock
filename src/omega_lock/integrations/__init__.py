# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""omega_lock.integrations — bridges from third-party tuners into the gates.

Modules here adapt EXISTING artifacts of other tools (an Optuna study, for
example) into the omega-lock audit gates. They are import-safe without the
third-party package installed: the heavy import happens lazily inside the
bridge function, which raises a clean ImportError with an install hint.

Public API:
    from omega_lock.integrations import audit_optuna_study, StudyAuditReport
"""
from __future__ import annotations

from omega_lock.integrations.optuna_bridge import (
    StudyAuditReport,
    TrialCandidate,
    audit_optuna_study,
)

__all__ = [
    "StudyAuditReport",
    "TrialCandidate",
    "audit_optuna_study",
]
