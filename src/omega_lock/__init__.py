# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Omega-Lock — sensitivity-driven coordinate descent calibration framework.

Methodology (from research/omega_lock_p1/):
    1. Measure perturbation sensitivity (stress) of each parameter
    2. Select top-K stress params (unlock set), fix the rest (lock)
    3. Grid search on K-dim subspace
    4. Walk-forward validation with Pearson rank gate

Designed for any parameterized system via the CalibrableTarget protocol.

Public API:
    from omega_lock import (
        CalibrableTarget, ParamSpec, EvalResult,
        LockedParams, measure_stress, select_unlock_top_k,
        GridSearch, WalkForward, HybridFitness,
        KCThresholds, check_kc2, check_kc4,
        run_p1, P1Config, P1Result,
    )
"""
from omega_lock.target import CalibrableTarget, ParamSpec, EvalResult
from omega_lock.params import LockedParams, clip, default_epsilon
from omega_lock.adapters import CallableAdapter
from omega_lock.stress import measure_stress, gini_coefficient, select_unlock_top_k, StressResult
from omega_lock.grid import GridSearch, ZoomingGridSearch, grid_points, grid_points_in
from omega_lock.random_search import RandomSearch, top_quartile_fitness, compare_to_grid
from omega_lock.walk_forward import WalkForward, pearson
from omega_lock.fitness import BaseFitness, HybridFitness
from omega_lock.kill_criteria import KCThresholds, check_kc2, check_kc4, KCReport
from omega_lock.orchestrator import (
    run_p1, P1Config, P1Result,
    run_p1_iterative, IterativeConfig, IterativeResult,
)

# P2 (TPE) is an optional path — importable only if optuna is installed.
# The import below always succeeds; run_p2_tpe() raises a clear ImportError
# at call time if optuna is missing.
from omega_lock.p2_tpe import run_p2_tpe, P2Config, P2Result

# Objective benchmark scorecard (RAGAS-style for calibration methods)
from omega_lock.benchmark import (
    BenchmarkSpec, CalibrationMethod, BenchmarkRow, BenchmarkReport,
    MethodSummary, run_benchmark,
    compute_effective_recall, compute_effective_precision,
    compute_param_L2_error, compute_fitness_gap_pct,
    compute_generalization_gap, compute_spearman,
)

# Audit surface — method-agnostic trail + constraints over any CalibrableTarget.
from omega_lock.audit import (
    AuditingTarget, Constraint, AuditedRun, AuditReport,
    make_report, render_scorecard,
)

__version__ = "0.1.7"

__all__ = [
    "CalibrableTarget", "ParamSpec", "EvalResult",
    "LockedParams", "clip", "default_epsilon",
    "CallableAdapter",
    "measure_stress", "gini_coefficient", "select_unlock_top_k", "StressResult",
    "GridSearch", "ZoomingGridSearch", "grid_points", "grid_points_in",
    "RandomSearch", "top_quartile_fitness", "compare_to_grid",
    "WalkForward", "pearson",
    "BaseFitness", "HybridFitness",
    "KCThresholds", "check_kc2", "check_kc4", "KCReport",
    "run_p1", "P1Config", "P1Result",
    "run_p1_iterative", "IterativeConfig", "IterativeResult",
    "run_p2_tpe", "P2Config", "P2Result",
    "BenchmarkSpec", "CalibrationMethod", "BenchmarkRow", "BenchmarkReport",
    "MethodSummary", "run_benchmark",
    "compute_effective_recall", "compute_effective_precision",
    "compute_param_L2_error", "compute_fitness_gap_pct",
    "compute_generalization_gap", "compute_spearman",
    "AuditingTarget", "Constraint", "AuditedRun", "AuditReport",
    "make_report", "render_scorecard",
]
