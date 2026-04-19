"""6T SRAM bitcell — analytical surrogate for methodology demo.

This is NOT SPICE-accurate. The model is a pedagogical surrogate whose only
job is to give `omega_lock.audit` a realistic-shaped target: multiple
parameters, multiple corners, a hard constraint structure (margins + leakage),
and an interior optimum that sensible calibration should find.

Do not quote numbers from this model in any technical paper.

Key equations (simplified textbook forms):
    V_T        = k_B * T / q                           thermal voltage
    I_leak     = I0 * (W/L) * exp((Vgs - Vth) / (n * V_T))   subthreshold
    I_read     = 0.5 * mu_n * Cox * (W_pd/L) * (Vdd - Vth_n)^2   saturation
    SNM        ~ k1 * (Vdd - Vth_n) * (beta-1)/(beta+1)   beta-ratio approx
    WM         ~ k2 * (Vdd - Vth_p) - k3 * w_ratio_pu     simplified
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from omega_lock.target import CalibrableTarget, EvalResult, ParamSpec
from omega_lock.audit import Constraint


# ── Physical constants (rough, not process-specific) ──────────────────────
K_B = 1.380649e-23       # Boltzmann constant (J/K)
Q   = 1.602176634e-19    # electron charge (C)
MU_N = 0.04              # NMOS mobility (m^2/V/s, rough)
COX  = 2e-3              # oxide capacitance per area (F/m^2, rough)
I0   = 3e-6              # subthreshold leakage prefactor (A) — tuned for demo scale
N_SUB = 1.3              # subthreshold slope factor

# SNM / write margin coefficients — tuned so interior optimum exists
K1 = 0.9     # SNM coefficient
K2 = 0.4     # WM coefficient (voltage term)
K3 = 0.15    # WM coefficient (pull-up width penalty)

C_BL = 100e-15  # bitline capacitance (F) — fixed, not a search parameter


def thermal_voltage(t_k: float) -> float:
    return K_B * t_k / Q


def leakage_current(w_over_l: float, vgs: float, vth: float, t_k: float) -> float:
    """Per-device subthreshold leakage. W/L is a dimensionless ratio."""
    vt = thermal_voltage(t_k)
    if vt <= 0:
        return 0.0
    return I0 * w_over_l * math.exp((vgs - vth) / (N_SUB * vt))


def read_current(w_pd_over_l: float, vdd: float, vth_n: float) -> float:
    """Saturation-region read current approximation."""
    v_ov = max(0.0, vdd - vth_n)
    return 0.5 * MU_N * COX * w_pd_over_l * v_ov ** 2


def read_snm(vdd: float, vth_n: float, beta_ratio: float) -> float:
    """Simplified beta-ratio SNM in volts."""
    if beta_ratio <= 0:
        return 0.0
    ratio_factor = (beta_ratio - 1.0) / (beta_ratio + 1.0)
    return max(0.0, K1 * (vdd - vth_n) * ratio_factor)


def write_margin(vdd: float, vth_p: float, w_ratio_pu: float) -> float:
    """Simplified write margin in volts."""
    return max(0.0, K2 * (vdd - vth_p) - K3 * w_ratio_pu)


# ── PVT corners ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Corner:
    name: str
    temp_k: float
    vdd_mul: float
    vth_mul: float


PVT_CORNERS: list[Corner] = [
    Corner("TT", 300.0, 1.00, 1.00),
    Corner("FF", 233.0, 1.10, 0.90),
    Corner("SS", 398.0, 0.90, 1.10),
    Corner("FS", 233.0, 1.00, 0.90),
    Corner("SF", 398.0, 1.00, 1.10),
]


def _apply_corner(params: dict[str, float], corner: Corner) -> dict[str, float]:
    """Scale voltage / threshold by corner multipliers. Widths / length unchanged."""
    return {
        "vdd":        params["vdd"]   * corner.vdd_mul,
        "vth_n":      params["vth_n"] * corner.vth_mul,
        "vth_p":      params["vth_p"] * corner.vth_mul,
        "w_ratio_pd": params["w_ratio_pd"],
        "w_ratio_pu": params["w_ratio_pu"],
        "l_channel":  params["l_channel"],
    }


def eval_corner(params: dict[str, float], corner: Corner) -> dict[str, float]:
    """All metrics for a single PVT corner."""
    p = _apply_corner(params, corner)
    vdd, vth_n, vth_p = p["vdd"], p["vth_n"], p["vth_p"]
    pd_ratio, pu_ratio = p["w_ratio_pd"], p["w_ratio_pu"]
    l_nm = p["l_channel"]
    t_k = corner.temp_k

    beta = pd_ratio
    snm_v = read_snm(vdd, vth_n, beta)

    wm_v = write_margin(vdd, vth_p, pu_ratio)

    l_scale = 40.0 / max(l_nm, 1.0)
    leak_pd = leakage_current(pd_ratio * l_scale, 0.0, vth_n, t_k)
    leak_ac = leakage_current(1.0      * l_scale, 0.0, vth_n, t_k)
    leak_pu = leakage_current(pu_ratio * l_scale, 0.0, vth_p, t_k)
    leak_total_a = 2 * leak_pd + 2 * leak_ac + 2 * leak_pu
    leak_na = leak_total_a * 1e9

    i_read = read_current(pd_ratio, vdd, vth_n)
    read_delay_ns = (C_BL * vdd / max(i_read, 1e-15)) * 1e9

    return {
        "corner":          corner.name,
        "read_snm_mv":     snm_v * 1000.0,
        "write_margin_mv": wm_v  * 1000.0,
        "leakage_na":      leak_na,
        "read_delay_ns":   read_delay_ns,
    }


# ── CalibrableTarget implementation ───────────────────────────────────────

class BitcellTarget:
    """6T SRAM bitcell across a set of PVT corners.

    evaluate(params) returns a single fitness that aggregates worst-case
    metrics across all corners, plus per-corner + worst-case breakdowns
    in metadata for constraint predicates to read.
    """

    def __init__(self, corners: list[Corner] | None = None, seed: int | None = None) -> None:
        self.corners = list(corners) if corners is not None else list(PVT_CORNERS)
        self.seed = seed

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="vdd",        dtype="float", neutral=0.90, low=0.60, high=1.20),
            ParamSpec(name="vth_n",      dtype="float", neutral=0.35, low=0.20, high=0.50),
            ParamSpec(name="vth_p",      dtype="float", neutral=0.35, low=0.20, high=0.50),
            ParamSpec(name="w_ratio_pd", dtype="float", neutral=2.0,  low=1.0,  high=4.0),
            ParamSpec(name="w_ratio_pu", dtype="float", neutral=1.0,  low=0.5,  high=2.0),
            ParamSpec(name="l_channel",  dtype="float", neutral=40.0, low=20.0, high=100.0),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        per_corner = [eval_corner(params, c) for c in self.corners]

        snm_worst   = min(r["read_snm_mv"]     for r in per_corner)
        wm_worst    = min(r["write_margin_mv"] for r in per_corner)
        leak_worst  = max(r["leakage_na"]      for r in per_corner)
        delay_worst = max(r["read_delay_ns"]   for r in per_corner)

        fitness = snm_worst + wm_worst - 50.0 * leak_worst - 5.0 * delay_worst

        return EvalResult(
            fitness=float(fitness),
            n_trials=0,
            metadata={
                "read_snm_mv_worst":     snm_worst,
                "write_margin_mv_worst": wm_worst,
                "leakage_na_worst":      leak_worst,
                "read_delay_ns_worst":   delay_worst,
                "per_corner":            per_corner,
            },
        )


# ── Demo constraints ──────────────────────────────────────────────────────

DEMO_CONSTRAINTS: list[Constraint] = [
    Constraint(
        "read_snm_gt_150mv",
        lambda p, r: r.metadata["read_snm_mv_worst"] > 150.0,
        "Worst-corner read SNM must exceed 150 mV",
    ),
    Constraint(
        "write_margin_gt_100mv",
        lambda p, r: r.metadata["write_margin_mv_worst"] > 100.0,
        "Worst-corner write margin must exceed 100 mV",
    ),
    Constraint(
        "leakage_lt_5na",
        lambda p, r: r.metadata["leakage_na_worst"] < 5.0,
        "Worst-corner leakage must stay below 5 nA per cell",
    ),
]
