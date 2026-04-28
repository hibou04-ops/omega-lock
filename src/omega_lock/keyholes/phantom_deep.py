# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""PhantomKeyholeDeep ??20-param synthetic target with effective_dim=6.

A deeper keyhole for exercising `run_p1_iterative`'s coordinate-descent
across multiple rounds. The caller sees 20 parameters; internally, six
meaningfully move fitness, organized as two nearly-orthogonal signal
channels so the fitness landscape is (approximately) additive over the
two effective trios.

Typical usage:
    train = PhantomKeyholeDeep(seed=42)
    test  = PhantomKeyholeDeep(seed=1337)
    result = run_p1_iterative(
        train_target=train,
        test_target=test,
        config=IterativeConfig(rounds=3, per_round_unlock_k=3),
    )

Hidden structure (two independent AR(1) channels, each with its own
threshold/lookback/mode trio):
    Channel A (stronger; labels 짹1):
        latent x[t], obs_a[t] = x[t] + 琯_a
        labels_a[t] = +1 if x[t] > LABEL_Z else -1
        val_a[t] = mean(obs_a[t-window : t])
        fires_a = (val_a > alpha) if long_mode else (val_a < -alpha)
        reward_a = 誇 labels_a[fires_a]
        Effectives: alpha, window, long_mode

    Channel B (weaker; labels 짹0.6):
        latent y[t] (independent of x), obs_b[t] = y[t] + 琯_b
        labels_b[t] = +0.6 if y[t] > LABEL_Z else -0.6
        z_b[t] = EMA(obs_b, alpha_ema) if use_ema else obs_b
        val_b[t] = mean(z_b[t-horizon : t])
        fires_b = (val_b > beta) if long_mode_b (true neutral=False) else ...
        Note: long_mode_b is NOT an effective (held via use_ema's
        direction + sign of beta is sufficient). The third channel-B
        effective is `use_ema` (smoothing on/off ??changes label
        alignment noticeably because smoothing washes out mean-reversion).
        Effectives: beta, horizon, use_ema

Total fitness = reward_a + reward_b + tiny decoy coupling.
Because x and y are independent, perturbing any channel-A effective
leaves channel B's fires/labels untouched (and vice versa) ??stress is
diagonal across the two channels, so the landscape is additive and
coordinate-descent across rounds genuinely discovers new axes.

Channel A is tuned to dominate (label magnitude 1.0 vs 0.6). So with
unlock_k=3, a single P1 round picks all three channel-A effectives, not
channel B ??and iterative's round 2 is where channel B's trio surfaces.

Neutrals are placed in a "poor" region on both channels so the baseline
is bad and calibration has room to improve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omega_lock.target import EvalResult, ParamSpec


# Hidden structure ??not exposed.
_RHO_A: float = 0.7                 # AR(1) persistence of latent x (channel A)
_RHO_B: float = 0.6                 # AR(1) persistence of latent y (channel B)
_LABEL_Z_A: float = 0.3             # latent-space threshold for label_a = +1
_LABEL_Z_B: float = 0.25            # latent-space threshold for label_b = +0.6
_OBS_NOISE_A: float = 0.10          # std of obs_a = x + 琯
_OBS_NOISE_B: float = 0.12          # std of obs_b = y + 琯
_LABEL_MAG_A: float = 1.0           # channel A dominates (per-fire reward magnitude)
_LABEL_MAG_B: float = 0.6           # channel B weaker ??stress ranks A > B
_EMA_ALPHA: float = 0.35            # fixed EMA smoothing coefficient when use_ema=True


@dataclass
class PhantomKeyholeDeep:
    """Deterministic 20-param keyhole with effective_dim=6 across two channels.

    Args:
        seed: PRNG seed ??determines both event streams.
        n_events: length of each synthetic series.
        noise_a: observation noise std for channel A.
        noise_b: observation noise std for channel B.
    """
    seed: int = 42
    n_events: int = 1000
    noise_a: float = _OBS_NOISE_A
    noise_b: float = _OBS_NOISE_B

    _obs_a: np.ndarray = field(init=False, repr=False)
    _obs_b: np.ndarray = field(init=False, repr=False)
    _labels_a: np.ndarray = field(init=False, repr=False)
    _labels_b: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        # --- Channel A: latent x, AR(1) with ?_A ---
        innov_a = math.sqrt(1.0 - _RHO_A * _RHO_A)
        x = np.empty(self.n_events, dtype=float)
        x[0] = rng.normal(0.0, 1.0)
        for t in range(1, self.n_events):
            x[t] = _RHO_A * x[t - 1] + rng.normal(0.0, innov_a)
        self._obs_a = x + rng.normal(0.0, self.noise_a, self.n_events)
        self._labels_a = np.where(x > _LABEL_Z_A, _LABEL_MAG_A, -_LABEL_MAG_A)

        # --- Channel B: latent y, AR(1) with ?_B, independent innovations ---
        innov_b = math.sqrt(1.0 - _RHO_B * _RHO_B)
        y = np.empty(self.n_events, dtype=float)
        y[0] = rng.normal(0.0, 1.0)
        for t in range(1, self.n_events):
            y[t] = _RHO_B * y[t - 1] + rng.normal(0.0, innov_b)
        self._obs_b = y + rng.normal(0.0, self.noise_b, self.n_events)
        self._labels_b = np.where(y > _LABEL_Z_B, _LABEL_MAG_B, -_LABEL_MAG_B)

    def param_space(self) -> list[ParamSpec]:
        return [
            # --- Channel A effectives (3) ---
            # Range caps at 0.6 to keep grid from picking pathologically rare
            # fire configs (慣=1.0 ??<20 fires/600 events, tripping KC-3).
            ParamSpec(name="alpha",      dtype="float", low=0.0, high=0.6, neutral=0.15),
            ParamSpec(name="window",     dtype="int",   low=3,   high=30,  neutral=16),
            ParamSpec(name="long_mode",  dtype="bool",  neutral=False),

            # --- Channel B effectives (3) ---
            ParamSpec(name="beta",       dtype="float", low=0.0, high=0.6, neutral=0.15),
            ParamSpec(name="horizon",    dtype="int",   low=3,   high=30,  neutral=16),
            ParamSpec(name="use_ema",    dtype="bool",  neutral=False),

            # --- Float decoys (9 more floats beyond the 2 effective floats ??11 floats; mix) ---
            ParamSpec(name="decoy_scale",  dtype="float", low=0.0,  high=1.0,   neutral=0.5),
            ParamSpec(name="decoy_offset", dtype="float", low=0.0,  high=10.0,  neutral=5.0),
            ParamSpec(name="decoy_bias",   dtype="float", low=-1.0, high=1.0,   neutral=0.0),
            ParamSpec(name="decoy_mag",    dtype="float", low=0.0,  high=100.0, neutral=50.0),
            ParamSpec(name="decoy_gain",   dtype="float", low=0.0,  high=5.0,   neutral=2.5),
            ParamSpec(name="decoy_drift",  dtype="float", low=-5.0, high=5.0,   neutral=0.0),
            ParamSpec(name="decoy_ratio",  dtype="float", low=0.0,  high=1.0,   neutral=0.5),
            ParamSpec(name="decoy_ofi",    dtype="float", low=0.1,  high=0.9,   neutral=0.5, ofi_biased=True),
            ParamSpec(name="decoy_temp",   dtype="float", low=0.0,  high=10.0,  neutral=5.0),
            ParamSpec(name="decoy_rate",   dtype="float", low=0.0,  high=1.0,   neutral=0.5),

            # --- Int decoys (3) ---
            ParamSpec(name="decoy_mult",   dtype="int",   low=1,    high=100,   neutral=50),
            ParamSpec(name="decoy_exp",    dtype="int",   low=0,    high=10,    neutral=5),
            ParamSpec(name="decoy_mode",   dtype="int",   low=0,    high=3,     neutral=1),

            # --- Bool decoy (1) ---
            ParamSpec(name="decoy_flag",   dtype="bool",  neutral=True),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        # --- Channel A effectives ---
        alpha = float(params["alpha"])
        window = int(params["window"])
        long_mode = bool(params["long_mode"])
        # --- Channel B effectives ---
        beta = float(params["beta"])
        horizon = int(params["horizon"])
        use_ema = bool(params["use_ema"])

        obs_a = self._obs_a
        labels_a = self._labels_a
        obs_b = self._obs_b
        labels_b = self._labels_b
        n = obs_a.shape[0]

        # --- Validate windows ---
        if window < 1 or window >= n:
            return EvalResult(fitness=-999.0, n_trials=0, metadata={"error": "bad_window"})
        if horizon < 1 or horizon >= n:
            return EvalResult(fitness=-999.0, n_trials=0, metadata={"error": "bad_horizon"})

        # --- Channel A: rolling mean over `window`, fire by trend/reversion + alpha ---
        cs_a = np.concatenate(([0.0], np.cumsum(obs_a, dtype=float)))
        val_a = (cs_a[window:] - cs_a[:-window]) / float(window)   # len = n - window + 1
        val_a_at = val_a[:-1]                                       # t = window..n-1
        fires_a = (val_a_at > alpha) if long_mode else (val_a_at < -alpha)
        fire_idx_a = np.nonzero(fires_a)[0] + window
        n_trials_a = int(fire_idx_a.size)
        reward_a = float(labels_a[fire_idx_a].sum()) if n_trials_a > 0 else 0.0

        # --- Channel B: optional EMA, then rolling mean over `horizon` ---
        if use_ema:
            # EMA with fixed 慣 ??use_ema=True turns on smoothing.
            # Smoothing dampens val_b range, shifting the optimal `beta`
            # and the sign of correlation with labels_b ??real effect.
            z_b = np.empty_like(obs_b)
            z_b[0] = obs_b[0]
            a = _EMA_ALPHA
            for i in range(1, n):
                z_b[i] = a * obs_b[i] + (1.0 - a) * z_b[i - 1]
        else:
            z_b = obs_b
        cs_b = np.concatenate(([0.0], np.cumsum(z_b, dtype=float)))
        val_b = (cs_b[horizon:] - cs_b[:-horizon]) / float(horizon)
        val_b_at = val_b[:-1]
        # Channel B uses `beta` as a magnitude threshold in "trend" direction always
        # (no separate long_mode on B). The neutral use_ema=False is the WRONG
        # smoothing choice for the optimal mean-direction ??so enabling smoothing
        # at the right beta moves fitness.
        fires_b = val_b_at > beta
        fire_idx_b = np.nonzero(fires_b)[0] + horizon
        n_trials_b = int(fire_idx_b.size)
        reward_b = float(labels_b[fire_idx_b].sum()) if n_trials_b > 0 else 0.0

        # --- Decoy coupling ??small, orthogonal, non-zero ---
        decoy_coupling = (
            0.005 * (float(params.get("decoy_scale", 0.5))  - 0.5)
            + 0.003 * (float(params.get("decoy_offset", 5.0)) - 5.0) / 10.0
            + 0.002 * float(params.get("decoy_bias", 0.0))
            + 0.001 * (float(params.get("decoy_mag", 50.0))   - 50.0) / 100.0
            + 0.002 * (float(params.get("decoy_gain", 2.5))   - 2.5) / 5.0
            + 0.001 * float(params.get("decoy_drift", 0.0))   / 5.0
            + 0.002 * (float(params.get("decoy_ratio", 0.5))  - 0.5)
            + 0.004 * (float(params.get("decoy_ofi", 0.5))    - 0.5)
            + 0.001 * (float(params.get("decoy_temp", 5.0))   - 5.0) / 10.0
            + 0.002 * (float(params.get("decoy_rate", 0.5))   - 0.5)
            + 0.002 * (int(params.get("decoy_mult", 50))      - 50) / 100.0
            + 0.001 * (int(params.get("decoy_exp", 5))        - 5) / 10.0
            + 0.002 * (int(params.get("decoy_mode", 1))       - 1) / 3.0
            + (0.003 if bool(params.get("decoy_flag", True)) else 0.0)
        )

        fitness = reward_a + reward_b + decoy_coupling
        return EvalResult(
            fitness=fitness,
            n_trials=n_trials_a + n_trials_b,
            metadata={
                "reward_a": reward_a,
                "reward_b": reward_b,
                "n_trials_a": n_trials_a,
                "n_trials_b": n_trials_b,
                "decoy_coupling": decoy_coupling,
                "effective": {
                    "alpha": alpha,
                    "window": window,
                    "long_mode": long_mode,
                    "beta": beta,
                    "horizon": horizon,
                    "use_ema": use_ema,
                },
            },
        )

    # ?? Ground-truth introspection (benchmark use only) ??????????????????
    # NOT part of CalibrableTarget. Calibrators must never consult these;
    # benchmark code uses them to score discovery quality.

    @staticmethod
    def true_effective_params() -> set[str]:
        return {"alpha", "window", "long_mode", "beta", "horizon", "use_ema"}

    @staticmethod
    def true_optimum_params() -> dict[str, Any]:
        return {
            "alpha": 0.25, "window": 3, "long_mode": True,
            "beta": 0.2, "horizon": 3, "use_ema": True,
        }

    @staticmethod
    def true_importance_ranking() -> list[str]:
        """Channel-A magnitudes (짹1.0) dominate channel-B (짹0.6), so
        the six effectives rank A-first within channel."""
        return ["alpha", "long_mode", "beta", "window", "use_ema", "horizon"]
