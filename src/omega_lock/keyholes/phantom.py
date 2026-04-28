# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""PhantomKeyhole ??deterministic synthetic calibration target.

The caller sees a 12-parameter black box. Internally, only 3 parameters
(`alpha`, `window`, `use_smoother`) meaningfully affect fitness; the other
9 are decoys with tiny orthogonal couplings. This is the canonical
"unknown lock" for exercising Omega-Lock end-to-end without external data.

Typical usage:
    train = PhantomKeyhole(seed=42)
    test  = PhantomKeyhole(seed=1337)     # same structure, different noise
    judge = PhantomKeyhole(seed=7, n_events=400)
    result = run_p1(
        train_target=train,
        test_target=test,
        validation_target=judge,
        config=P1Config(...),
    )

Hidden structure (the calibrator must rediscover via stress measurement):
    - AR(1) latent series x[t] with ? = 0.7
    - Observations obs[t] = x[t] + 琯_obs  (observation noise)
    - Labels label[t] = +1 if x[t] > LABEL_Z else -1   (hidden threshold)
    - Effective params: alpha (threshold), window (lookback), long_mode (direction flip)
    - Decoy params contribute a sub-percent fitness coupling (non-zero for
      realism; far below any effective param's per-evaluation swing)

Strategy (parametric):
    val[t] = mean(obs[t-window : t])
    if long_mode:  fire if val > alpha       (trend following)
    else:          fire if val < -alpha      (reversion ??the wrong sign here)
    reward += label[t] per fire

Neutral placement is deliberately "wrong" on all three effective axes so
the baseline is poor (n_trials ??50 but reward strongly negative) and the
grid can clearly improve ??which is what Omega-Lock is meant to do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omega_lock.target import EvalResult, ParamSpec


# Hidden structure ??not exposed.
_RHO: float = 0.7                   # AR(1) persistence of latent x
_LABEL_Z: float = 0.3               # latent-space threshold for label = +1
_OBSERVATION_NOISE: float = 0.10    # std of obs = x + 琯


@dataclass
class PhantomKeyhole:
    """Deterministic 12-param keyhole.

    Args:
        seed: PRNG seed ??determines the event stream.
        n_events: length of the synthetic series.
        noise: observation noise std (keep small vs latent std=1).
    """
    seed: int = 42
    n_events: int = 300
    noise: float = _OBSERVATION_NOISE

    _obs: np.ndarray = field(init=False, repr=False)
    _labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        innov_scale = math.sqrt(1.0 - _RHO * _RHO)
        x = np.empty(self.n_events, dtype=float)
        x[0] = rng.normal(0.0, 1.0)
        for t in range(1, self.n_events):
            x[t] = _RHO * x[t - 1] + rng.normal(0.0, innov_scale)
        self._obs = x + rng.normal(0.0, self.noise, self.n_events)
        self._labels = np.where(x > _LABEL_Z, 1.0, -1.0)

    def param_space(self) -> list[ParamSpec]:
        return [
            # --- effective trio (hidden; only these three move fitness meaningfully) ---
            # Neutrals deliberately placed in a poor region so the baseline
            # fitness is much worse than the grid optimum ??demonstrating
            # Omega-Lock's improvement rather than starting already tuned.
            # Ground truth (for benchmark use, NOT visible to calibrators):
            #   true effectives = {alpha, window, long_mode}
            #   true optimum    ??{alpha=0.35, window=3, long_mode=True}  (empirical)
            ParamSpec(name="alpha",     dtype="float", low=0.0, high=1.0, neutral=0.2),
            ParamSpec(name="window",    dtype="int",   low=3,   high=30,  neutral=16),
            ParamSpec(name="long_mode", dtype="bool",  neutral=False),

            # --- decoys (coupling is ~1% of effective param swing) ---
            ParamSpec(name="decoy_scale",  dtype="float", low=0.0,  high=1.0,   neutral=0.5),
            ParamSpec(name="decoy_offset", dtype="float", low=0.0,  high=10.0,  neutral=5.0),
            ParamSpec(name="decoy_bias",   dtype="float", low=-1.0, high=1.0,   neutral=0.0),
            ParamSpec(name="decoy_mag",    dtype="float", low=0.0,  high=100.0, neutral=50.0),
            ParamSpec(name="decoy_ofi",    dtype="float", low=0.1,  high=0.9,   neutral=0.5, ofi_biased=True),
            ParamSpec(name="decoy_mult",   dtype="int",   low=1,    high=100,   neutral=50),
            ParamSpec(name="decoy_exp",    dtype="int",   low=0,    high=10,    neutral=5),
            ParamSpec(name="decoy_mode",   dtype="int",   low=0,    high=3,     neutral=1),
            ParamSpec(name="decoy_flag",   dtype="bool",  neutral=True),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        alpha = float(params["alpha"])
        window = int(params["window"])
        long_mode = bool(params["long_mode"])

        obs = self._obs
        labels = self._labels
        n = obs.shape[0]

        # val[t] = mean of obs over the preceding `window` bars (pure backward-looking).
        if window < 1 or window >= n:
            return EvalResult(fitness=-999.0, n_trials=0, metadata={"error": "bad_window"})
        cs = np.concatenate(([0.0], np.cumsum(obs, dtype=float)))
        val = (cs[window:] - cs[:-window]) / float(window)   # len = n - window + 1
        # Align so val[t] uses obs[t-window : t], for t in [window, n).
        val_at = val[:-1]                                     # indices correspond to t = window..n-1

        # Direction flip: trend (long) vs reversion (short). labels favor
        # upside, so `long_mode=True` is the correct direction at any alpha.
        fires = (val_at > alpha) if long_mode else (val_at < -alpha)

        fire_idx = np.nonzero(fires)[0] + window              # absolute t-indices
        n_trials = int(fire_idx.size)
        reward = float(labels[fire_idx].sum()) if n_trials > 0 else 0.0

        # Decoy coupling ??deliberately small so effective params dominate
        # stress, but non-zero so KC-2's top/bot ratio is finite and meaningful.
        decoy_coupling = (
            0.005 * (float(params.get("decoy_scale", 0.5))  - 0.5)
            + 0.003 * (float(params.get("decoy_offset", 5.0)) - 5.0) / 10.0
            + 0.002 * float(params.get("decoy_bias", 0.0))
            + 0.001 * (float(params.get("decoy_mag", 50.0))   - 50.0) / 100.0
            + 0.004 * (float(params.get("decoy_ofi", 0.5))    - 0.5)
            + 0.002 * (int(params.get("decoy_mult", 50))      - 50) / 100.0
            + 0.001 * (int(params.get("decoy_exp", 5))        - 5) / 10.0
            + 0.002 * (int(params.get("decoy_mode", 1))       - 1) / 3.0
            + (0.003 if bool(params.get("decoy_flag", True)) else 0.0)
        )

        fitness = reward + decoy_coupling
        return EvalResult(
            fitness=fitness,
            n_trials=n_trials,
            metadata={
                "reward_raw": reward,
                "decoy_coupling": decoy_coupling,
                "effective": {
                    "alpha": alpha,
                    "window": window,
                    "long_mode": long_mode,
                },
            },
        )

    # ?? Ground-truth introspection (for benchmark use only) ??????????????
    # These are NOT part of CalibrableTarget; a well-behaved calibrator
    # must not consult them. Benchmark code uses them to score how close
    # the calibrator came to the true hidden structure.

    @staticmethod
    def true_effective_params() -> set[str]:
        return {"alpha", "window", "long_mode"}

    @staticmethod
    def true_optimum_params() -> dict[str, Any]:
        # Empirical optimum found by exhaustive 1D sweep at seed=42.
        # Approximate since seed-dependent; use as reference center.
        return {"alpha": 0.35, "window": 3, "long_mode": True}

    @staticmethod
    def true_importance_ranking() -> list[str]:
        """Ground-truth stress ordering (most ??least important).
        Compared against measured stress via Spearman ?."""
        return ["alpha", "long_mode", "window"]
