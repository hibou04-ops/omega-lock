# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Walk-forward gate demo — why "best score" is not deployable.

A deterministic, offline case study of search overfitting and how the
Omega-Lock gates respond to it:

    1. A synthetic target has a real, transferable optimum (fitness ~5.0
       inside a "rated" operating envelope) and a fragile region where a
       slice-dependent noise term can spike far higher (up to ~6 above
       the envelope limit). The noise re-draws on every data slice, so a
       noise spike found on the train slice does NOT transfer.
    2. Naive selection — "take the highest train score" (`best_any`) —
       picks a lucky-noise point in the fragile region.
    3. Omega-Lock's walk-forward gate (KC-4) re-evaluates the train-best
       top-N on a *test* slice. The train ranking does not survive
       (Pearson collapses), so the run is stamped FAIL:KC-4 instead of
       shipping the lucky candidate.
    4. Feasible-best selection (a declared hard constraint: stay inside
       the rated envelope) picks a candidate that holds up on a holdout
       slice that no selection step ever consulted.

Everything is seeded and hash-based: no RNG state, no network, no API
keys. Repeated runs print identical numbers. Runtime is well under 60s.

Since 0.3.4 the case-study engine lives in ``omega_lock._demo`` so the
installed console command ``omega-lock demo`` can run it from a wheel
(where ``examples/`` does not ship). This example re-exports the same
symbols and prints the identical narrative.

Run:
    python examples/walkforward_gate_demo.py
        (or, once installed: omega-lock demo)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/walkforward_gate_demo.py` without pip install
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omega_lock._demo import (
    FRAGILE_NOISE_AMPLITUDE,
    HOLDOUT_SEED,
    RATED_ENVELOPE,
    RATED_GAIN_MAX,
    SIGNAL_PEAK,
    STABLE_NOISE_AMPLITUDE,
    TEST_SEED,
    TRAIN_SEED,
    NoisySliceTarget,
    run_demo,
)

__all__ = [
    "FRAGILE_NOISE_AMPLITUDE",
    "HOLDOUT_SEED",
    "RATED_ENVELOPE",
    "RATED_GAIN_MAX",
    "SIGNAL_PEAK",
    "STABLE_NOISE_AMPLITUDE",
    "TEST_SEED",
    "TRAIN_SEED",
    "NoisySliceTarget",
    "main",
]


def main() -> int:
    return run_demo()


if __name__ == "__main__":
    sys.exit(main())
