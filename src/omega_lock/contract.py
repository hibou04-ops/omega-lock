# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Consumed-surface contract manifest (producer-internal, advisory).

A frozen enumeration of the *narrow* omega-lock surface that the downstream
``omegaprompt`` consumer actually reads. It is asserted by
``tests/test_contract_manifest.py`` against omega-lock's own public OUTPUT
(``to_summary()`` / ``to_dict()`` keys + call signatures) as an EARLY-WARNING
smoke: a producer rename of a consumed name reds at the omega-lock PR rather
than only at the consumer's next dependency bump.

This is ADVISORY, not the contract of record. The contract of record is the
consumer's own executable test
(``omegaprompt/tests/test_omega_lock_contract.py``), run against
``omega-lock@main`` by omegaprompt's compatibility CI -- that executes the real
reads and cannot silently stale. This manifest is a hand-maintained copy: it
only catches a rename of an *already-known* consumed name, and is deliberately
pruned to the consumed surface so normal internal refactors (timing fields,
artifact cleanup, canonical-name migrations) do NOT false-red it.

Deliberately NOT exported in ``omega_lock.__all__``: the consumer must not
import it (it keeps its own independent copy; that asymmetry is intentional and
avoids re-introducing the version-skew coupling this guard exists to remove).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumedContract:
    """The omega-lock surface omegaprompt consumes (verified by grep, 2026-06).

    Every field is asserted by ``test_contract_manifest.py`` against omega-lock's
    public output or call signatures -- there are no inert entries.
    """

    # Fields omega-lock READS off the EvalResult the target returns
    # (stress.py / grid.py / walk_forward.py).
    eval_result_read_fields: tuple[str, ...] = ("fitness", "sample_count")
    # ParamSpec ctor kwargs the consumer passes (targets/prompt_target.py).
    param_spec_ctor: tuple[str, ...] = ("name", "dtype", "neutral", "low", "high")
    # Nested wire-key the consumer reads off P1Result.grid_best
    # (= GridPoint.to_summary()).
    grid_summary_keys: tuple[str, ...] = ("unlocked",)
    # Keys the consumer reads off each P1Result.stress_results item
    # (= StressResult.to_dict()).
    stress_dict_keys: tuple[str, ...] = ("name", "raw_stress", "normalized_stress")
    # Call signatures (subset) the consumer relies on (runtime.py).
    run_p1_params: tuple[str, ...] = ("train_target", "test_target", "config")
    p1config_params: tuple[str, ...] = ("unlock_k",)
    measure_stress_params: tuple[str, ...] = ("target", "baseline_params", "baseline_result")


CONSUMED_CONTRACT = ConsumedContract()

# Advisory shape marker for this manifest (documentation only; not a gate).
# Bump when the consumed surface above legitimately changes.
CONTRACT_VERSION = "omega-lock.consumed-contract.v1"
