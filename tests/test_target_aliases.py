"""Back-compat alias contract for the domain-neutral rename (A1).

EvalResult.n_trials   -> sample_count
ParamSpec.ofi_biased  -> stress_suppressed

Old names must keep working (construction + read + write) so 0.2.x
callers do not break.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from omega_lock.target import EvalResult, ParamSpec


# ── EvalResult.sample_count / n_trials alias ──

def test_evalresult_canonical_sample_count():
    r = EvalResult(fitness=1.0, sample_count=7)
    assert r.sample_count == 7
    assert r.n_trials == 7  # read alias


def test_evalresult_n_trials_kwarg_populates_sample_count():
    r = EvalResult(fitness=1.0, n_trials=5)
    assert r.sample_count == 5
    assert r.n_trials == 5


def test_evalresult_positional_count_is_sample_count():
    # second positional was n_trials, is now sample_count — same value
    r = EvalResult(1.0, 9)
    assert r.sample_count == 9


def test_evalresult_default_count_zero():
    r = EvalResult(fitness=2.0)
    assert r.sample_count == 0
    assert r.n_trials == 0


def test_evalresult_n_trials_setter_writes_sample_count():
    r = EvalResult(fitness=1.0)
    r.n_trials = 4
    assert r.sample_count == 4


def test_evalresult_both_aliases_same_value_ok():
    r = EvalResult(fitness=1.0, sample_count=3, n_trials=3)
    assert r.sample_count == 3


def test_evalresult_both_aliases_conflict_raises():
    with pytest.raises(ValueError):
        EvalResult(fitness=1.0, sample_count=3, n_trials=4)


def test_evalresult_independent_default_containers():
    a = EvalResult(fitness=1.0)
    b = EvalResult(fitness=2.0)
    a.metadata["x"] = 1
    assert b.metadata == {}  # no shared mutable default
    assert a.artifacts is not b.artifacts


def test_evalresult_equality_uses_count_value():
    assert EvalResult(1.0, n_trials=2) == EvalResult(1.0, sample_count=2)


# ── ParamSpec.stress_suppressed / ofi_biased alias ──

def test_paramspec_canonical_stress_suppressed():
    p = ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5, stress_suppressed=True)
    assert p.stress_suppressed is True
    assert p.ofi_biased is True  # read alias


def test_paramspec_ofi_biased_kwarg_sets_stress_suppressed():
    p = ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5, ofi_biased=True)
    assert p.stress_suppressed is True
    assert p.ofi_biased is True


def test_paramspec_default_not_suppressed():
    p = ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5)
    assert p.stress_suppressed is False
    assert p.ofi_biased is False


def test_paramspec_still_frozen():
    p = ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5)
    with pytest.raises(Exception):
        p.stress_suppressed = True  # type: ignore[misc]


def test_paramspec_validation_still_runs():
    with pytest.raises(ValueError):
        ParamSpec(name="bad", dtype="float", low=1.0, high=0.0, neutral=0.5)  # low > high


# ── serialization boundary: asdict() emits canonical field names ──

def test_evalresult_asdict_keys_are_canonical():
    keys = set(dc.asdict(EvalResult(1.0, n_trials=5)))
    assert keys == {"fitness", "sample_count", "metadata", "artifacts"}
    assert "n_trials" not in keys  # property, not a field


def test_paramspec_asdict_keys_are_canonical():
    p = ParamSpec(name="a", dtype="float", low=0.0, high=1.0, neutral=0.5, ofi_biased=True)
    keys = set(dc.asdict(p))
    assert "stress_suppressed" in keys
    assert "ofi_biased" not in keys


# ── hashing + cross-alias equality (ParamSpec is frozen) ──

def test_paramspec_hash_and_cross_alias_eq():
    a = ParamSpec(name="x", dtype="float", low=0.0, high=1.0, neutral=0.5, stress_suppressed=True)
    b = ParamSpec(name="x", dtype="float", low=0.0, high=1.0, neutral=0.5, ofi_biased=True)
    assert a == b
    assert len({a, b}) == 1  # equal -> dedup in a set


# ── dataclasses.replace() contract ──

def test_replace_with_canonical_works():
    r = dc.replace(EvalResult(1.0, n_trials=5), sample_count=9)
    assert r.sample_count == 9 and r.n_trials == 9


def test_replace_with_alias_kwarg_is_unsupported():
    # documented limitation: replace re-passes the stored sample_count,
    # so the alias path conflicts; use the canonical field name instead.
    with pytest.raises((TypeError, ValueError)):
        dc.replace(EvalResult(1.0, n_trials=5), n_trials=9)


# ── alias kwargs are keyword-only ──

def test_n_trials_cannot_be_passed_positionally():
    # signature is (fitness, sample_count, metadata, artifacts, *, n_trials)
    # so a 5th positional is rejected; n_trials must be keyword.
    with pytest.raises(TypeError):
        EvalResult(1.0, 0, {}, {}, 5)  # type: ignore[misc]


# ── explicit-0 vs alias conflict edge ──

def test_explicit_zero_count_conflicts_with_nonzero_alias():
    with pytest.raises(ValueError):
        EvalResult(1.0, 0, n_trials=5)  # sample_count=0 explicitly, n_trials=5 -> conflict
