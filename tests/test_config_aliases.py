"""Dual-field / dual-key back-compat contract for the config-flag rename (A1 Slice 2, C).

exclude_ofi_in_unlock  -> exclude_suppressed_in_unlock   (config field)
top_k_ex_ofi (result)  -> top_k_excl_suppressed           (result field)

Unlike the EvalResult/ParamSpec alias (which drops the old key from asdict),
config flags keep BOTH names live for one release so old serialized
config_full / search_settings dicts round-trip. So these tests assert BOTH
keys are PRESENT.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from omega_lock.orchestrator import (
    IterativeConfig,
    P1Config,
    _iterative_search_settings,
    _p1_legacy_config,
    _p1_search_settings,
)
from omega_lock.p2_tpe import P2Config, _p2_legacy_config, _p2_search_settings


def _make(cls):
    if cls is IterativeConfig:
        return lambda **kw: cls(rounds=2, per_round_unlock_k=2, **kw)
    return cls


CONFIGS = [P1Config, IterativeConfig, P2Config]


@pytest.mark.parametrize("cls", CONFIGS)
def test_old_kwarg_sets_canonical(cls):
    cfg = _make(cls)(exclude_ofi_in_unlock=True)
    assert cfg.exclude_suppressed_in_unlock is True
    assert cfg.exclude_ofi_in_unlock is True  # mirror stays in sync


@pytest.mark.parametrize("cls", CONFIGS)
def test_new_kwarg_mirrors_old(cls):
    cfg = _make(cls)(exclude_suppressed_in_unlock=True)
    assert cfg.exclude_ofi_in_unlock is True
    assert cfg.exclude_suppressed_in_unlock is True


@pytest.mark.parametrize("cls", CONFIGS)
def test_default_is_false_both(cls):
    cfg = _make(cls)()
    assert cfg.exclude_suppressed_in_unlock is False
    assert cfg.exclude_ofi_in_unlock is False


@pytest.mark.parametrize("cls", CONFIGS)
def test_asdict_emits_both_keys(cls):
    keys = set(dc.asdict(_make(cls)()))
    assert "exclude_suppressed_in_unlock" in keys
    assert "exclude_ofi_in_unlock" in keys  # dual-key for config_full back-compat


def test_p1_dict_emitters_dual_key():
    for d in (
        _p1_legacy_config(P1Config()),
        _p1_search_settings(P1Config()),
        _iterative_search_settings(IterativeConfig(rounds=2, per_round_unlock_k=2)),
    ):
        assert "exclude_suppressed_in_unlock" in d
        assert "exclude_ofi_in_unlock" in d


def test_p2_dict_emitters_dual_key():
    for d in (_p2_legacy_config(P2Config()), _p2_search_settings(P2Config())):
        assert "exclude_suppressed_in_unlock" in d
        assert "exclude_ofi_in_unlock" in d


def test_config_full_round_trips_old_serialized_dict():
    # an old config_full carrying only the deprecated key must reconstruct
    old = dc.asdict(P1Config())
    old.pop("exclude_suppressed_in_unlock")  # simulate a pre-rename artifact
    old["exclude_ofi_in_unlock"] = True
    cfg = P1Config(**old)
    assert cfg.exclude_suppressed_in_unlock is True
