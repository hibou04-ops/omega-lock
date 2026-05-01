# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reviewer P0: WalkForward must evaluate each top-N candidate on the
test target exactly once.

Pre-fix: the train-best candidate (top[0]) was evaluated twice — once
inside the top-N loop for the fitness vector, then again as
``test_best = self.test_target.evaluate(best_on_train.params)`` to
extract a trade count. Stochastic test targets produced an artifact
where ``test_fitnesses[0]`` and ``test_best_trades`` came from
different evaluations of the same params — internally inconsistent.

The fix caches the EvalResult per candidate and reads test_best from
the cache. WalkForwardResult also exposes ``test_best_fitness`` (from
the same cached eval as ``test_best_trades``) and ``test_best_params``
so downstream consumers can verify alignment.
"""

from __future__ import annotations

import itertools
from typing import Any

from omega_lock.grid import GridPoint
from omega_lock.target import EvalResult, ParamSpec
from omega_lock.walk_forward import WalkForward


class _CountingTestTarget:
    """Test target that returns a different EvalResult on each call.

    Mimics a stochastic backtest: same params, fresh evaluation, fresh
    fitness number. Lets us assert exactly how many times each candidate
    was evaluated.
    """

    def __init__(self) -> None:
        self.call_log: list[dict[str, Any]] = []
        self._counter = itertools.count()

    def param_space(self) -> list[ParamSpec]:
        return []

    def neutral_defaults(self) -> dict[str, Any]:
        return {}

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        idx = next(self._counter)
        self.call_log.append(dict(params))
        # Different fitness on each call so a duplicate eval would be
        # immediately distinguishable in the artifact.
        return EvalResult(fitness=float(idx) + 0.1, n_trials=10 + idx)


def _grid_point(name: str, fitness: float, n_trials: int = 5) -> GridPoint:
    return GridPoint(
        idx=0,
        unlocked={"axis": name},
        params={"axis": name},
        result=EvalResult(fitness=fitness, n_trials=n_trials),
    )


def test_walk_forward_evaluates_each_top_candidate_exactly_once():
    """Pre-fix: train-best (top[0]) was hit twice. Post-fix: once each."""
    train_grid = [
        _grid_point("a", 0.9),
        _grid_point("b", 0.8),
        _grid_point("c", 0.7),
    ]
    target = _CountingTestTarget()
    wf = WalkForward(test_target=target)
    result = wf.run(train_grid=train_grid, top_n=3)

    # 3 top candidates, 3 calls — not 4.
    assert len(target.call_log) == 3
    # Each unique params dict was called exactly once.
    seen = [tuple(sorted(p.items())) for p in target.call_log]
    assert len(set(seen)) == 3


def test_walk_forward_test_best_fitness_matches_cached_top0_eval():
    """test_best_fitness must come from the same evaluation as
    test_fitnesses[0] / test_n_trials[0] / test_best_trades. A
    stochastic target would have produced a different number on a
    second call — so equality here is the contract."""
    train_grid = [
        _grid_point("a", 0.9),
        _grid_point("b", 0.8),
    ]
    target = _CountingTestTarget()
    wf = WalkForward(test_target=target)
    result = wf.run(train_grid=train_grid, top_n=2)

    assert result.test_fitnesses[0] == result.test_best_fitness
    assert result.test_n_trials[0] == result.test_best_trades


def test_walk_forward_test_best_params_match_train_best():
    train_grid = [
        _grid_point("winner", 0.95, n_trials=20),
        _grid_point("runner_up", 0.85, n_trials=15),
        _grid_point("third", 0.7, n_trials=10),
    ]
    target = _CountingTestTarget()
    wf = WalkForward(test_target=target)
    result = wf.run(train_grid=train_grid, top_n=3)

    assert result.test_best_params == {"axis": "winner"}


def test_walk_forward_to_dict_round_trips_new_fields():
    train_grid = [_grid_point("a", 0.5), _grid_point("b", 0.4)]
    target = _CountingTestTarget()
    wf = WalkForward(test_target=target)
    result = wf.run(train_grid=train_grid, top_n=2)
    d = result.to_dict()
    assert "test_best_fitness" in d
    assert "test_best_params" in d
    assert d["test_best_params"] == result.test_best_params
    # Ensure the dict is a copy, not a live reference.
    d["test_best_params"]["mutated"] = True
    assert "mutated" not in result.test_best_params


def test_walk_forward_top_n_smaller_than_grid_only_evals_top_n():
    train_grid = [_grid_point(str(i), 1.0 - i * 0.1) for i in range(5)]
    target = _CountingTestTarget()
    wf = WalkForward(test_target=target)
    result = wf.run(train_grid=train_grid, top_n=2)

    # Only the top 2 train candidates were ever sent to test target.
    assert len(target.call_log) == 2
    assert {p["axis"] for p in target.call_log} == {"0", "1"}
    assert result.top_n == 2
