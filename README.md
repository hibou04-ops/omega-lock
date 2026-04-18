# Omega-Lock

[![PyPI version](https://img.shields.io/pypi/v/omega-lock.svg?v=0.1.2)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg?v=0.1.2)](https://pypi.org/project/omega-lock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sensitivity-driven coordinate descent calibration framework.**

> "Use the keyhole as the mold." Lock every parameter tight. Unlock only those that push back hardest under perturbation. Search the low-dimensional subspace, **then iterate like a fractal vise**: lock the winners, re-measure stress on what remains, narrow the grid geometrically around each winner. Walk-forward and an optional holdout target catch overfitting; an objective RAGAS-style scorecard makes every run comparable.

The framework was distilled from a real calibration experiment that ended in KC-4 FAIL. The methodology detected overfitting exactly as designed, and that detection is what the framework is built to produce. It has since grown beyond the single-round prototype into a multi-scale system (iterative lock-in, zooming grid, optional TPE, holdout defense, objective benchmark).

한국어 README: [README_KR.md](https://github.com/hibou04-ops/omega-lock/blob/main/README_KR.md)

## At a Glance

| | |
|---|---|
| What it solves | High-dim parameter search where most params don't matter, few iterations are available, and the optimizer will overfit if you let it. |
| How it's different | Measure which params matter (stress) before searching. Search a 3-dim subspace, not a 22-dim one. Pre-declared kill criteria prevent threshold fudging. |
| When to use it | You have ≥10 parameters, a costly fitness function, a train/test split, and you want machine-verifiable "this configuration generalizes" rather than a single number. |
| When not to use it | Your optimum lies in effective dimension ≈ nominal dimension, or you have effectively unlimited samples. Use plain TPE / random search. |
| Install | `pip install omega-lock` (core) or `pip install "omega-lock[p2]"` (with Optuna TPE). |
| Core API | `run_p1(target)` · `run_p1_iterative(target)` · `run_p2_tpe(target)` · `run_benchmark(specs, methods)` |
| Status | 0.1.2 · 149 tests passing · benchmark gold baseline frozen for CI regression guard. |

**Concrete numbers from the reference keyhole** (`PhantomKeyhole`, 12 params, 3 effective):

- Plain grid (5 pts/axis, 1 round): `alpha=0.5, fitness=12.00, pass 60%` across 5 seeds.
- Fractal vise (2 rounds × 4 zoom passes): `alpha=0.4375, fitness=13.00` (`+8%`, lands off the coarse lattice).
- Optuna TPE (200 trials): `alpha=0.4037, fitness=14.00` (closest to true optimum) but `pass 10%` (walk-forward catches TPE's finer overfit).

The framework catches what it's supposed to catch, and the numbers are reproducible from a seed.

## Table of Contents

- [Philosophy](#philosophy)
- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [Kill Criteria](#kill-criteria-pre-declared)
- [Module Structure](#module-structure)
- [Search Strategy Comparison](#search-strategy-comparison)
- [vs External Alternatives](#vs-external-alternatives)
- [Holdout Target](#holdout-target)
- [Fractal-vise Mode](#fractal-vise-mode-multi-scale-refinement)
- [Objective Benchmark (RAGAS-style)](#objective-benchmark-ragas-style)
- [Adapter Patterns](#adapter-patterns)
- [Tests](#tests)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Philosophy

Most parameter search suffers from the **curse of dimensionality**. You can hit a 22-dimensional space with random search or TPE, but if samples are scarce and evaluations are expensive, iterations run out and you converge on a Goodhart local optimum.

Omega-Lock makes three assumptions:

- **Effective dimension ≪ nominal dimension.** Most parameters don't meaningfully affect the result.
- Therefore, **measure sensitivity first** and only search the top-K.
- **Kill criteria must be pre-declared.** The experimenter cannot fudge thresholds post-hoc (Winchester prevention).

If these assumptions don't hold, Omega-Lock doesn't work. The original case study confirmed assumptions 1 and 2 but failed KC-4 in walk-forward. Even reduced to 3 dimensions, the underlying signal layer was overfit. The framework flagged it, which is the job.

---

## Pipeline

Two levels: an **inner pipeline** (one round of stress → unlock → search → verify) and an **outer loop** (fractal-vise coordinate descent that runs the inner pipeline repeatedly, locking winners each round).

### Inner pipeline (`run_p1`)

```
target.evaluate(baseline_params)              # baseline (neutrals or prior round's locks)
    ↓
for each unlocked param:                      # stress (KC-2)
    perturb by ±ε, measure |Δfitness|/ε
    ↓
sort stress desc, pick top-K                  # unlock set
    ↓
search over K-dim subspace                    # train fitness
    GridSearch         ─ 1 round × n^K                    (default)
    ZoomingGridSearch  ─ r rounds, range × zoom_factor    (fractal refinement)
    run_p2_tpe         ─ Optuna TPE, fully continuous     (optional)
    ↓
walk-forward on test_target                   # KC-4 (Pearson + trade ratio)
    ↓
[optional] hybrid re-rank with judge target   # slow-but-accurate B over top-K
    ↓
[optional] SC-2 advisory                      # grid top-q vs random top-q (Bergstra-Bengio)
    ↓
KC-1 (time box) + KC-3 (action count floor)
    ↓
[optional] holdout_target evaluated ONCE      # honest out-of-sample
    ↓
P1Result (JSON-serializable)
```

### Outer loop (`run_p1_iterative`)

```
base_params = neutral_defaults
locked = {}
for round r in 0..max_rounds:
    remaining = all_params - locked
    result = run_p1(target, base_params, subset=remaining)
    if result.status != "PASS":  break   # Winchester defense
    if improvement < min_improvement:  break
    lock winners of this round into base_params
    ↓
final_baseline (all locked values) + per-round P1Results + holdout_result
```

Each round's KC-1..4 are enforced independently — thresholds are **never relaxed across rounds** (Winchester prevention). The outer loop halts on the first failed round.

---

## Quick Start

### 1. Install

```bash
# PyPI (recommended)
pip install omega-lock

# With optional Optuna TPE (P2) support
pip install "omega-lock[p2]"

# From source (development)
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

### 2. Run the toy examples

```bash
python examples/rosenbrock_demo.py      # 2D Rosenbrock — grid convergence sanity check
python examples/phantom_demo.py         # 12-param synthetic keyhole — full P1 end-to-end
python examples/full_showcase.py        # 5-mode comprehensive: plain / fractal / random / TPE / deep-iteration
python examples/benchmark_battery.py    # RAGAS-style objective scorecard across methods × keyholes × seeds
python examples/adapter_example.py      # wrap arbitrary external systems as CalibrableTarget
```

- `rosenbrock_demo.py` — 2D static function, no walk-forward / KC-4.
- `phantom_demo.py` — **`PhantomKeyhole`** (12 params: 3 effective + 9 decoy, seed-driven train / test / validation). Exercises stress → top-K unlock → grid → walk-forward → hybrid, with KC-1..4 all PASS. The reference keyhole.
- `full_showcase.py` — every search mode against both reference keyholes, prints results side-by-side.
- `benchmark_battery.py` — runs every method × keyhole × seed combination, prints an objective scorecard (effective_recall, param_L2_error, fitness_gap, generalization_gap, stress_rank_spearman, pass_rate).
- `adapter_example.py` — two patterns for wrapping external systems: `CallableAdapter` (one-liner for pure functions) and a stateful class template.

### 3. Implement your own target

Implement the `CalibrableTarget` protocol:

```python
from omega_lock import CalibrableTarget, EvalResult, ParamSpec, P1Config, run_p1

class MyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="threshold", dtype="float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec(name="window",    dtype="int",   low=10,  high=100, neutral=50),
            ParamSpec(name="use_cache", dtype="bool",  neutral=False),
        ]

    def evaluate(self, params: dict) -> EvalResult:
        # ... your logic here ...
        return EvalResult(
            fitness=score,       # scalar to maximize
            n_trials=n_actions,  # for KC-3
            metadata={"mode": ...},
        )

result = run_p1(train_target=MyTarget())
print(result.status)               # "PASS" or "FAIL:KC-..."
print(result.grid_best["unlocked"])
```

### 4. Walk-forward

For time-series targets, pass separate train / test targets:

```python
result = run_p1(
    train_target=MyTarget(data=train_slice),
    test_target=MyTarget(data=test_slice),
    config=P1Config(trade_ratio_scale=len(test_slice) / len(train_slice)),
)
```

### 5. Hybrid fitness (A+B pattern)

Search cheaply with A, re-validate the top-K with an expensive-but-accurate B:

```python
# A: fast heuristic (e.g. diversity score from history)
class FastTarget:
    def param_space(self): return SHARED_SPECS
    def evaluate(self, params): return EvalResult(fitness=cheap_score(params))

# B: slow judge (e.g. LLM rubric)
class JudgeTarget:
    def param_space(self): return SHARED_SPECS
    def evaluate(self, params): return EvalResult(fitness=gemini_judge(params))

result = run_p1(
    train_target=FastTarget(),
    validation_target=JudgeTarget(),   # B re-evaluates only the top-K
    config=P1Config(walk_forward_top_n=5),
)
# result.hybrid_top[0] is the #1 by B's score
```

### 6. Fractal-vise mode (iterative lock-in + zooming grid)

When `effective_dim > unlock_k`, single-round grid search can only capture K effectives — the rest stay at neutrals. The iterative orchestrator locks each round's winners and re-measures stress on what remains, surfacing the next wave. Zooming narrows the grid geometrically around each winner so the final values aren't stuck on the coarse lattice.

```python
from omega_lock import IterativeConfig, KCThresholds, run_p1_iterative

result = run_p1_iterative(
    train_target=MyTarget(),
    test_target=MyTargetAtDifferentSlice(),
    holdout_target=MyTargetAtThirdSlice(),          # evaluated ONCE at the end, never during rounds
    config=IterativeConfig(
        rounds=3,
        per_round_unlock_k=3,
        zoom_rounds=4,          # geometric refinement inside each round
        zoom_factor=0.5,        # range shrinks by half each zoom pass
        min_improvement=0.5,
        kc_thresholds=KCThresholds(trade_count_min=50),
    ),
)

print(result.final_status)                # "PASS" only if every round passed KC-1..4
print(result.locked_in_order)             # [['alpha', 'long_mode', 'beta'], ['window', 'use_ema', 'horizon'], ...]
print(result.round_best_fitness)          # [32.4, 143.6, 143.61]  — each round's grid_best
print(result.holdout_result)              # {'fitness': 144.41, 'n_trials': ..., 'params': ...}
```

### 7. Optuna TPE (continuous search)

Install with `pip install "omega-lock[p2]"`. TPE replaces the grid with adaptive Bayesian sampling.

```python
from omega_lock import P2Config, run_p2_tpe

result = run_p2_tpe(
    train_target=MyTarget(),
    test_target=MyTargetAtDifferentSlice(),
    config=P2Config(unlock_k=3, n_trials=200, seed=42),
)
# Same KC-1..4 gates as run_p1 — TPE is a search-method swap, not a threshold relaxation.
```

---

## Kill Criteria (pre-declared)

| KC | Checked at | Default threshold | Purpose |
|----|-----------|-------------------|---------|
| KC-1 | end of run | elapsed ≤ 3 days | time box |
| KC-2 | after stress measurement | Gini ≥ 0.2, top/bot ratio ≥ 2.0 | differentiation guaranteed |
| KC-3 | final stage | baseline / train_best / test_best ≥ 50 trades | statistical power |
| KC-4 | after walk-forward | Pearson ≥ 0.3, trade_ratio ≥ 0.5 | overfitting defense |

All thresholds are overridable via the `KCThresholds` dataclass. Toy examples typically relax them (e.g. `trade_count_min=1`).

---

## Module Structure

```
src/omega_lock/
├── target.py         # CalibrableTarget Protocol + ParamSpec + EvalResult
├── params.py         # LockedParams + clip / default_epsilon
├── stress.py         # measure_stress + gini + select_unlock_top_k
├── grid.py           # GridSearch + ZoomingGridSearch + grid_points(_in)
├── random_search.py  # RandomSearch + top_quartile_fitness + compare_to_grid (SC-2)
├── walk_forward.py   # WalkForward + pearson
├── fitness.py        # BaseFitness + HybridFitness
├── kill_criteria.py  # KCThresholds + check_kc1..4 (+ KCStatus "ADVISORY" for SC-2)
├── orchestrator.py   # run_p1 + run_p1_iterative (+ holdout + SC-2 wire-in)
├── p2_tpe.py         # run_p2_tpe — Optuna TPE continuous-space optimizer (optional dep)
├── adapters.py       # CallableAdapter — wrap any callable as a CalibrableTarget
├── benchmark.py      # run_benchmark + BenchmarkReport — RAGAS-style objective scorecard
└── keyholes/
    ├── phantom.py        # PhantomKeyhole — effective_dim 3 / nominal 12 (happy-path demo)
    └── phantom_deep.py   # PhantomKeyholeDeep — effective_dim 6 / nominal 20 (iteration required)
```

## Search Strategy Comparison

| Method | Continuity | Resolution | Use case |
|---|---|---|---|
| `GridSearch` | discrete | 1 round × $n^K$ | fast first pass |
| `ZoomingGridSearch` | discrete (geometric shrink) | $n^K \times r$ rounds | refine beyond grid lattice |
| `RandomSearch` | mixed discrete / continuous | same-budget random sampling | SC-2 baseline (grid top-q ≥ 1.5× random) |
| `run_p2_tpe` (Optuna) | fully continuous | TPE adaptive | true continuous-space optimizer, optional `pip install "omega-lock[p2]"` |

## vs External Alternatives

| Tool | Approach | Omega-Lock's difference |
|---|---|---|
| Optuna / Hyperopt (TPE) | Bayesian adaptive sampling, full-dim | Omega-Lock fixes a top-K subspace via stress *before* sampling. When `effective_dim ≪ nominal_dim` holds, sample efficiency dominates. Complementary, wrap TPE via `run_p2_tpe`. |
| Ray Tune / scikit-optimize | general-purpose HPO frameworks | single fitness, no built-in walk-forward / overfit gate. Omega-Lock makes KC-4 (Pearson + trade_ratio) a required gate. |
| Plain grid search | exhaustive | high-dim explosion ($n^D$). Omega-Lock reduces to $n^K$ via stress → top-K unlock. |
| Nelder-Mead / Powell | local continuous search | continuous-only, no categoricals or bools. Omega-Lock handles mixed int / bool / continuous. |

**Omega-Lock's USP**: *pre-declared kill criteria + low-dim subspace hypothesis.* Not another adaptive-sampling optimizer, a **methodology framework**. Ideally layered on top of existing optimizers (TPE / Bayesian / Genetic); `run_p2_tpe` is the reference example.

## Holdout Target

Pass a third target that is *never touched during rounds* via `run_p1(..., holdout_target=T3)` or `run_p1_iterative(..., holdout_target=T3)`. The final `grid_best` or `final_baseline` is evaluated on it exactly once, and the result is recorded in `holdout_result`. This is an honest auxiliary check, in iterative mode the test_set gets reused for lock-in decisions round after round, which weakens KC-4 evidence.

## Fractal-vise Mode (multi-scale refinement)

Think of a fractal vise: a large segment clamps the object first (round 1 lock-in), then smaller segments conform within that coordinate system (zooming within a round, or the next round on remaining params).

Two independent axes:

1. **Iterative lock-in** (`run_p1_iterative` + `IterativeConfig`):
   After round 1 unlocks top-K and locks the grid-best, round 2 re-measures stress on the remaining params, and so on. Valuable when `effective_dim > unlock_k`.

2. **Zooming grid** (`ZoomingGridSearch`, or `P1Config(zoom_rounds=N)`):
   Within a single round, the grid shrinks geometrically around the previous winner. Reaches finer values (e.g. `alpha=0.4375`) that the initial discrete grid (e.g. `alpha=0.5`) cannot. Roughly 4× error reduction every two zoom rounds.

The two axes compose: `run_p1_iterative(config=IterativeConfig(rounds=3, zoom_rounds=4))` is the full fractal vise. On `PhantomKeyhole`, plain grid (`alpha=0.5`, fitness=12.0) vs. fractal (`alpha=0.4375`, fitness=13.0) makes the contrast visible.

**Warning**: KC thresholds are strictly enforced every round, Winchester prevention. Because `test_set` is reused across rounds, `KC-4` PASS becomes weaker evidence as rounds deepen. In practice, splitting out a hold-out set is recommended.

---

## Objective Benchmark (RAGAS-style)

"Does it pass?" (binary KC gate) is necessary but not sufficient. For comparing methods or detecting silent regressions, Omega-Lock provides a mechanical scorecard where every metric is computed from run outputs + keyhole ground truth (no human judgment).

| Metric | Definition | Want |
|---|---|---|
| `effective_recall` | \|found ∩ true_effective\| / \|true_effective\| | → 1.0 |
| `effective_precision` | \|found ∩ true_effective\| / \|found\| | → 1.0 |
| `param_L2_error` | Normalized L2 of found params vs true optimum | → 0.0 |
| `fitness_gap_pct` | `(optimum − found) / |optimum|` | ≤ 0 (found beats reference) |
| `generalization_gap` | `|train_best − test_best| / |train_best|` | small |
| `stress_rank_spearman` | ρ(measured stress ranking, true importance ranking) | → 1.0 |
| `pass_rate` | fraction of runs with `status == "PASS"` | — |
| `walltime_s` / `n_evaluations` | efficiency | — |

```python
from omega_lock import BenchmarkSpec, CalibrationMethod, run_benchmark
from omega_lock.keyholes.phantom import PhantomKeyhole

spec = BenchmarkSpec("PhantomKeyhole", PhantomKeyhole, seeds=[42, 7, 100, 314, 55])
methods = [
    CalibrationMethod("plain_grid",   runner=lambda t, s: _wrap_p1(run_p1(t, ...))),
    CalibrationMethod("fractal_vise", runner=lambda t, s: _wrap_iter(run_p1_iterative(t, ...))),
]

report = run_benchmark([spec], methods, output_path=Path("bench.json"))
print(report.render_scorecard())
```

Sample output (combined over 10 runs):

```
method              recall  prec   L2err  fit_gap%  gen_gap  pass%
plain_grid          0.750   1.000  1.052  32.3%     0.958    60.0%
fractal_vise        0.400   0.217  1.003  14.7%     0.820    40.0%
optuna_tpe          0.750   1.000  0.970  23.9%     0.858    10.0%
```

Reading the table: TPE has the tightest `L2err` (closest to true optimum) but also the lowest `pass_rate`. That's the framework catching TPE's finer-grained overfitting. `plain_grid` passes most often because it's coarser and harder to overfit with. `fractal_vise` trades precision for broader coverage across rounds.

**CI regression guard**: `tests/test_benchmark_regression.py` compares the current run against a frozen `tests/fixtures/benchmark_gold.json`. Any drift > `1e-6` on deterministic metrics fails the test. Regenerate intentionally via `OMEGA_LOCK_UPDATE_GOLD=1 pytest tests/test_benchmark_regression.py`.

---

## Adapter Patterns

Wrap arbitrary external systems as `CalibrableTarget`. Two idiomatic patterns, both in `examples/adapter_example.py`.

### Pattern 1: `CallableAdapter` (one-liner for pure functions)

```python
from omega_lock import CallableAdapter, ParamSpec, run_p1

def external_score(params: dict) -> float:
    return -((params["a"] - 3.0) ** 2 + (params["b"] - 7.0) ** 2)

target = CallableAdapter(
    fitness_fn=external_score,
    specs=[
        ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
    ],
)

result = run_p1(train_target=target, config=P1Config(unlock_k=2, zoom_rounds=4))
```

### Pattern 2: Stateful class (for systems with setup cost)

Implement `param_space()` + `evaluate()` directly when your target has internal state (trained models, pre-loaded data, active sessions). The template in `examples/adapter_example.py` shows the full shape.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/                    # all
pytest tests/test_stress.py -v   # single module
pytest --cov=omega_lock          # coverage
```

---

## Limitations

- **Determinism assumption.** Stress measurement is accurate only when the target is deterministic. For non-deterministic targets, fix the seed or average multiple evaluations.
- **Suppressed-stress flag.** If a parameter's stress is known to be artificially low due to an environmental constraint (e.g. an upstream subsystem was mocked or disabled during measurement), mark it with `ParamSpec(ofi_biased=True)`. The flag appears in the result for observability, but nothing is auto-filtered.
- **Continuous + int mixed.** Epsilon is type-aware (continuous = 10% of range, int = 1, bool = flip). Override via `StressOptions(epsilons={...})`.
- **Grid dimension explosion.** K=3 / 5 points-per-axis = 125 combos. For larger K, adaptive search like Optuna TPE is better (currently outside P2 TPE's scope; future enhancement).

---

## Roadmap

### Shipped in current version

- ✅ **Iterative coordinate descent** — `run_p1_iterative`, multi-round lock-in.
- ✅ **Zooming grid** — `ZoomingGridSearch`, geometric refinement inside a round.
- ✅ **Optuna TPE (P2)** — `run_p2_tpe`, continuous-space search as opt-in (`pip install "omega-lock[p2]"`).
- ✅ **Random-search baseline** — `RandomSearch` + `compare_to_grid`, SC-2 advisory gate in `run_p1`.
- ✅ **Holdout target** — single-shot out-of-sample evaluation, never touched during rounds.
- ✅ **Objective benchmark** — `run_benchmark` + `BenchmarkReport`, RAGAS-style scorecard + CI regression guard.
- ✅ **Adapter patterns** — `CallableAdapter` + stateful-class template.

### Still out of scope (application-specific)

- **Domain-specific adapters** — wrapping a particular external system (trading strategy, ML model, simulation) as a `CalibrableTarget` belongs outside this generic library. See `CallableAdapter` and the stateful-class template in `examples/adapter_example.py` for the general pattern.
- **Ensemble-averaged `evaluate` helper** — for non-deterministic targets; the `CalibrableTarget` docstring says "report ensemble averages" but no helper ships. Add when a real use case appears.

---

## Citation

If you use Omega-Lock in research or a published project, please cite:

```bibtex
@software{omega_lock_2026,
  author  = {hibou},
  title   = {Omega-Lock: Sensitivity-driven coordinate descent calibration framework},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

---

## License

MIT License. See [LICENSE](https://github.com/hibou04-ops/omega-lock/blob/main/LICENSE) for details.

Copyright (c) 2026 hibou.
