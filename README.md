# Omega-Lock

[![PyPI version](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sensitivity-driven coordinate descent calibration framework.**

> "Use the keyhole as the mold." Lock every parameter tight. Unlock only those that push back hardest under perturbation. Grid-search the low-dimensional subspace, then validate with walk-forward to catch overfitting.

This package generalizes the methodology from `Omega_TB_1/research/omega_lock_p1/` (a v1 HeartCore target that ended in KC-4 FAIL) into a reusable library for arbitrary parameter-search problems. The original HeartCore experiment was not a "success" in the naive sense, it was a successful *overfitting detection*, which is exactly what this framework is designed to produce (see `archive/`).

한국어 README: **[README_KR.md](README_KR.md)**

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
- [Tests](#tests)
- [Limitations](#limitations)
- [Roadmap](#roadmap-out-of-scope-for-this-package)
- [Citation](#citation)
- [License](#license)

---

## Philosophy

Most parameter search suffers from the **curse of dimensionality**. You can hit a 22-dimensional space with random search or TPE, but if samples are scarce and evaluations are expensive, iterations run out and you converge on a Goodhart local optimum.

Omega-Lock makes three assumptions:

- **Effective dimension ≪ nominal dimension.** Most parameters don't meaningfully affect the result.
- Therefore, **measure sensitivity first** and only search the top-K.
- **Kill criteria must be pre-declared.** The experimenter cannot fudge thresholds post-hoc (Winchester prevention).

If these assumptions don't hold, Omega-Lock doesn't work. P1 HeartCore confirmed assumptions 1 and 2 but failed KC-4 in walk-forward, meaning even reduced to 3 dimensions, v1's signal layer was fundamentally overfit. That outcome was itself useful information.

---

## Pipeline

```
target.evaluate(neutral_defaults)        # baseline
    ↓
for each param:                          # stress measurement (KC-2)
    perturb by ±ε, measure |Δfitness|/ε
    ↓
sort stress desc, pick top-K             # unlock set
    ↓
grid search over K-dim subspace          # train fitness
    ↓
walk-forward: top-N on test target       # KC-4 (Pearson + trade ratio)
    ↓
[optional] hybrid validation: top-K with slower judge target
    ↓
KC-1 (time box) + KC-3 (action count floor)
    ↓
P1Result (JSON-serializable)
```

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
```

- `rosenbrock_demo.py` — 2D static function, no walk-forward / KC-4.
- `phantom_demo.py` — **`PhantomKeyhole`** (12 params: 3 effective + 9 decoy, seed-driven train / test / validation). Exercises stress → top-K unlock → grid → walk-forward → hybrid, with KC-1..4 all PASS. The reference keyhole for the framework.

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
├── kill_criteria.py  # KCThresholds + check_kc1..4
├── orchestrator.py   # run_p1() + run_p1_iterative() (+ holdout support)
├── p2_tpe.py         # run_p2_tpe() — Optuna TPE continuous-space optimizer (optional dep)
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
- **OFI-biased parameters.** If a parameter's stress is artificially low due to environmental constraints, mark it with `ParamSpec(ofi_biased=True)`. It gets flagged in results but not auto-filtered (observational only).
- **Continuous + int mixed.** Epsilon is type-aware (continuous = 10% of range, int = 1, bool = flip). Override via `StressOptions(epsilons={...})`.
- **Grid dimension explosion.** K=3 / 5 points-per-axis = 125 combos. For larger K, adaptive search like Optuna TPE is better (currently outside P2 TPE's scope; future enhancement).

---

## Roadmap (out of scope for this package)

- **Omega_X adapter** — `adapters/omega_x/` implementing `SelectorTarget`, `ValidationTarget` for X thread pipeline calibration.
- **P2 Optuna TPE** — `orchestrator.run_p2()`, adaptive search instead of grid.
- **P3 enrichment** — faithful OFI reconstruction from bookDepth / aggTrades (HeartCore-specific).
- **Random-search baseline** — actually compare SC-2 "top-quartile ≥ 1.5× random" (missed in P1).

---

## Citation

If you use Omega-Lock in research or a published project, please cite:

```bibtex
@software{omega_lock_2026,
  author  = {hibou},
  title   = {Omega-Lock: Sensitivity-driven coordinate descent calibration framework},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

---

## Archive (private, not in public repo)

The methodological origin, **Omega-Lock P1 HeartCore** applied case (2026-04-13 to 04-14), lives in a separate local `archive/` directory (gitignored).

- `P1_HeartCore_SPEC.md` — original design document for the 21-param v1 HeartCore target.
- `P1_HeartCore_RESULT.md` — KC-4 FAIL report (Pearson 0.119, successful train/test overfit detection).

Both documents are **immutable**, preserved as the **first recorded case** of the methodology detecting overfitting as intended. Not publicly released (Omega_TB_1 internal research + BTCUSDT real-data references).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 hibou.
