# Omega-Lock

[![PyPI version](https://img.shields.io/pypi/v/omega-lock.svg?v=0.1.4)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg?v=0.1.4)](https://pypi.org/project/omega-lock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-176%20passing-brightgreen.svg)](tests/)
[![Methodology](https://img.shields.io/badge/methodology-Antemortem-blueviolet.svg)](https://github.com/hibou04-ops/Antemortem)

> **A method-agnostic audit surface for calibration — plus the sensitivity-driven search framework it grew out of.**
>
> Bring your own optimizer. Omega-Lock decides whether the tuned candidate is reviewable, constraint-feasible, and likely to generalize — not whether your optimizer "found something."

### What the name means

`omega-lock` is a **calibration audit discipline**, not security or DRM software. *Lock* refers to locking a candidate behind audit gates — hard constraints, stability checks, and out-of-sample generalization — so a tuned result never ships until it clears the same mechanical review every time.

### Built for

- **Quant / strategy tuning** — filter candidates that look great in-sample but collapse under walk-forward, with KC-4 (Pearson + trade-ratio) as the gate.
- **Hardware / simulation calibration** — PVT sweeps, process control, materials discovery: costly surrogate or SPICE-like evaluation with hard physical constraints (see `examples/demo_sram.py` — 6T SRAM bitcell across 5 PVT corners).
- **ML / HPO governance** — turn an optimizer's "best trial" into a deployment-safe artifact with an append-only trail, not a lone fitness number.

### Headline feature (new in 0.1.4)

`omega_lock.audit` is the hero surface. Wrap any `CalibrableTarget` with `AuditingTarget`, hand it to any optimizer (grid, TPE, random, Bayesian, your own), and get an append-only trail with phase / role / round context, declarative hard constraints, a feasible-vs-absolute best split, and a JSON-serializable reviewable artifact.

What 0.1.4 ships:

- **`omega_lock.audit`** — new submodule: `AuditingTarget`, `Constraint`, `AuditReport`, `make_report`, `render_scorecard`. Protocol-based, so no optimizer changes required. See [Audit Module](#audit-module-new-in-014) below.
- **`examples/demo_sram.py`** — 6T SRAM bitcell analytical surrogate across 5 PVT corners with 3 hard constraints, demonstrating the audit scorecard on a realistic-shaped target.
- **The original framework** — three integrated search pipelines (`run_p1`, `run_p1_iterative`, `run_p2_tpe`), perturbation sensitivity, walk-forward, kill criteria, RAGAS-style benchmark. All unchanged from 0.1.3 except the audit wrapper now works with every pipeline natively.
- **176 tests passing** (149 from 0.1.3 + 20 new audit tests + 7 new SRAM demo tests). Benchmark gold baseline unchanged.

Origin: extracted from a trading-strategy calibration experiment that ended in KC-4 FAIL, overfitting detected exactly as designed. That controlled-failure outcome is the behaviour the framework is built to produce.

한국어 README: [README_KR.md](https://github.com/hibou04-ops/omega-lock/blob/main/README_KR.md)

## At a Glance

| | |
|---|---|
| What it is | An audit module (`omega_lock.audit`) that works over any `CalibrableTarget`, shipped alongside the sensitivity-driven calibration framework it was built on. |
| Why it matters | The audit runs the same way regardless of which pipeline produced the candidate. That separates "found something" from "it generalizes — and meets the constraints." |
| When to use it | Costly fitness function, a train/test (and ideally holdout) split, and you want a mechanical pass/fail on generalization plus hard constraints, not a single fitness number. |
| When not to use it | Effective dim ≈ nominal dim, samples effectively unlimited, out-of-sample stability not a concern. Use a stock optimizer instead. |
| Install | `pip install omega-lock` (core) or `pip install "omega-lock[p2]"` (Optuna TPE included) |
| Hero API | `from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard` |
| Core API | `run_p1` · `run_p1_iterative` · `run_p2_tpe` · `run_benchmark` · `CallableAdapter` |
| Status | 0.1.4 on PyPI · 176 tests passing · 30-run benchmark gold baseline frozen for CI regression guard |
| Built | 2026-04-18 (audit module) · 2026-04-20 (SRAM demo + 0.1.4 release) |

### Raw benchmark scorecard (30 runs: 2 keyholes × 3 methods × 5 seeds)

This is the output of `examples/benchmark_battery.py` against the shipped reference keyholes. No cherry-picking, no single-seed dramatics.

```
keyhole                method          recall  L2err  fit_gap%  gen_gap  pass%
PhantomKeyhole         plain_grid      1.00    0.24    -9.3     1.26     60%
PhantomKeyhole         fractal_vise    0.60    0.50   -16.6     1.13     60%
PhantomKeyhole         optuna_tpe      1.00    0.07   -22.1     1.10      0%
PhantomKeyholeDeep     plain_grid      0.50    1.86   +73.9     0.66     60%
PhantomKeyholeDeep     fractal_vise    0.20    1.51   +45.9     0.51     20%
PhantomKeyholeDeep     optuna_tpe      0.50    1.87   +70.0     0.61     20%
```

What the numbers actually show:

- **No single search method dominates.** Plain grid has the highest pass rate, TPE has the tightest optimum (lowest L2err) on the easy keyhole, fractal-vise (iterative lock-in) is not a strict improvement over single-round grid. That is a legitimate finding, not a bug.
- **Stress ranking is reliable across methods.** Spearman ρ(measured stress, true importance) ≈ **0.95** across all 30 runs. That is the part of the old lock-by-weight idea that still earns its keep: it is a cheap, accurate screening step.
- **The audit catches what the searcher misses.** Optuna TPE lands closest to the true optimum on `PhantomKeyhole` but has **pass_rate = 0%**: walk-forward correctly flags the finer-grained overfit. That separation between "found something" and "it generalizes" is what the framework is for.

The framework ships three integrated search pipelines. Each reuses the same audit components (stress / walk-forward / KCs / holdout / benchmark). The benchmark above compares them on identical keyholes under identical gates.

## Table of Contents

- [Audit Module (new in 0.1.4)](#audit-module-new-in-014)
- [Philosophy](#philosophy)
- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [Release History](#release-history)
- [Origin](#origin)
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

## Audit Module (new in 0.1.4)

Every calibration run should produce a reviewable artifact. `omega_lock.audit` is the minimal surface that makes that possible for any optimizer conforming to the `CalibrableTarget` protocol.

### 30-second Quick Start

```python
from omega_lock import run_p1, P1Config
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

constraints = [
    Constraint("read_margin_ok",
               lambda p, r: r.metadata["read_snm_mv_worst"] > 150.0,
               "Worst-corner read SNM must exceed 150 mV"),
    Constraint("leakage_ok",
               lambda p, r: r.metadata["leakage_na_worst"] < 5.0,
               "Worst-corner leakage must stay below 5 nA"),
]

wrapped = AuditingTarget(bitcell_target, constraints=constraints)
result  = run_p1(train_target=wrapped, config=P1Config())
report  = make_report(wrapped, method="run_p1", seed=42)

print(render_scorecard(report))
open("audit.json", "w").write(report.to_json())
```

### What it gives you

- **Append-only trail.** Every `evaluate()` call becomes one `AuditedRun`. Append-only means no post-hoc rewrites — the trail is the source of truth.
- **Positional context per call.** `phase` (baseline / stress / search / walk_forward / holdout), `target_role` (train / test / validation / holdout), `round_index` (for coordinate-descent runs), `call_index` (monotonic).
- **Constraints as first-class.** Declare hard predicates once; every call records pass/fail. The report distinguishes `best_feasible` from `best_any` — the separation that matters in real-world deployment.
- **Multi-target, one trail.** `run_p1` juggles train + test + holdout targets. Wrap each with `AuditingTarget` sharing `shared_trail` and `shared_counter`; the trail stays globally ordered.
- **Method-agnostic by construction.** Because `AuditingTarget` implements the `CalibrableTarget` protocol, every optimizer in this repo works unchanged — grid, zooming grid, random, TPE. External optimizers wrapped via `CallableAdapter` work the same way.
- **JSON roundtrip.** `report.to_json()` / `AuditReport.from_json(s)` — reports are versionable, diffable, archivable.

### When to use it

Any setting where "was this calibration run valid?" needs a mechanical answer. Typical: chip-design PVT sweeps, process control, materials discovery, any multi-constraint expensive-evaluation problem. See `examples/demo_sram.py` for a worked 6T SRAM bitcell demo across 5 PVT corners with 3 hard constraints.

### When it's overkill

If you're running a one-shot toy optimization and nobody else is going to look at the trail, skip it. Audit is for the case where the run itself ends up as a decision artifact someone downstream has to trust.

### Methodology behind the build

The `omega_lock.audit` module was built with a pre-implementation reconnaissance discipline I call [**Antemortem**](https://github.com/hibou04-ops/Antemortem) — an AI-assisted protocol for stress-testing a change on paper before writing code. The discipline emerged during `omega_lock.audit`'s own development. Applied to this module, Antemortem caught one ghost trap, downgraded three risks, and surfaced one new spec requirement — before a line was written.

---

## Philosophy

The framework separates two concerns that most optimization tools conflate.

**Search** is how you propose candidates. Grid, zoom, random, Bayesian, gradient-based, a custom heuristic, whatever. Every method has a region where it does well and a region where it fails. There is no universal best.

**Audit** is how you decide whether a proposed candidate actually generalizes. This has nothing to do with how the candidate was produced. It has everything to do with whether its train fitness predicts its test fitness, whether the optimum is stable under perturbation, whether it clears a pre-declared bar on action count and time, whether it still looks good on data the searcher never saw.

Omega-Lock is an audit-first framework. It ships multiple search methods, but the value proposition is that **the audit is the same for all of them**. If you bring your own optimizer (via `CallableAdapter`), it gets the same audit.

Three assumptions the framework still leans on:

- **Effective dim ≪ nominal dim is common.** When it holds, stress measurement is a cheap screening step that shrinks the search region before the expensive part.
- **Pre-declared kill criteria are non-negotiable.** Thresholds cannot be fudged post-hoc. This is the structural defense against the common failure mode where a founder tunes the test set, declares victory, and ships an overfit.
- **No method is immune to overfitting.** The nicer your optimizer, the more skill it has at finding plausible-looking false peaks. This is why the audit layer is method-agnostic by design.

If all three hold, the framework earns its keep. If effective dim ≈ nominal dim or samples are effectively unlimited, a stock optimizer is fine and this framework is overkill.

---

## Pipeline

Two axes, independent.

### Axis 1 — Search (swappable)

Pick one. Or bring your own via `CallableAdapter`. They all return the same downstream shape, so the audit does not care which one you chose.

| Method | Module | When it fits |
|---|---|---|
| `GridSearch` | `grid.py` | Low-dim, want exhaustive, easy to debug |
| `ZoomingGridSearch` | `grid.py` | Refine around a winner to below the initial lattice |
| `RandomSearch` | `random_search.py` | SC-2 baseline, or when you suspect grid coverage is wasted |
| `run_p2_tpe` | `p2_tpe.py` | Continuous Bayesian, non-separable objectives (opt-in Optuna dep) |
| any callable | `adapters.py` | Your existing optimizer. The framework wraps it, not replaces it. |

### Axis 2 — Audit (invariant)

This runs for every method. Same gates, same thresholds, same scorecard.

```
baseline evaluation on neutrals
    ↓
stress measurement                        # KC-2: Gini + top/bot ratio
    (optional: a cheap screening to pick a smaller search region)
    ↓
[ Search runs here, whichever method you chose ]
    ↓
walk-forward re-evaluation on test target  # KC-4: Pearson + trade_ratio
    ↓
[optional] hybrid re-rank with judge target
    ↓
[optional] SC-2 advisory                    # grid top-q vs random top-q
    ↓
KC-1 time box + KC-3 action-count floor
    ↓
[optional] holdout_target evaluated ONCE   # honest out-of-sample, never touched by search
    ↓
Result (JSON-serializable) + status PASS or FAIL:KC-N
```

### High-level orchestrators

- **`run_p1`** — one pass through the axis 2 audit with axis 1 set to `GridSearch` (or `ZoomingGridSearch` if `zoom_rounds > 1`).
- **`run_p1_iterative`** — runs `run_p1` in a loop. Each round locks the grid winners, then re-measures stress on what remains, then searches again. Same KCs per round, not relaxed across rounds (Winchester defense). This is still inside the lock-by-weight frame; useful when effective dim > unlock_k and the landscape is approximately additive, less useful when parameters interact.
- **`run_p2_tpe`** — axis 2 audit with axis 1 set to Optuna TPE. Drops the lock-by-weight commitment: TPE samples the unlocked subspace adaptively without ranking params.
- **`run_benchmark`** — run multiple (search method × keyhole × seed) combinations, emit the objective scorecard shown in the At a Glance section.

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

## Release History

**0.1.3** (2026-04-18) — initial public release. Three integrated search pipelines (`run_p1`, `run_p1_iterative`, `run_p2_tpe`), perturbation sensitivity, walk-forward, KC-1..4, holdout support, SC-2 advisory, `run_benchmark` + 30-run gold baseline regression guard. `CallableAdapter` for wrapping external optimizers. Two reference keyholes (`PhantomKeyhole`, `PhantomKeyholeDeep`) with ground-truth methods. 149 tests, PyPI, MIT.

**0.1.4** (2026-04-20) — **audit surface as the headline.** New `omega_lock.audit` submodule: `AuditingTarget`, `Constraint`, `AuditReport`, `make_report`, `render_scorecard`. Protocol-based, so no optimizer changes required — wrap any `CalibrableTarget` and hand it to grid / TPE / random / Bayesian / your own optimizer. Ships alongside `examples/demo_sram.py` — a 6T SRAM bitcell analytical surrogate across 5 PVT corners (TT / SS / FF / FS / SF) with 3 hard constraints, demonstrating the audit scorecard on a realistic-shaped target. Overfit pathology is physics-informed: a candidate optimized for the typical corner systematically breaks fast/slow corners under the transistor strength ratio. Same pattern kills trading-strategy calibrations and silicon tape-outs. 176 tests (149 + 20 audit + 7 SRAM demo). Benchmark gold baseline unchanged.

## Origin

`omega-lock`'s origin is a calibration experiment in one domain (trading strategies) that failed its own overfitting check. The 0.1.4 SRAM bitcell demo shows the same pathology catching a bitcell sized for typical-process silicon that dies in slow-slow corner. The audit surface is domain-agnostic by design: any candidate from any source, verified through the same mechanical checks.

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

| Tool | What it does | What Omega-Lock adds |
|---|---|---|
| Optuna / Hyperopt (TPE) | Bayesian adaptive sampling over a full-dim space | Audit layer around the result. Stress-based subspace reduction is an optional pre-filter, not required. You can run TPE directly via `run_p2_tpe` and still get the same KC gates + walk-forward + holdout + scorecard. |
| Ray Tune / scikit-optimize | Generic HPO frameworks (many searchers, many schedulers) | A standard audit surface with declared kill criteria, not a single fitness score. KC-4 (Pearson + trade_ratio) and a holdout check are opinionated defaults. |
| Plain grid search | Exhaustive Cartesian | Same grid when you want it, plus a zooming variant for sub-lattice precision, plus an automatic random-sample baseline (SC-2 advisory) that flags cases where your grid coverage was wasted. |
| Nelder-Mead / Powell | local continuous search | continuous-only, no categoricals or bools. Omega-Lock handles mixed int / bool / continuous. |

**Omega-Lock's USP**: *pre-declared kill criteria + low-dim subspace hypothesis.* Not another adaptive-sampling optimizer, a **methodology framework**. Ideally layered on top of existing optimizers (TPE / Bayesian / Genetic); `run_p2_tpe` is the reference example.

## Holdout Target

Pass a third target that is *never touched during rounds* via `run_p1(..., holdout_target=T3)` or `run_p1_iterative(..., holdout_target=T3)`. The final `grid_best` or `final_baseline` is evaluated on it exactly once, and the result is recorded in `holdout_result`. This is an honest auxiliary check, in iterative mode the test_set gets reused for lock-in decisions round after round, which weakens KC-4 evidence.

## Fractal-vise Mode (multi-scale refinement)

Two independent refinement axes. Both sit inside the same audit envelope.

1. **Iterative lock-in** (`run_p1_iterative` + `IterativeConfig`):
   After round 1 unlocks top-K and locks the grid-best, round 2 re-measures stress on the remaining params, and so on. Useful when `effective_dim > unlock_k` AND parameters are approximately additive. Still inside the lock-by-weight frame. Per the benchmark, this is not a strict win over a single wider round, so use it when you have reason to believe the landscape separates.

2. **Zooming grid** (`ZoomingGridSearch`, or `P1Config(zoom_rounds=N)`):
   Within a single round, the grid shrinks geometrically around the previous winner. Reaches values that the initial discrete lattice cannot express. Roughly 4× error reduction per two zoom rounds on smooth landscapes. This is geometric, not weight-based, so it composes with any search method.

The two axes compose: `run_p1_iterative(config=IterativeConfig(rounds=3, zoom_rounds=4))`. On a single seed of `PhantomKeyhole`, this moves `alpha` from `0.5` (on the 5-point grid) to `0.4375` (between lattice points) with fitness 12 → 13. Across 5 seeds the picture is more mixed, see the raw scorecard in [At a Glance](#at-a-glance).

**KC thresholds are enforced every round and never relaxed across rounds** — this is the Winchester defense. Because `test_target` is consulted repeatedly for lock-in decisions, `KC-4 PASS` becomes weaker evidence as rounds accumulate. Pair iterative runs with a `holdout_target` when you care about the final answer.

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

See the [At a Glance](#at-a-glance) section for the per-keyhole breakdown. Short version: no search method wins on every metric, the audit (KC gates + walk-forward) is what makes the scorecard comparable, and the stress-rank Spearman stays around 0.95 across all 30 runs (stress measurement is reliable even where the search methods disagree).

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
  version = {0.1.4},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

---

## License

MIT License. See [LICENSE](https://github.com/hibou04-ops/omega-lock/blob/main/LICENSE) for details.

Copyright (c) 2026 hibou.
