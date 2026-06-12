# omega-lock Power API

[← Back to README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Trust & audit model](TRUST_MODEL.md)

The README uses plain language. The public Python API keeps its established symbol
names for backward compatibility — other repositories import them, so they are
**frozen** and will not be renamed. This page maps the plain-English concepts to
those exact symbols and lists their real signatures.

All symbols below are importable from the package root unless noted:

```python
from omega_lock import (
    run_p1, P1Config, P1Result,
    measure_stress,
    check_kc4, KCThresholds,
    audit_optuna_study, StudyAuditReport,
    render_html,
)
from omega_lock.simple import gate_scores, GateVerdict, audit
```

---

## Plain name → frozen symbol

| Plain name in the README | Frozen API symbol | Where |
|---|---|---|
| run the gate | `run_p1` | `omega_lock` |
| the gate's config | `P1Config` | `omega_lock` |
| the gate's result | `P1Result` | `omega_lock` |
| walk-forward transfer check | `check_kc4` | `omega_lock` |
| transfer-check pass thresholds | `KCThresholds` | `omega_lock` |
| rank parameters by sensitivity | `measure_stress` | `omega_lock` |
| gate an Optuna study | `audit_optuna_study` | `omega_lock` |
| Optuna study gate result | `StudyAuditReport` | `omega_lock` |
| HTML scorecard | `render_html` | `omega_lock` |
| two-list quick gate | `gate_scores` | `omega_lock.simple` |
| two-list gate verdict | `GateVerdict` | `omega_lock.simple` |
| audit a scoring function | `audit` | `omega_lock.simple` |
| a tunable parameter's range | `ParamSpec` | `omega_lock` |
| one scored candidate | `EvalResult` | `omega_lock` |

---

## The simple facade (`omega_lock.simple`)

The lowest-friction entry points. New names, not renames.

### `gate_scores` — gate two score lists

```python
def gate_scores(
    train_scores: Iterable[float],
    holdout_scores: Iterable[float],
    *,
    thresholds: KCThresholds | None = None,
) -> GateVerdict
```

Pass the per-candidate scores measured on the data the search consumed
(`train_scores`) and the same candidates re-scored on data it never saw
(`holdout_scores`), **index-aligned**. It applies the walk-forward Pearson transfer
gate (`check_kc4`). `thresholds` defaults to `KCThresholds.pure_objective()` because
plain score lists carry no action counts.

> Note: `train_scores` / `holdout_scores` are **sequences of numbers**, not file
> paths. If you have JSON files, load them first (`json.load(...)`) or use the
> `omega-lock gate --train a.json --holdout b.json` CLI, which reads the files for
> you.

`GateVerdict` fields:

```python
@dataclass(frozen=True)
class GateVerdict:
    passed: bool                       # True when the ranking transferred
    pearson: float | None             # measured correlation, or None if not computable
    reasons: tuple[str, ...]          # human-readable failure reasons; empty when passed
    kc_report: KCReport               # the underlying KC-4 report (full detail dict)
    train_scores: tuple[float, ...]   # the gated inputs (for render_html)
    holdout_scores: tuple[float, ...]
    def to_dict(self) -> dict[str, Any]: ...
```

> The README's marketing snippet shows `result.reason`; the real attribute is the
> plural `result.reasons` (a tuple). Use `result.passed` for the boolean decision.

### `audit` — audit a scoring function over a parameter space

```python
def audit(
    target_fn: Callable[[dict[str, Any]], float],
    param_specs: Sequence[ParamSpec] | Mapping[str, Sequence[float]],
    *,
    holdout_fn: Callable[[dict[str, Any]], float] | None = None,
    output_path: str | Path | None = None,
    **cfg: Any,
) -> P1Result
```

A thin `CallableAdapter` + `run_p1` wrapper. `param_specs` accepts a `ParamSpec`
list or a friendly `{name: (low, high)}` / `{name: (low, high, neutral)}` mapping.
`holdout_fn`, when provided, becomes the walk-forward test target. `**cfg` is
forwarded to `P1Config`. Importable as `omega_lock.simple.audit` only (the root
name `omega_lock.audit` is the audit-trail subpackage).

---

## The full pipeline

### `run_p1` — run the gate

```python
def run_p1(
    train_target: CalibrableTarget,
    config: P1Config | None = None,
    test_target: CalibrableTarget | None = None,
    validation_target: CalibrableTarget | None = None,
    output_path: Path | None = None,
    base_params: dict[str, Any] | None = None,
    stress_subset: list[str] | None = None,
    holdout_target: CalibrableTarget | None = None,
) -> P1Result
```

`train_target` is used for baseline + stress + grid search. When `test_target` is
provided, the walk-forward transfer gate (`check_kc4`) runs. Returns a `P1Result`
artifact (the same object `render_html` and the CLI consume).

### `P1Config` — gate configuration (selected fields)

```python
@dataclass
class P1Config:
    unlock_k: int = 3
    grid_points_per_axis: int = 5
    kc_thresholds: KCThresholds = field(default_factory=KCThresholds)
    walk_forward_top_n: int = 10          # how many top candidates the transfer gate looks at
    trade_ratio_scale: float = 1.0
    constraint_policy: str = "record"     # "record" | "prefer_feasible" | "hard_fail"
    # ... (zoom, baseline-comparison, and advisory fields; see source)
```

`constraint_policy="prefer_feasible"` selects the highest-scoring *feasible*
candidate; `"hard_fail"` blocks the run when no candidate is feasible.

---

## The transfer check directly

### `check_kc4` — walk-forward consistency

```python
def check_kc4(
    train_fitnesses: list[float],
    test_fitnesses: list[float],
    trade_ratio: float,
    thresholds: KCThresholds,
) -> KCReport
```

Computes the Pearson correlation between the train and test fitness of the top-N
grid points and compares it against `thresholds.pearson_min`. This is the function
`gate_scores` wraps.

### `KCThresholds` — pass thresholds (defaults from source)

```python
@dataclass(frozen=True)
class KCThresholds:
    time_box_seconds: float = 3 * 24 * 3600   # run budget (3 days)
    gini_min: float = 0.2                      # stress differentiation
    top_bot_ratio_min: float = 2.0             # stress head-vs-tail ratio
    min_nonzero_stress_count: int | None = None
    trade_count_min: int | None = 50           # per-candidate action floor (None -> SKIP)
    pearson_min: float = 0.3                    # minimum transfer correlation
    trade_ratio_min: float | None = 0.5         # test/train action ratio (None -> SKIP)

    @classmethod
    def pure_objective(cls, **overrides) -> "KCThresholds": ...
```

`KCThresholds.pure_objective()` disables the action-count gates (sets
`trade_count_min` and `trade_ratio_min` to `None`) for non-action objectives, while
keeping the time box, stress differentiation, and the transfer correlation gate.
Override any field, e.g. `KCThresholds.pure_objective(pearson_min=0.5)`.

---

## Parameter sensitivity

### `measure_stress` — rank parameters by perturbation sensitivity

```python
def measure_stress(
    target: CalibrableTarget,
    baseline_params: dict[str, Any],
    baseline_result: EvalResult,
    subset: list[str] | None = None,
    options: StressOptions | None = None,
    *,
    executor: Executor | None = None,
) -> list[StressResult]
```

Returns per-parameter stress (sensitivity) results, used by the stress-
differentiation gate and reported in the scorecard.

---

## Gate an existing search

### `audit_optuna_study` — gate an Optuna study

```python
def audit_optuna_study(
    study: Any,                                 # an optuna.study.Study (single-objective)
    *,
    holdout_evaluate: HoldoutEvaluate | None = None,
    thresholds: KCThresholds | None = None,
    top_n: int = 10,
) -> StudyAuditReport
```

Extracts the study's completed trials, re-evaluates the train-best top-N under
`holdout_evaluate`, runs the reused `WalkForward` + `check_kc4` gate (no duplicated
math), and splits `best_any` vs `best_feasible` from per-trial
`user_attrs["feasible"]` flags. `import optuna` is **lazy inside the function**, so
the module imports safely without optuna installed (install the optional `[p2]`
extra to use it). Minimize-direction studies are handled; multi-objective studies
are rejected. Works conceptually on any leaderboard — for Ax / Ray Tune / Hyperopt
/ `GridSearchCV` / hand-rolled sweeps, feed the train and holdout score lists to
`gate_scores` instead.

---

## Reporting

### `render_html` — single-file HTML scorecard

```python
def render_html(
    obj: Any,                          # P1Result | AuditReport | StudyAuditReport | GateVerdict (or their dicts)
    path: str | Path | None = None,
    *,
    generated_at: str | None = None,
) -> str
```

Pure-stdlib string templating (no matplotlib, no template engine). Produces a
deterministic, single-file dark-theme HTML scorecard: verdict banner, `best_any` vs
`best_feasible` table, stress ranking, and an inline SVG train-vs-holdout scatter.
Output is deterministic unless `generated_at=` is passed. Returns the HTML string;
also writes to `path` when given.

---

## Stability

These symbols are part of the consumed-surface contract documented in
[`DOCKING.md`](../DOCKING.md). Consumers pinning `omega-lock>=0.3.0,<0.4.0` rely on
them; they are additive across minor versions and are not renamed or re-defaulted
within the pin window.
