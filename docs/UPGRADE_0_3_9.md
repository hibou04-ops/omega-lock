# 0.3.9: numerical boundaries and bounded parallel evaluation

## What changed

`pearson_result` translates and scales finite score columns before computing
centered dot products with `math.fsum`. This avoids squaring unscaled huge values
and underflowing subnormal differences. Nonfinite input now has explicit
`NON_FINITE_X` / `NON_FINITE_Y` outcomes with `value=None`; `pearson()` retains its
legacy 0.0 fallback for uncomputable input. KC-4 fails these cases.

Optional parallel evaluation retains at most 64 submitted but unconsumed Futures,
regardless of grid length. Every input is evaluated once on success and results
stay in input order. Returned results and the existing input grid still take
O(N) memory. On failure the original exception is raised and outstanding queued
Futures are cancelled best-effort; already running tasks can finish. The caller
retains ownership of the Executor.

Use a `ThreadPoolExecutor` or compatible in-process executor: public evaluation
paths use closures that a process pool cannot pickle. Targets must be safe for
concurrent evaluation. Shared global random generators and mutable target state
do not have a serial/parallel reproducibility guarantee. The 64-task window can
limit concurrency when an executor has more than 64 workers; default execution
remains serial. No evaluation reuse or duplicate-candidate elimination was added.

`gate_scores` optionally checks paired `train_ids` and `holdout_ids`. Both lists
must contain one unique, nonempty string per score in matching order. Missing,
duplicated, reordered or different IDs raise `ValueError`. Without IDs, existing
index alignment remains the caller's responsibility.

## Reproduction and scope of the evidence

Baseline: `1005970931403bdc426d11dc21a16b41f09007f5` (`v0.3.8`), which was both
the starting main HEAD and the latest public release. Use separate environments
for baseline and the candidate. The same standard-library resource/correctness
probe was run against each installed package:

```bash
python scripts/measure_upgrade.py
python -m pytest tests/test_numerical_boundaries.py tests/test_parallel_executor_seam.py -q
```

The probe's workloads and accounting were fixed before the baseline run: five
explicit numeric columns and 10,000 deterministic evaluations, counting retained
unconsumed Futures and real evaluation invocations. This is a development
regression probe, not a blind external benchmark.

Observed locally on Windows 11 x64, Python 3.12.10, NumPy 2.5.4, Optuna 4.9.0:

| Probe | 0.3.8 | 0.3.9 |
|---|---|---|
| `[1e200,2e200,3e200]` versus its reverse | `OverflowError` | `OK`, Pearson -1 |
| `[1e-310,2e-310,3e-310]` versus its reverse | false zero-variance result | `OK`, Pearson -1 |
| `[-1e308,0,1e308]` versus its reverse | `OverflowError` | `OK`, Pearson -1 |
| representable differences near `1e16` | `OK`, approximately -1 | `OK`, approximately -1 |
| NaN in train column | `OK`, value NaN | `NON_FINITE_X`, uncomputable |
| Peak retained unconsumed Futures, 10,000 items | 10,000 | 64 |
| Actual evaluation calls / input-order outputs | 10,000 / correct | 10,000 / correct |

The Future probe uses an instrumented synchronous Executor to count retention
exactly; it does not measure real provider latency or throughput. Separate real
ThreadPoolExecutor regressions check out-of-order completion and preserve
grid/stress/walk-forward results. New numeric tests also compare nontrivial
correlations at five scales against an independent 100-digit Decimal calculation.
The unchanged baseline suite had 556 passing tests in this environment.

No end-to-end runtime speedup, total-process memory reduction, model-quality gain
or generalization result is claimed. The resource improvement is specifically
the number of Future objects retained by the parallel evaluation map.

## Compatibility and interpreting a result

The package remains in the 0.3.x dependency range. Public identifiers, positional
arguments, existing index-only behavior, audit schema IDs and golden audit
fixtures are retained. Float results can differ in final bits because of stable
arithmetic. Exhaustive `PearsonStatus` readers must accept `NON_FINITE_X` and
`NON_FINITE_Y`. Historical artifacts are not reinterpreted or rewritten.

Previously nonsensical thresholds now raise `ValueError`: `pearson_min` must be
finite within `[-1,1]`, and `trade_ratio_min` must be finite and nonnegative or
`None`. An enabled action-ratio gate fails NaN/Inf measurements. `top_n` must be
a positive integer. The CLI maps nonfinite score files and invalid correlation
thresholds to input error exit 2; valid input below the gate threshold remains
exit 1, and passing the declared gate remains exit 0.

Pearson measures **linear score association**. It is not rank correlation, an
absolute-quality gate, proof of independent holdout data or evidence of general
performance. `[1,2,3]` and `[-99,-98,-97]` still pass the default correlation gate.
Use separately declared quality constraints for such requirements; a PASS does
not establish that deployment is safe. Candidate IDs do not establish data
independence, and paired remeasurement is not unseen-input validation.

## Install and offline smoke

```bash
python -m pip install omega-lock==0.3.9
python -c "import json; from pathlib import Path; Path('train.json').write_text(json.dumps([1,2,3])); Path('holdout.json').write_text(json.dumps([1,2,3]))"
omega-lock gate --train train.json --holdout holdout.json --report gate.html
```

Expected: PASS, Pearson 1.000, exit 0. Reversing holdout produces FAIL/1.
Invalid JSON or nonfinite scores produce input error/2. No API key is required.
