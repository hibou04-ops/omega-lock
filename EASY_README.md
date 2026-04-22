# Omega-Lock — Easy Start

> The short version, for people who found the main README intimidating.
> Full doc: [README.md](README.md) · 한국어 Easy: [EASY_README_KR.md](EASY_README_KR.md)

## What problem does it fix?

You tune some parameters. The score looks amazing on your training data. You ship. **It breaks in production.**

This is overfitting. Omega-Lock is a bouncer at the door — it refuses to let "amazing on training" pass unless it also survives train→test→holdout checks and your hard constraints.

## 60-second mental model

```
Your optimizer (grid / TPE / Bayesian / custom)
        ↓ produces a candidate
   [ Omega-Lock audit ]
        ↓
   PASS  →  ship it, with a signed trail
   FAIL  →  told you exactly which gate failed (KC-1..4)
```

You keep your optimizer. Omega-Lock only decides whether the tuned result is **trustworthy**.

## Install

```bash
pip install omega-lock
```

## The minimum working example

```python
from omega_lock import run_p1, P1Config
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

# 1. Wrap your target (something with param_space() + evaluate())
wrapped = AuditingTarget(
    my_target,
    constraints=[
        Constraint("score_positive",
                   lambda p, r: r.fitness > 0,
                   "Score must be positive"),
    ],
)

# 2. Hand it to any optimizer. run_p1 is the shipped one.
result = run_p1(train_target=wrapped, config=P1Config())

# 3. Get an audit report
report = make_report(wrapped, method="run_p1", seed=42)
print(render_scorecard(report))
```

You get a pass/fail scorecard + a JSON trail of every evaluation. Drop the trail in Git; future-you will thank you.

## When it's worth it

- Each evaluation is **expensive** (SPICE sims, backtests, LLM calls).
- You have a **train / test** split, ideally a **holdout** too.
- Someone downstream needs to **trust** the tuned candidate (regulator, ops, you in 6 months).

## When it's overkill

- Throwaway toy problem.
- Samples are effectively unlimited and the objective is smooth.
- Nobody is going to review the run afterwards.

In those cases, just use Optuna / grid search directly.

## Three things newcomers trip on

1. **"What's KC-4?"** A pre-declared threshold on Pearson correlation between train and test fitness. If train fitness doesn't predict test fitness, the candidate is overfit — and the run is marked `FAIL:KC-4`. Non-negotiable. That's the point.
2. **"Can I use my own optimizer?"** Yes. Wrap any callable with `CallableAdapter`. The audit doesn't care where the candidate came from.
3. **"Do I need Optuna?"** No. The default pipeline uses grid search. `pip install "omega-lock[p2]"` only if you want TPE.

## Where to go from here

- **Just try it**: `python examples/phantom_demo.py` — a 12-parameter synthetic problem end-to-end.
- **Realistic demo**: `python examples/demo_sram.py` — a 6T SRAM bitcell across 5 physical corners with 3 hard constraints.
- **Full docs**: [README.md](README.md) has the benchmark numbers, the philosophy, and the API reference.

License: Apache 2.0. Copyright (c) 2026 hibou.
