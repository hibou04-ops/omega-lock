# How omega-lock works

[← Back to README](../README.md) · [Power API for integrators](API.md) · [Trust & audit model](TRUST_MODEL.md)

This page explains, in plain English, what omega-lock actually does when it judges
a candidate — the transfer gate, the feasibility check, and the audit trail — and
the real numeric thresholds it uses by default. Every threshold below comes
straight from the source (`src/omega_lock/kill_criteria.py`,
`KCThresholds`), not from marketing copy.

---

## The problem in one sentence

When you try many candidates and keep the single best one, the winning score is
the **most likely candidate to be inflated by luck**, because selecting-the-max is
exactly the operation that concentrates noise. omega-lock measures how much of that
score was luck by re-testing the winner on data it was never selected against.

---

## Check 1 — the walk-forward transfer gate

This is the core check. It answers: *does the score earned on the tuned data carry
over to a held-out slice the search never saw?*

**How it works.** You give omega-lock two index-aligned score lists for the same
set of candidates:

- the score each candidate earned on the data the search consumed ("train"), and
- the score of those *same* candidates re-evaluated on a held-out slice ("holdout").

omega-lock looks at the top-ranked candidates (by default the top 10,
`walk_forward_top_n = 10`) and computes the **Pearson correlation between the train
ranking and the holdout ranking**.

- **If the correlation is high**, the candidates that won on tuned data also win on
  fresh data — the ranking is real, and the gate **PASSes**.
- **If the correlation collapses**, the winner's lead was noise that did not
  reappear on fresh data — the gate **FAILs** and blocks the deploy.

**The threshold.** The default minimum transfer correlation is
**`pearson_min = 0.3`** (`KCThresholds.pearson_min`). A run whose top-N
train↔holdout Pearson falls below `0.3` fails the transfer gate. In the bundled
`omega-lock demo`, the naive top-score winner scores a train↔holdout Pearson of
`0.179`, which is below `0.3`, so the gate stamps FAIL — while the
constraint-feasible candidate scores `0.909` and PASSes.

**Why Pearson on a ranking.** A real edge keeps the same candidates near the top on
both slices, so their two score columns move together (high correlation). Pure
selection luck reshuffles the top on the unseen slice, so the columns decorrelate.
The correlation is the signal-vs-luck meter.

For non-action objectives (plain math / ML / simulation scores that have no "number
of actions" concept), use `KCThresholds.pure_objective()`. It keeps the transfer
gate and the stress-differentiation check but disables the action-count gates (they
would otherwise fail vacuously). `omega_lock.simple.gate_scores()` already defaults
to this preset, which is why you can hand it two bare number lists.

---

## Check 2 — hard-constraint feasibility

A high score is worthless if the candidate that earned it is one you can't actually
ship — too slow, too expensive, over a risk limit, against a rule you declared.

**How it works.** You attach constraints to your target (latency caps, cost caps,
risk limits — any predicate over a candidate's params and result). omega-lock
evaluates and records pass/fail for every constraint on every candidate, then
reports two distinct winners:

- **`best_any`** — the highest raw score, constraints ignored.
- **`best_feasible`** — the highest score *among candidates that satisfy every
  declared constraint*.

When **`best_feasible ≠ best_any`**, your top score came from a candidate you can't
run. The selection policy (`P1Config.constraint_policy`) controls what happens:

- `"record"` (default) — report both winners; violations live in the audit trail.
- `"prefer_feasible"` — select `best_feasible` as the certified winner.
- `"hard_fail"` — block the run (`FAIL:CONSTRAINTS`) if no candidate is feasible.

This is why the demo's verdict line reads `hard-constraint feasibility … FAIL
(best_feasible ≠ best_any)`: the lucky top-scorer sat outside the declared envelope.

---

## Check 3 — the append-only audit trail

This check never blocks a deploy. Its job is **proof, not gating**: so that months
later you can reconstruct exactly what was judged, when, and why.

**How it works.** Every probed candidate is appended — in order — to a reviewable
JSON trail that records the candidate, its score, the phase/role/round context, and
the thresholds in force at decision time. The trail is **append-only**: entries are
added, never rewritten.

**Tamper evidence.** The audit JSON can carry an optional **SHA-256 hash chain**:
each record's digest folds in the previous record's digest, so any after-the-fact
edit to an earlier entry breaks the chain and is detected on readback. The verdict,
inputs, and thresholds are all part of the recorded, chained payload.

---

## The other built-in checks (brief)

omega-lock's full pipeline also runs two domain-neutral sanity gates, with these
default thresholds (`KCThresholds`):

- **Time box** — a run must finish within its budget (`time_box_seconds`, default
  3 days). A blown budget is a FAIL.
- **Stress differentiation** — the parameters must show real, differentiated
  sensitivity, not a flat or single-spike profile. Defaults: Gini of the stress
  ranking `gini_min = 0.2`, head-vs-tail ratio `top_bot_ratio_min = 2.0`. There is
  an optional advisory that demands non-zero stress on at least N axes
  (`min_nonzero_stress_count`, default off).

The action-count gates (a per-candidate action floor and a test/train action ratio,
`trade_count_min = 50`, `trade_ratio_min = 0.5`) apply only to action-style targets
and are turned off by `pure_objective()`.

---

## What omega-lock does NOT claim

- It does not prove a candidate is **globally optimal** — it judges the candidate
  your search already chose.
- It does not prove **correctness or root cause** — it measures transfer, not truth.
- It does not prove **PyPI / GitHub publication** — registry status needs separate
  post-release verification. Local version metadata is not registry proof.
- It is **not a tracker or dashboard** — it is a pass/fail gate over the output of
  the search you already run.

See [TRUST_MODEL.md](TRUST_MODEL.md) for the full trust boundary, and [API.md](API.md)
for the exact function signatures behind each check.
