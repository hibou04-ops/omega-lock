# Omega family docking — "born docked"

> Family anchor for the 3-repo omega family: **omega-lock** (this repo) ·
> **omegaprompt** · **antemortem-cli**.
> This is the **Tier C** deliverable of the omega docking hardlock: the family
> convention (C1), the single coupling registry (C2), and the graduation spec
> (C3). It governs how cross-repo couplings are created and guarded so that
> future patch/version skew cannot silently break structure across the family.
>
> Source of truth for the design: `omega-docking-plan/FINAL_PLAN.md` (Tier C
> section). This file lives at the omega-lock repo root because omega-lock is the
> family anchor (greenfield — no CONTRIBUTING / AGENTS / CLAUDE.md exists in any
> of the three repos). Filename `DOCKING.md` is a convention choice; the spec
> mandated only "omega-lock repo root."

---

## C1 — The "born docked" rule

**Every new cross-repo coupling in this family ships with its guard in the same
change.** A coupling is "born docked" or it does not merge.

Concretely, no new inter-repo coupling merges without all three of:

1. **A contract declaration** — an explicit, written statement of exactly what
   one repo consumes from another (the field names / signatures / citation
   targets), so the seam is reviewable instead of implicit.
2. **A version pin** — a declared, bounded reference to the depended-on repo
   (a dependency range for a runtime seam; an immutable checkout ref for a
   doc-citation seam).
3. **A CI guard test** — an automated check that fails **loud, named, and
   pre-merge** when the coupling drifts, so a break surfaces at a PR rather than
   as a silent mid-run failure in production.

If a change introduces a new way that one repo depends on another and any of
(1)–(3) is missing, the change is not done. Add the guard in the same PR, or do
not add the coupling.

### Asymmetry is load-bearing — Tier B never gains a runtime handshake

The family is **not symmetrically coupled**, so the dock is **not uniform**:

- A **runtime seam** (Tier A) is where one repo imports and executes the other
  at run time. It is guarded by two *independent* contract declarations that
  meet in CI **with no import between the guards** — a producer self-check in the
  producer's CI and a consumer fail-loud test in the consumer's CI.
- A **doc-citation seam** (Tier B) is where one repo only *cites* another's
  source in prose (README / docs). It is guarded by a **CI-time pinned checkout**
  that reuses the citing repo's own citation verifier. A checkout is **not** a
  runtime dependency.

**Tier B must NEVER acquire a runtime handshake.** antemortem-cli keeps
**zero `import omega_lock`** and **zero omega-lock dependency**. The citation
guard runs only at CI time against a pinned checkout; it never links the two
packages at run time. Adding a runtime import or an omega-lock dependency to
antemortem-cli to "tighten" the citation seam is the cardinal asymmetry sin and
is explicitly forbidden — it would re-introduce exactly the version-skew coupling
this dock exists to remove. A doc-only seam stays doc-only.

---

## C2 — Coupling registry

The single source of truth for which cross-repo couplings exist, how each is
pinned, and which test guards each. This table lists **only the real couplings
implemented to date**. A coupling that is not in this table does not exist; a row
without a working guard is a bug.

| # | Coupling (direction) | Tier | Kind | Pin | Guard(s) | Runtime import? | Status |
|---|---|---|---|---|---|---|---|
| 1 | omega-lock (producer) → omegaprompt (consumer) | **A** | Runtime seam | `omega-lock>=0.3.0,<0.4.0` (omegaprompt `pyproject.toml`) | **Producer CI:** `tests/test_contract_manifest.py` (omega-lock) self-checks its emitted wire-keys + signatures against `src/omega_lock/contract.py`'s `CONSUMED_CONTRACT`. **Consumer CI:** `tests/test_omega_lock_contract.py` (omegaprompt) fails loud against the installed dep; **scheduled `@main` canary** `.github/workflows/omega-lock-compat.yml` (omegaprompt) runs the same consumer test against bleeding-edge omega-lock. **No import between the two CI guards.** | Yes (omegaprompt imports `omega_lock`) | ACTIVE |
| 2 | antemortem-cli → omega-lock | **B** | Doc-citation only (asymmetric) | omega-lock **SHA `c03b8ac3c97752f64796dee49f9f11ab90cbce7d`** (= tag `v0.3.0`), pinned as a CI-time `actions/checkout` ref into `_omega_lock_pin/` — **NOT a dependency** | `scripts/check_omega_lock_citations.py` (antemortem, reuses antemortem's own `citations.verify_citation`) run by the isolated `omega-lock-citation-drift` CI job in `.github/workflows/ci.yml`; plus the offline namespace-invariant test `tests/test_omega_lock_citation_invariant.py` (antemortem) | **No — zero `import omega_lock`, zero omega-lock dep (load-bearing; see C1)** | ACTIVE |

### Pin discipline notes

- **Tier A pin is a dependency range** (`>=0.3.0,<0.4.0`) because omegaprompt
  installs and runs omega-lock. The consumer guards are hardcoded to the keys /
  signatures they read — they do **not** import omega-lock's `CONSUMED_CONTRACT`
  manifest; the two declarations are deliberately independent and meet only in
  CI.
- **Tier B pin is an immutable SHA**, not a range and not a mutable tag.
  `c03b8ac` is annotated as `= tag v0.3.0` for human readability, but CI pins the
  SHA so a re-tag cannot silently change what the citation guard checks against.
  (The FINAL_PLAN C2 prose originally said "TAG v0.3.0"; the friction-review
  superseding item — and the shipped CI — use the immutable SHA. SHA wins.)

### Tier B citation status

The two previously LIVE-broken omega-lock citations that FINAL_PLAN C2 said to
"seed as OPEN until B2 lands" are **RESOLVED** — B2 (and B1, B5) have landed:

- `README.md:300` / `README_KR.md:124` → `src/omega_lock/walk_forward.py:153`
  (was the broken `:82`). **RESOLVED (EN + KR).**
- `README.md:904` / `README_KR.md:659` link text → `src/omega_lock/kill_criteria.py`
  (URL was already correct; omega-lock has no `docs/methodology.md`).
  **RESOLVED (EN + KR).**
- The one fictional few-shot citation was de-namespaced
  (`src/omega_lock/core.py` → `src/example_pkg/core.py`), so "any
  `src/omega_lock/*` citation == a real claim" holds with no allowlist; the
  namespace-invariant test guards recurrence. **RESOLVED.**

No OPEN cross-repo citation rows remain. The unrelated Antemortem-repo
`docs/methodology.md` links are intentionally out of scope and untouched.

### Registry enforcement (machine-checked)

The rows above are no longer enforced by convention alone. The tier-aware
offline presence-lint — `scripts/check_docking_presence.py` (tested by
`tests/test_docking_presence.py`), shipped in this anchor repo — mechanically
asserts that each coupling in this registry carries its declared guard(s) and
its tier-correct pin. See **C4** for the full description; it is the implemented
"teeth" of the C1 "born docked" rule, not a new coupling.

---

## C3 — Graduation spec (what we deliberately DEFER)

The dock is intentionally the **lightest possible** mechanism. The following
heavier options are **deferred, not adopted** — listing them here so a future
contributor does not re-litigate them without cause:

- **DEFER O7 — a shared package.** No shared cross-repo package now. The
  producer-internal `CONSUMED_CONTRACT` manifest is the concrete payload a future
  shared mechanism *would* carry, but the family asymmetry argues against a
  heavyweight shared package today.
- **DEFER O6 — a runtime handshake.** No runtime handshake on the doc-only
  Tier B (see C1 — this is permanent for Tier B), and none added to Tier A
  beyond the two independent CI declarations.

**Escalation trigger.** Revisit graduation (toward O7 / a shared contract
package) only when a **SECOND live runtime coupling** appears in the family — the
canonical candidate being the KC-4 per-candidate gate becoming a real runtime
seam. One runtime coupling (Tier A) does not justify a shared package; two does
warrant re-opening the question.

**Documented non-goal.** The `mini-omega-lock` / `mini-antemortem-cli` ripple is
explicitly **not** in scope for this dock. Those distribute separately and are
noted here only so their existence is not mistaken for an unguarded coupling.

---

## C4 — Optional teeth (IMPLEMENTED — tier-aware presence-lint)

C4 was an open owner decision, deferred by design. It has since been **built**:
a tiny **offline presence-lint** that asserts, for each row in the C2 registry,
that the named guard-test file exists and the named pin exists. It does **not**
gate landing by historical mandate, but it is now available as the machine-checked
"teeth" backing the C1 "born docked" convention.

**Implementation (lives in the omega-lock anchor repo):**

- `scripts/check_docking_presence.py` — the tier-aware offline lint. It performs
  only deterministic local filesystem checks: no PyPI, GitHub, registry, or live
  provider access, and it imports none of the three repos.
- `tests/test_docking_presence.py` — the offline test suite that exercises the
  lint (present/missing guards, pin variants, asymmetry breach detection,
  missing-sibling SKIP behavior).

It **is** tier-aware, exactly as the original spec demanded:

- **Tier A** pin lives in `pyproject.toml` (a dependency range) — the lint
  parses `omegaprompt/pyproject.toml` via `tomllib` (whitespace-normalized) and
  checks for `omega-lock>=0.3.0,<0.4.0`.
- **Tier B** pin lives in the CI workflow as a checkout `ref` (an immutable SHA),
  **not** in any `pyproject.toml` — the lint checks for the
  `omega-lock-citation-drift` CI job in antemortem's `ci.yml` (the tier-aware
  stand-in for the forbidden runtime pin) and positively asserts **zero**
  `import omega_lock` in antemortem `src/`.

A naive "every row's pin must be a `pyproject` dependency" check would
**false-fail antemortem-cli** (Tier B has no dependency) or, worse, pressure
someone into adding an omega-lock dependency to antemortem — which is the
cardinal asymmetry sin (see C1). The shipped lint avoids this: it never looks
for a runtime pin on Tier B. Missing sibling repos **SKIP** (never hard-fail),
so omega-lock's single-repo CI stays green; a missing guard inside a present
repo, or an asymmetry breach, exits nonzero.
