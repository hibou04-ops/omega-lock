# Changelog

This changelog records local repository release notes only. It is not PyPI
publication proof, GitHub release proof, or release approval.

## Unreleased

- New deterministic case-study example `examples/walkforward_gate_demo.py`:
  naive best-score selection overfits to slice noise, the KC-4 walk-forward
  gate fails the run, and constraint-gated feasible-best selection holds up
  on a holdout slice. Offline, seeded, hash-based noise; runs in well under
  a second. Pinned by `tests/test_walkforward_gate_demo.py`.
- New bridge example `examples/optuna_audit_demo.py`: gate an existing Optuna
  study's completed trials through `WalkForward` + `check_kc4` and
  feasible-best selection in a ~15-line bridge. Skips gracefully when optuna
  is not installed.
- README entry-point rework: leads with the value proposition ("the best
  score is not deployable"), the two demos above as Quickstart, and a
  terminology decoder table (P1/P2, KC-1..4, SC-2, `best_any` vs
  `best_feasible`, stress/unlock/lock, `pure_objective`). The per-release
  summary block moved to `docs/WHATS_NEW.md`; no content was deleted.
- `docs/EXAMPLES_GALLERY.md` lists the two new examples. No library code
  changes and no public API changes.

## 0.3.3

- Classifier promotion only. The PyPI trove classifier `Development Status`
  advanced from `3 - Alpha` to `4 - Beta`. There is no functional code change
  since 0.3.2: the dormant, default-off parallel-execution executor seam
  (`GridSearch.run`, `ZoomingGridSearch.run`, `measure_stress`,
  `WalkForward.run` `executor=` keyword) and the source-distribution packaging
  fix that shipped in 0.3.2 both stand unchanged. Consumers that pin
  `omega-lock>=0.3.0` are unaffected.
- Golden audit fixtures regenerated only to carry the new `omega_lock_version`
  string. The audit report schema and SHA-256 hash chain are unchanged (the
  version is not part of the hashed payload, so every chain digest is
  byte-identical).

## 0.3.2

- Source-distribution packaging fix (the reason for this release). The 0.3.1
  sdist shipped `tests/` but not `scripts/`, `tests/fixtures/`, or `examples/`,
  so `pip download omega-lock --no-binary :all:` then `pytest --collect-only`
  on the unpacked sdist threw collection errors (module-level script loaders
  and one example importer). A `MANIFEST.in` now grafts `scripts/`, `tests/`
  (including `tests/fixtures/`), and `examples/` into the sdist. The wheel is
  unchanged — it still ships only the import package, with no tests, scripts, or
  examples — and the sdist excludes compiled bytecode (`__pycache__`, `*.pyc`).
- Dormant, default-off parallel-execution seam. `GridSearch.run`,
  `ZoomingGridSearch.run`, `measure_stress`, and `WalkForward.run` gained an
  optional `executor: concurrent.futures.Executor | None = None` keyword
  argument. When omitted (the default), evaluation is strictly serial and
  byte-identical to prior behavior. When an executor is supplied, work is
  dispatched through a shared internal `_ordered_eval_map` helper that
  reassembles results in INPUT order (load-bearing for walk-forward Pearson
  pairing and stress z-score normalization). For `ZoomingGridSearch`, only the
  within-round combo loop parallelizes; the zoom rounds stay sequential because
  each round re-centers on the prior winner.
- These are additive optional keyword arguments only — no `EvalResult`,
  `P1Result`, or `StressResult` shape changed, and no consumed contract key
  changed, so consumers that pin `omega-lock>=0.3.0` are unaffected and the
  consumed-surface contract manifest subset check stays green.
- Golden audit fixtures regenerated only to carry the new `omega_lock_version`
  string. The default-off seam produces zero additional golden change; the
  audit report schema and SHA-256 hash chain are unchanged (the version is not
  part of the hashed payload, so every chain digest is byte-identical).

## 0.3.1

- Internal guard and documentation hardening only. No public API changes and
  no runtime behavior changes; consumers that pin `omega-lock>=0.3.0` are
  unaffected.
- Hardened the omega family docking guard surface: the producer contract
  manifest (`src/omega_lock/contract.py`), the family docking convention
  (`DOCKING.md`), and a tier-aware offline presence-lint
  (`scripts/check_docking_presence.py`, tested by
  `tests/test_docking_presence.py`) that mechanically asserts each declared
  cross-repo coupling carries its guard and tier-correct pin.
- `DOCKING.md` updated: the presence-lint is now recorded as the implemented C4
  "teeth" (previously deferred as an owner decision) and referenced from the C2
  registry as the machine-checked enforcement of its rows.
- README family + version surfaces synchronized to `0.3.1`.
- Golden audit fixtures regenerated to carry the new `omega_lock_version`
  string; the audit report schema and SHA-256 hash chain are unchanged. The
  only fixture delta is the embedded `omega_lock_version` value — the version is
  not part of the hashed payload, so every chain digest is byte-identical.

## 0.3.0

- Add `KCThresholds.pure_objective()` preset: disables the action-count gates
  (KC-3 and the KC-4 trade-ratio sub-gate) for non-action objectives while
  keeping the domain-neutral gates (time box, stress differentiation,
  walk-forward correlation).
- Domain-neutral public field names with backward-compatible aliases:
  `EvalResult.sample_count` (alias `n_trials`), `ParamSpec.stress_suppressed`
  (alias `ofi_biased`), `StressResult.stress_suppressed` (`to_dict` dual-emits
  both keys), config `exclude_suppressed_in_unlock` (mirror
  `exclude_ofi_in_unlock`), and result `top_k_excl_suppressed` (mirror
  `top_k_ex_ofi`). No breaking changes.
- Documentation and example wording cleanup; README family + version surfaces
  synchronized to `0.3.0`.
- Tamper-evident audit report schema and golden fixtures unchanged
  (SHA-256 hash chain preserved).

## 0.2.7

- Local package version surfaces synchronized to `0.2.7`.
- README family (`README.md`, `README_KR.md`, `EASY_README.md`,
  `EASY_README_KR.md`) top-section refactor for faster trust, positioning, and
  verification scanning: "Use it when", "Trust loop", and verification/evidence
  links are now near the top.
- Added a qualitative "How is this different?" comparison table near the top of
  `README.md`, tracked as the `comparative_positioning` qualitative marker in
  the claim ledger.
- Regenerated `docs/claims/generated_readme_claims.*` and the golden audit
  fixtures so embedded version metadata matches `0.2.7`.
- No runtime behavior changes; only version metadata, documentation, and
  regenerated deterministic artifacts changed.
- Registry publication status is not asserted here and must be verified
  separately after release.

## 0.2.6

- Local package version surfaces use `0.2.6`.
- Fixed `scripts/post_release_verify.py` PyPI JSON fetch timeout handling so
  the timeout is passed to `urllib` as `timeout`, not as request data.
- Preserved the injected opener test interface as `(request, timeout)`.
- Registry publication status is not asserted here.
- Release preparation and verification remain governed by `RELEASE.md` and the
  offline release audit scripts.
