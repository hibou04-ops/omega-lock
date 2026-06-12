# Changelog

This changelog records local repository release notes only. It is not PyPI
publication proof, GitHub release proof, or release approval.

## 0.3.7

Marketplace: shorten action.yml description to GitHub's 125-char limit so the
composite Action can be published to GitHub Marketplace. No library/API changes.

- The composite Action's `action.yml` `description` was rewritten to a single
  line under GitHub's 125-character Marketplace limit (114 chars) so the
  overfit gate can be published to the GitHub Marketplace and consumed as
  `uses: hibou04-ops/omega-lock@v0.3.7`. The action's inputs, outputs, run
  steps, and branding are unchanged; only the version default and the usage
  example pin advanced to `0.3.7`.
- No source, behavior, or public-API change. Every public symbol, the
  consumed-surface contract, the SHA-256 audit hash chain, and the docking
  guard (12/12) are unchanged, so consumers pinning
  `omega-lock>=0.3.0,<0.4.0` are unaffected. The release is backward
  compatible; `publish.yml` reads the version from `pyproject.toml`.

## 0.3.6

Ship a composite GitHub Action (`action.yml`) so the overfit gate can run as
`uses: hibou04-ops/omega-lock@v0.3.6` in CI / be published to GitHub
Marketplace. No library/API changes.

- New repo-root `action.yml`: a composite action that sets up Python, installs
  `omega-lock` from PyPI (pinned to this release by default, or `latest`), and
  runs `omega-lock gate` over a `train` / `holdout` score pair with a
  configurable `pearson-min` transfer threshold and an optional HTML
  `report`. It exposes a `passed` output and fails the job (non-zero exit)
  when the held-out scores do not track the in-sample ranking. Branding:
  `icon: shield`, `color: blue`.
- No source, behavior, or public-API change. Every public symbol, the
  consumed-surface contract, the SHA-256 audit hash chain, and the docking
  guard (12/12) are unchanged, so consumers pinning
  `omega-lock>=0.3.0,<0.4.0` are unaffected. The release is backward
  compatible; `publish.yml` reads the version from `pyproject.toml`.

## 0.3.5

Documentation / distribution release: jargon-free, search-optimized README
family (EN/KR + easy variants), comparison tables, HOW_IT_WORKS + API docs;
no API changes — backward compatible, docking surface untouched.

- The README family was rewritten for distribution: a 30-second hook, a
  searchable keyword line, plain-English explanations, and comparison tables
  positioning omega-lock against the tools people already know (Optuna / Ax /
  Ray Tune, MLflow / W&B, promptfoo / DSPy). Private internal jargon was
  removed from all prose; the frozen public API symbols (`run_p1`,
  `check_kc4`, `KCThresholds`, ...) survive only as a one-line plain-language
  glossary in a single bottom note. The four README variants
  (`README.md`, `README_KR.md`, `EASY_README.md`, `EASY_README_KR.md`)
  cross-link to each other.
- New `docs/HOW_IT_WORKS.md`: a plain-English explanation of the walk-forward
  transfer gate, hard-constraint feasibility, and the append-only audit trail,
  with the real default thresholds drawn from source.
- New `docs/API.md`: a power-user reference mapping the plain README names to
  the frozen public symbols, with exact signatures.
- No source, behavior, or public-API change. Every public symbol, the
  consumed-surface contract, the SHA-256 audit hash chain, and the docking
  guard (12/12) are unchanged, so consumers pinning
  `omega-lock>=0.3.0,<0.4.0` are unaffected. Golden audit fixtures were
  regenerated only to carry the new version string (the version is not part of
  the hashed payload, so every chain digest is byte-identical).

## 0.3.4

All changes are additive: no existing public symbol was renamed, moved, or
re-defaulted, and no consumed wire key changed, so consumers pinning
`omega-lock>=0.3.0,<0.4.0` are unaffected.

- New installed console command `omega-lock` via `[project.scripts]`
  (`src/omega_lock/cli.py`, argparse only) with three subcommands:
  `omega-lock demo` (runs the walk-forward gate case study), `omega-lock
  gate --train a.json --holdout b.json [--report out.html] [--pearson-min X]`
  (KC-4 Pearson transfer gate over two JSON arrays of scores; exit code 0/1),
  and `omega-lock report --input p1_result.json -o out.html` (renders a saved
  `P1Result` or audit-report JSON artifact). There is still no `omega-lock
  diff` command. The case-study engine moved into the package as
  `omega_lock._demo` so the wheel can run it; the example file re-exports the
  same symbols and prints the identical, test-pinned narrative.
- New Optuna bridge API `audit_optuna_study(study, *, holdout_evaluate=None,
  thresholds=None, top_n=10) -> StudyAuditReport` in
  `omega_lock.integrations.optuna_bridge` (re-exported at the package root).
  It extracts completed trials, re-evaluates the train-best top-N under the
  caller's `holdout_evaluate`, runs the reused `WalkForward` + `check_kc4`
  gate (no duplicated math), and splits `best_any` vs `best_feasible` from
  per-trial `user_attrs["feasible"]` flags (documented as absent otherwise).
  Minimize-direction studies are handled; multi-objective studies are
  rejected. `import optuna` stays lazy inside the function with a clean
  install hint, so the module imports safely without optuna.
  `examples/optuna_audit_demo.py` now uses this API instead of hand-rolled
  bridge plumbing.
- New stdlib-only HTML scorecard `render_html(obj, path)` in
  `omega_lock.report_html`, accepting `P1Result` | `AuditReport` |
  `StudyAuditReport` | `GateVerdict` (objects or their serialized dict
  forms): verdict banner per KC gate, `best_any` vs `best_feasible` table
  (train vs holdout), stress ranking table, and an inline SVG scatter of
  train vs holdout fitness with the identity line. Pure string templating —
  no matplotlib, no template engine — and deterministic output (no
  timestamps unless `generated_at=` is passed).
- New plain-language facade `omega_lock.simple`:
  `gate_scores(train_scores, holdout_scores) -> GateVerdict` (wraps the KC-4
  Pearson gate; `GateVerdict` carries `passed`, `pearson`, `reasons`),
  `audit(target_fn, param_specs, *, holdout_fn=None, **cfg) -> P1Result`
  (thin `CallableAdapter` + `run_p1` wrapper with a `pure_objective`
  threshold default and friendly `{name: (low, high)}` spec syntax), and a
  `render_html` re-export. `GateVerdict`, `gate_scores`, `render_html`,
  `StudyAuditReport`, `TrialCandidate`, and `audit_optuna_study` are added
  to `omega_lock.__all__`; `simple.audit()` itself stays at
  `omega_lock.simple.audit` because the root name `omega_lock.audit` already
  belongs to the audit subpackage.
- New deterministic case-study example `examples/walkforward_gate_demo.py`:
  naive best-score selection overfits to slice noise, the KC-4 walk-forward
  gate fails the run, and constraint-gated feasible-best selection holds up
  on a holdout slice. Offline, seeded, hash-based noise; runs in well under
  a second. Pinned by `tests/test_walkforward_gate_demo.py`.
- Bridge example `examples/optuna_audit_demo.py`: gate an existing Optuna
  study's completed trials and write an HTML scorecard via the new API.
  Skips gracefully when optuna is not installed.
- README entry-point rework: leads with the value proposition ("the best
  score is not deployable"), the demos as Quickstart, and a terminology
  decoder table (P1/P2, KC-1..4, SC-2, `best_any` vs `best_feasible`,
  stress/unlock/lock, `pure_objective`). The per-release summary block moved
  to `docs/WHATS_NEW.md`; no content was deleted.
- `docs/EXAMPLES_GALLERY.md` lists the new examples and the CLI;
  `docs/claims/public_claims.yml` gains ledger-backed claims for the CLI,
  the Optuna bridge, the HTML scorecard, and the facade.
- Golden audit fixtures regenerated only to carry the new version string;
  the audit report schema and SHA-256 hash chain are unchanged (the version
  is not part of the hashed payload, so every chain digest is
  byte-identical).

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
