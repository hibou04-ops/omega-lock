# Omega-Lock Trust Model

This document defines what Omega-Lock is intended to make trustworthy, where
that trust stops, and which local evidence backs each strong guarantee. It is
not release approval, security certification, or formal verification.

## Trust Boundary

Omega-Lock treats raw optimizer output as untrusted until it is replayed through
local audit gates. The optimizer may find a high-scoring candidate, but the
audit surface decides whether that candidate is feasible, reviewable, and
eligible to move forward under the declared constraints.

The trust boundary is local and artifact-based:

- Inputs: target evaluations, declared constraints, configuration, and optional
  train/test/holdout targets supplied by the caller.
- Audit process: Omega-Lock records evaluations, applies configured gates, and
  emits structured artifacts.
- Outputs: result JSON, audit reports, scorecards, generated claim documents,
  golden fixtures, and deterministic demo summaries.

Anything outside those local inputs and artifacts remains outside the trust
model.

## What Omega-Lock Guarantees

| Guarantee | Boundary | Evidence |
| --- | --- | --- |
| Declared hard constraints are evaluated and recorded on audited target calls. | The guarantee covers `Constraint` predicates supplied by the caller; it does not prove the predicate is the right business rule. | Claim ledger `hard_constraint_compliance`; [src/omega_lock/audit/_target.py](../src/omega_lock/audit/_target.py); [tests/test_constraint_aware_selection.py](../tests/test_constraint_aware_selection.py); [tests/test_constraint_uniqueness.py](../tests/test_constraint_uniqueness.py). |
| `best_any` and `best_feasible` are separate audit concepts. | `best_feasible` can be absent when no candidate satisfies declared constraints. `best_any` is still reported for review context. | Claim ledger `feasible_best_vs_absolute_best`; [src/omega_lock/audit/_types.py](../src/omega_lock/audit/_types.py); [src/omega_lock/audit/_scorecard.py](../src/omega_lock/audit/_scorecard.py); [tests/test_audit.py](../tests/test_audit.py). |
| Walk-forward validation can gate train-to-test behavior when a test target is supplied. | The gate checks configured statistical signals such as Pearson and trade ratio over the provided targets; it does not prove future performance. | Claim ledger `walk_forward_validation`; [src/omega_lock/walk_forward.py](../src/omega_lock/walk_forward.py); [src/omega_lock/orchestrator.py](../src/omega_lock/orchestrator.py); [tests/test_walk_forward_cache.py](../tests/test_walk_forward_cache.py); [tests/test_holdout_gate_mode.py](../tests/test_holdout_gate_mode.py). |
| Audit trails append one `AuditedRun` per evaluated audited call with phase, role, round, and call index context. | Append-only describes the in-process trail construction. A serialized JSON file can still be edited after it is written. | Claim ledger `append_only_audit_trail`; [src/omega_lock/audit/_target.py](../src/omega_lock/audit/_target.py); [src/omega_lock/audit/_types.py](../src/omega_lock/audit/_types.py); [tests/test_audit.py](../tests/test_audit.py); [tests/test_auto_phase_tracking.py](../tests/test_auto_phase_tracking.py). |
| Optional SHA-256 hash-chain data can detect mutation after verification when included and checked. | Hash-chain output is opt-in through `with_hash_chain=True`. It is not a signature, not key-backed authentication, and not cryptographic immutability. Detection happens only when a verifier recomputes and compares the chain. | Claim ledger `sha256_hash_chain_tamper_detection`; [src/omega_lock/audit/_types.py](../src/omega_lock/audit/_types.py); [tests/test_audit_hash_chain.py](../tests/test_audit_hash_chain.py); [tests/fixtures/golden_audits/append_only_hash_chain.json](../tests/fixtures/golden_audits/append_only_hash_chain.json). |
| Current audit report schema mismatches are rejected by `AuditReport.from_json`. | This covers the current audit report schema boundary. It is not a general migration system for every historical artifact. | [src/omega_lock/audit/_types.py](../src/omega_lock/audit/_types.py); [tests/test_audit_hash_chain.py](../tests/test_audit_hash_chain.py); [tests/fixtures/golden_audits/schema_validation_roundtrip.json](../tests/fixtures/golden_audits/schema_validation_roundtrip.json). |
| Deterministic offline demos and golden audit fixtures can catch semantic drift. | They validate stable local semantics, not external provider behavior or machine performance thresholds. | Claim ledger `deterministic_offline_demos`; [examples/demo_replay.py](../examples/demo_replay.py); [examples/demo_sram.py](../examples/demo_sram.py); [scripts/run_golden_audit_cases.py](../scripts/run_golden_audit_cases.py); [tests/test_golden_audit_cases.py](../tests/test_golden_audit_cases.py); [tests/test_demo_sram.py](../tests/test_demo_sram.py). |
| Public README claims are classified and regenerated from a local ledger. | The ledger records proof class and status. It does not make unproven claims true; qualitative claims stay qualitative. | [docs/claims/public_claims.yml](claims/public_claims.yml); [docs/claims/generated_readme_claims.md](claims/generated_readme_claims.md); [scripts/generate_readme_claims.py](../scripts/generate_readme_claims.py); [tests/test_generated_readme_claims.py](../tests/test_generated_readme_claims.py). |

## What Omega-Lock Does Not Guarantee

- It does not prove optimizer output is globally optimal.
- It does not prove the caller's objective function measures the right outcome.
- It does not prove a hard constraint is complete, correct, or legally
  sufficient.
- It does not guarantee future production performance, market behavior,
  provider behavior, data quality, or causal correctness.
- It does not prevent someone from editing JSON artifacts after they are
  written.
- It does not provide a cryptographic signature, trusted timestamp, key
  management, remote attestation, or immutable storage layer.
- It does not imply formal verification.
- It does not publish packages, create tags, or create GitHub releases.

## Optimizer Output Is Untrusted Until Audited

Raw optimizer output is treated as an untrusted proposal because a high score
can be caused by overfitting, sparse trials, invalid parameter combinations, or
constraint violations. Omega-Lock therefore records both the scoring path and
the gate path:

1. Evaluate baseline and candidates.
2. Record each audited call.
3. Apply hard constraints and advisory checks.
4. Preserve absolute-best evidence as `best_any`.
5. Preserve feasible-best evidence as `best_feasible`.
6. Apply walk-forward or holdout gates when configured.
7. Emit artifacts that reviewers and CI can inspect.

The strongest local contract is not "the optimizer is right." The contract is
"the accepted candidate is the one the configured audit gate accepted, and the
artifact exposes the evidence used for that decision."

## Hard Constraints vs Advisory Checks

Hard constraints are caller-declared blocking predicates. When
`constraint_policy="prefer_feasible"` is used, feasible candidates are preferred
over infeasible candidates even when the infeasible candidate has a higher raw
fitness. When `constraint_policy="hard_fail"` is used, no feasible candidate is
a blocking outcome.

Advisory checks remain visible but are not the same as hard constraints. They
can flag weak evidence, sparse trials, or diagnostic issues without replacing
the caller's hard constraints. Tests for this boundary include
[tests/test_constraint_aware_selection.py](../tests/test_constraint_aware_selection.py),
[tests/test_holdout_gate_mode.py](../tests/test_holdout_gate_mode.py), and
[tests/test_kill_criteria.py](../tests/test_kill_criteria.py).

## Feasible-Best vs Absolute-Best

`best_any` answers: which candidate had the highest raw fitness?

`best_feasible` answers: which candidate had the highest raw fitness among
candidates that satisfied declared hard constraints?

These fields are intentionally not collapsed. If they differ, the report should
make the tradeoff visible rather than silently certifying the raw winner. The
boundary is covered by claim ledger `feasible_best_vs_absolute_best`,
[tests/test_audit.py](../tests/test_audit.py), and
[tests/test_constraint_aware_selection.py](../tests/test_constraint_aware_selection.py).

## Walk-Forward Validation Gate

Walk-forward validation re-evaluates selected train candidates against a caller
provided test target and computes configured evidence such as Pearson
correlation and trade-ratio checks. This can catch train-only winners that do
not transfer to the test target.

The gate is only as meaningful as the target split and threshold choices. It is
not a proof of future generalization. Relevant evidence includes claim ledger
`walk_forward_validation`,
[src/omega_lock/walk_forward.py](../src/omega_lock/walk_forward.py),
[tests/test_walk_forward_cache.py](../tests/test_walk_forward_cache.py), and
[tests/test_holdout_gate_mode.py](../tests/test_holdout_gate_mode.py).

## Append-Only Audit Trail

During execution, `AuditingTarget` appends an `AuditedRun` for each audited
evaluation. Each run carries parameters, fitness, metadata, phase, target role,
round index, and call index. This supports reviewability and replay of the
decision surface.

After serialization, normal files remain editable. The append-only trail is an
in-process construction invariant, not a storage-system immutability guarantee.
Use hash-chain verification when post-write mutation detection is needed.

## Optional Hash-Chain and Tamper Detection Boundaries

Hash-chain output is optional. By default, JSON serialization omits it. When
`to_json(with_hash_chain=True)` or `to_dict(with_hash_chain=True)` is used, each
chain entry records the current run hash and previous hash. Verification
recomputes the chain from the loaded report and compares it to the supplied
chain.

This detects many post-write mutations when verification is actually performed:
changed run params, changed fitness, changed metadata, deleted runs, reordered
runs, duplicated runs, changed `previous_hash`, changed `run_hash`, and changed
call index are covered by [tests/test_audit_hash_chain.py](../tests/test_audit_hash_chain.py).

This does not make an artifact immune to tampering. A user who can edit both the
payload and its hash chain can create a new internally consistent artifact
unless an external trust anchor preserves the original chain or file digest.

## Schema Validation Boundaries

Audit reports include a schema version, and `AuditReport.from_json` rejects
unsupported audit report schema versions. This catches accidental or unsupported
schema mismatch for the current audit report format.

This is not full formal schema verification for every artifact in the
repository and not a historical migration policy. Tests and fixtures:
[tests/test_audit_hash_chain.py](../tests/test_audit_hash_chain.py),
[tests/test_artifact_reproducibility.py](../tests/test_artifact_reproducibility.py),
and [tests/fixtures/golden_audits/schema_validation_roundtrip.json](../tests/fixtures/golden_audits/schema_validation_roundtrip.json).

## Deterministic Offline Validation Priority

Default validation favors deterministic offline checks:

- `python scripts/check_repo_consistency.py --check`
- `python scripts/generate_readme_claims.py --check`
- `python scripts/run_golden_audit_cases.py --check`
- `python examples/demo_replay.py --check`
- `python examples/demo_sram.py --check`
- `python -m pytest -q`

Offline validation is preferred because it is reproducible, inspectable, and
does not depend on provider uptime, registry availability, API keys, network
policy, or live data drift. This policy is reflected in
[.github/workflows/quality-ci.yml](../.github/workflows/quality-ci.yml),
[scripts/publish_readiness.py](../scripts/publish_readiness.py), and claim
ledger `deterministic_offline_demos`.

## Why Live Provider/API Tests Are Not Default CI

Live provider/API tests are excluded from default CI because they introduce
network dependence, credentials, provider-side nondeterminism, quota failures,
and temporal drift. A default CI failure should identify a repository problem,
not a registry outage or provider behavior change.

Network checks belong in explicit verification scripts such as
[scripts/post_release_verify.py](../scripts/post_release_verify.py), where
blocked access is reported as `ENVIRONMENT_BLOCKED` and never treated as
approval.

## TOOLING_MISSING and ENVIRONMENT_BLOCKED

`TOOLING_MISSING` means a required local tool or Python module is unavailable.
Examples include missing `python`, `pytest`, `pyright`, `ruff`, `build`,
`twine`, `venv`, or `pip`. This is not success and not release approval.

`ENVIRONMENT_BLOCKED` means required registry, GitHub, PyPI, network, or
environment access was blocked. This is also not success and not release
approval.

Scripts that surface these states include:

- [scripts/check_repo_consistency.py](../scripts/check_repo_consistency.py)
- [scripts/release_audit.py](../scripts/release_audit.py)
- [scripts/wheel_smoke_install.py](../scripts/wheel_smoke_install.py)
- [scripts/publish_readiness.py](../scripts/publish_readiness.py)
- [scripts/post_release_verify.py](../scripts/post_release_verify.py)

## Release Approval Boundaries

No local document, version string, git tag, dist artifact, or offline check is
by itself proof that a PyPI release exists or is correct. Release readiness and
post-release verification are separate:

- `scripts/publish_readiness.py` aggregates local pre-publish checks and never
  publishes.
- `scripts/release_audit.py` checks local release surfaces and marks offline
  registry/GitHub status as non-approval.
- `scripts/wheel_smoke_install.py` installs only a locally built wheel and does
  not query PyPI.
- `scripts/post_release_verify.py` is for explicit post-release PyPI
  verification and reports blocked network access as `ENVIRONMENT_BLOCKED`.

Release approval requires all required checks to pass in an environment with
the required tooling and, where applicable, explicit registry verification.
`TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are blockers, not approvals.
