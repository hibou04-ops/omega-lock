# Changelog

This changelog records local repository release notes only. It is not PyPI
publication proof, GitHub release proof, or release approval.

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
