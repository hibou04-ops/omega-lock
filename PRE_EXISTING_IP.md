# Pre-existing Intellectual Property Declaration

> **Purpose**: This document is a tamper-evident timestamped declaration that the
> work in this repository constitutes pre-existing personal intellectual property
> of the Primary Author, authored prior to and independent of any current or
> future employment relationship.

## Repository Identification

- **Repository**: [hibou04-ops/omega-lock](https://github.com/hibou04-ops/omega-lock)
- **License**: Apache License 2.0
- **Primary Author**: **Kyunghoon Gwak (곽경훈)** — operating as [@hibou04-ops](https://github.com/hibou04-ops)
  - Primary email: `hibouaile04@gmail.com` (verified, primary on the GitHub account; the Primary Author's only personal email)
  - Note: some commits in the git history before 2026-04-29 carry `hibou04@gmail.com` in the author email field. That address is **not** an account belonging to the Primary Author — it was an unintended local git client misconfiguration. The author *name* `Hibou04-ops` is the unambiguous identifier across the entire history. From 2026-04-29 onwards every repository in this toolkit is configured to commit under `hibouaile04@gmail.com`.

## Authorship Timeline (Tamper-Evident)

The following git artifacts establish the authorship timeline. The git commit graph
and the public GitHub remote (`github.com/hibou04-ops/omega-lock`) provide
independent timestamp witnesses.

| Anchor | Commit Hash | Date (KST) | Description |
|---|---|---|---|
| First commit | `20cf429b24f58c1ae17f9a0807df187463154e0c` | 2026-04-18 09:52:15 +0900 | Initial: Omega-Lock v0.1.0 — sensitivity-driven coordinate descent calibration framework |
| v0.1.4 release | `b135975` | 2026-04-26 (approx) | `omega_lock.audit` module + SRAM bitcell demo |
| Apache 2.0 relicense | `8a5e66d` | 2026-04-26 (approx) | MIT → Apache 2.0 for patent grant + trademark preservation |
| Pre-employment snapshot | (tagged on commit) | 2026-04-28 | This declaration committed; tagged `pre-employment-ip-snapshot-2026-04-28` |

## Scope of Pre-existing IP

The following work product is declared as pre-existing personal intellectual property:

1. **Calibration Framework**: The sensitivity-driven coordinate descent calibration
   methodology, including:
   - Keyhole detection (`omega_lock.keyholes.phantom`, `phantom_deep`)
   - Fractal vise refinement (the two-stage history of keyhole 80% + fractal
     vise 99% as distinct contributions)
   - Walk-forward validation (`omega_lock.walk_forward`)
   - Stress testing (`omega_lock.stress`)
   - Kill-criteria gating (`omega_lock.kill_criteria`)
2. **Audit Module**: All code under `src/omega_lock/audit/` —
   `_scorecard`, `_target`, `_types` and the audit orchestration logic.
3. **Optimizers and Search**: `random_search`, `p2_tpe`, `grid`, `orchestrator`.
4. **Adapters and Targets**: `adapters`, `target`, `params`, `fitness`, `benchmark`.
5. **Test Suite**: All materials under `tests/`.
6. **Examples and Demonstrations**: All materials under `examples/`,
   including the SRAM bitcell demo.
7. **Documentation**: README, README_KR, EASY_README, EASY_README_KR, NOTICE,
   methodology references.
8. **Specific Terminology and Application**: The compound term "Omega-Lock"
   as used to label this specific sensitivity-driven coordinate descent
   calibration framework (a coined compound term). Module names and internal
   API terminology as defined within `src/omega_lock/`, including
   "keyhole detection", "fractal vise", "phantom search", and the audit-module
   types. *No claim is made to the generic words "omega" or "lock" in
   isolation; the claim is to the specific compound and its application
   within this corpus.*

## Companion Methodology Repository

The Antemortem methodology applied during the development of `omega_lock.audit`
is published in the companion repository [hibou04-ops/Antemortem](https://github.com/hibou04-ops/Antemortem),
authored by the same Primary Author. See that repository's `PRE_EXISTING_IP.md`
for its own authorship binding.

## Development Conditions

This work was developed:

- Using **personal time** (outside of any third-party working hours)
- Using **personal equipment** (no employer-issued hardware)
- Using **personal accounts** (no employer-issued cloud, LLM, or API credentials)
- **Without reference** to any third party's confidential or proprietary information

## Use in Future Employment Agreements

This declaration is intended to be attached as a Schedule / Exhibit (commonly
"Schedule A: Pre-existing IP") to any future employment, contractor, or
consulting agreement, to clarify that:

- The work in this repository remains the personal property of the Primary Author.
- Future development on this codebase, conducted on personal time and outside the
  scope of any employment, continues to be the Primary Author's personal IP.
- Any contributions from a future employer's domain, made on employer time using
  employer resources, would be governed by the relevant employment agreement —
  the boundary is preserved by maintaining a separate repository, fork, or
  branch for any such employer-domain contributions.

## Verification

To independently verify this declaration:

1. Inspect git log:
   ```
   git log --format="%H | %ai | %an <%ae>" | grep "Hibou04-ops"
   ```
2. Confirm tag (when committed):
   ```
   git tag -l "pre-employment-ip-snapshot-*"
   git show pre-employment-ip-snapshot-2026-04-28
   ```
3. Cross-reference with public GitHub timestamps:
   - https://github.com/hibou04-ops/omega-lock/commit/20cf429b24f58c1ae17f9a0807df187463154e0c
   - https://github.com/hibou04-ops/omega-lock/releases

---

**Declaration date**: 2026-04-28
**License**: Apache License 2.0
**Document version**: 1.0
