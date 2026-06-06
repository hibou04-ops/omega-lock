# Release Candidate Scope Freeze

This document records the local release-candidate marker used by
`scripts/scope_freeze_check.py`. It is a reversible scope-control aid, not a
tag, release, publish instruction, or approval.

```text
release_candidate_marker: RC_MARKER_UNSET
release_audit_after_commit: RC_AUDIT_UNSET
release_audit_command: python scripts/release_audit.py --intended-version 0.3.0 --offline --json
```

## What The Marker Means

- `release_candidate_marker` is the local git commit or ref used as the start of
  the release-candidate freeze window.
- `release_audit_after_commit` is the local commit or ref after which the
  offline release audit was last run.
- `RC_MARKER_UNSET` means no freeze is established yet; checks are advisory
  except for deterministic drift checks.
- `RC_AUDIT_UNSET` means release audit freshness has not been recorded.

To establish a candidate, edit this file in a normal PR or local commit with a
specific commit SHA or branch ref. Moving the marker is allowed when the scope
intentionally changes; the move should be visible in review.

## What The Check Enforces

`scripts/scope_freeze_check.py --check` runs offline and reports:

- files changed since the release-candidate marker
- whether code changed without generated artifact updates
- whether generated README claim files are stale
- whether golden audit artifacts are stale
- whether the recorded release audit ref is at or after the latest relevant
  repository change

The check never publishes to PyPI, creates or pushes git tags, creates GitHub
releases, or queries the network.

## Expected Release-Candidate Flow

1. Finish the intended code, tests, docs, and generated artifacts.
2. Run deterministic local checks, including generated claims and golden audit
   cases.
3. Run the offline release audit command recorded above.
4. Record the release-candidate marker and release-audit ref in this file.
5. Run `python scripts/scope_freeze_check.py --check`.

`TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are not approval states. If git
metadata is unavailable, the freeze check reports `TOOLING_MISSING` because it
cannot establish what changed after the marker.

## Scope Changes After A Candidate

A release-candidate freeze is not irreversible. If a fix is required:

1. make the fix;
2. regenerate any affected docs or deterministic artifacts;
3. rerun the offline release audit;
4. update the marker/ref in this file; and
5. rerun the scope-freeze check.
