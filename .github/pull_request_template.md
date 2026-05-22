## Summary

Describe the change and the smallest review context needed.

For trivial docs-only changes, mark non-applicable audit items as `N/A` with a
short reason.

## Trust Surface Checklist

- Affected audit invariant:
  - [ ] `N/A`
  - [ ] hard constraint semantics
  - [ ] walk-forward / holdout gate behavior
  - [ ] append-only audit trail
  - [ ] schema validation
  - [ ] hash-chain / tamper detection
  - [ ] release audit / repository consistency tooling
  - [ ] README / public claims / claim ledger
- Hard constraints are affected:
  - [ ] no
  - [ ] yes, tests included
  - [ ] unknown, reviewer attention requested
- Walk-forward gate behavior changes:
  - [ ] no
  - [ ] yes, tests included
  - [ ] unknown, reviewer attention requested
- Schema, artifact, or hash-chain behavior changes:
  - [ ] no
  - [ ] yes, tests included
  - [ ] unknown, reviewer attention requested
- README or public claims need claim ledger updates:
  - [ ] no
  - [ ] yes, `docs/claims/public_claims.yml` and generated claim files updated
  - [ ] unknown, reviewer attention requested
- Live provider/API behavior is involved:
  - [ ] no
  - [ ] yes, not part of default CI

## Exact Verification Commands Run

List exact commands and statuses. Use `TOOLING_MISSING` for missing local tools
and `ENVIRONMENT_BLOCKED` for blocked registry, GitHub, PyPI, provider, or
network checks.

```text
command:
status:
notes:
```

## Release-Safety Checklist

- [ ] This PR does not publish to PyPI.
- [ ] This PR does not create or push git tags.
- [ ] This PR does not create GitHub releases.
- [ ] This PR does not make the publish workflow easier or more permissive.
- [ ] This PR does not add live API/provider/network tests to default CI.
- [ ] This PR does not weaken hard constraints, audit invariants, walk-forward
      gates, append-only audit trails, schema validation, or hash-chain
      tamper-detection behavior.
- [ ] New or changed public claims are backed by the claim ledger or are marked
      qualitative/TODO.
- [ ] Any CLI executable change is intentional and reflected in `pyproject.toml`
      and repository-surface docs.
