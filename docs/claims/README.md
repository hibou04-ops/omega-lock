# README Claims Framework

This directory tracks public claims made in `README.md` without adding live
network checks or release approval.

## Files

- `public_claims.yml`: source ledger. It is JSON-compatible YAML so the
  generator can use only the Python standard library.
- `generated_readme_claims.md`: generated human-readable review table.
- `generated_readme_claims.json`: generated deterministic machine-readable
  artifact.

## Claim Classes

Each claim must use exactly one primary classification:

- `source_of_truth`: backed by local source, tests, package metadata, or code.
- `generated_doc`: backed by a generated local document.
- `reproducible_command`: backed by an offline command a reviewer can run.
- `deterministic_artifact`: backed by a checked-in deterministic artifact.
- `qualitative_marker`: positioning or scope language that is not promoted to
  a quantitative proof claim.

Claims without proof must stay `qualitative_marker` with `status` set to
`qualitative` or `todo`. Do not mark unproven claims as source-of-truth claims.

## Badge And Download Wording

Downloads and stars may be visibility signals, not proof of correctness,
trustworthiness, package quality, or release readiness. Static badges must not
imply release readiness unless a release audit backs that statement.

Do not add PyPI, GitHub, or third-party download analytics when this repository
has no checked source for them. If a README badge or download statement is
public-facing, represent it in `public_claims.yml` and keep unsupported wording
as `qualitative_marker`.

## Workflow

Regenerate after changing README claim text or claim proof paths:

```bash
python scripts/generate_readme_claims.py
```

Check for drift:

```bash
python scripts/generate_readme_claims.py --check
```

The generator is offline by default. It validates proof paths and records
commands, but it does not call PyPI, GitHub, provider APIs, or the network.
