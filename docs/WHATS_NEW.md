# What's New

Per-release highlights, relocated from the README top so the README can lead
with the value proposition and quickstart. The authoritative history is
[CHANGELOG.md](../CHANGELOG.md); this page keeps only the short summaries.

## What's new in 0.3.4

All additive — no existing public symbol was renamed, moved, or re-defaulted:

- **Installed console command `omega-lock`** (`[project.scripts]`, argparse
  only): `omega-lock demo` runs the walk-forward gate case study;
  `omega-lock gate --train a.json --holdout b.json [--report out.html]`
  applies the KC-4 Pearson transfer gate to two JSON score arrays and exits
  0/1; `omega-lock report --input p1_result.json -o out.html` renders a saved
  result artifact. There is still no `omega-lock diff` command.
- **Optuna bridge API** `audit_optuna_study(study, holdout_evaluate=...)`:
  gates an existing study's completed trials through the reused
  `WalkForward` + `check_kc4` machinery and splits `best_any` vs
  `best_feasible` from per-trial `user_attrs["feasible"]` flags. The optuna
  import stays lazy; the `[p2]` extra is unchanged.
- **Single-file HTML scorecard** `render_html(result, path)`: stdlib-only,
  dark-theme, deterministic (no timestamps unless passed in) — verdict
  banner, `best_any` vs `best_feasible` table, stress ranking, inline SVG
  train-vs-holdout scatter.
- **Plain-language facade** `omega_lock.simple`: `gate_scores` (KC-4 over two
  plain score lists -> `GateVerdict(passed, pearson, reasons)`) and `audit`
  (a `CallableAdapter` + `run_p1` wrapper with friendly
  `{name: (low, high)}` specs).
- Golden audit fixtures carry the new version string only; the audit report
  schema and SHA-256 hash chain are unchanged.

## What's new in 0.3.3

Classifier promotion only — no functional change for existing users:

- `Development Status` promoted from `3 - Alpha` to `4 - Beta`. There is no
  functional code change since 0.3.2: the dormant, default-off parallel-execution
  executor seam and the sdist packaging fix that shipped in 0.3.2 both stand.
- Golden audit fixtures regenerated only to carry the new version string; the
  audit report schema and SHA-256 hash chain are unchanged.

## Earlier releases

See [CHANGELOG.md](../CHANGELOG.md) for 0.3.2 and earlier.

(The 0.3.3 summary above is retained verbatim; the authoritative history for
all releases is the changelog.)
