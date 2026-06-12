# Examples Gallery

This gallery lists deterministic, offline examples that can be run from the
repository root. It is a usage index only: it does not publish to PyPI, create
tags, create GitHub releases, call provider APIs, or imply live provider
coverage.

Public claim links point to `docs/claims/public_claims.yml` claim IDs. Claims
with no direct ledger backing are marked `none`.

## Summary

| Example | Purpose | Exact command | Expected high-level output | Self-checking | Audit artifact | Public claim support |
| --- | --- | --- | --- | --- | --- | --- |
| Rosenbrock simple objective | Minimal `CalibrableTarget` over a 2D deterministic objective. | `python examples/rosenbrock_demo.py` | P1 summary, KC rows, best point near `(1, 1)`, and `Rosenbrock demo PASSED.` | Yes, via embedded assertions; no `--check` mode. | Yes: `output/rosenbrock_run.json`. | `package_naming_install` for import/package usage; otherwise illustrative. |
| Phantom synthetic optimizer audit | Synthetic black-box optimizer audit with decoys, walk-forward validation, hybrid validation, and iterative zoom. | `python examples/phantom_demo.py` | P1 PASS summary, stress top-3, walk-forward Pearson/trade ratio, KC reports, fractal-vise summary, and `PhantomKeyhole demo PASSED.` | Yes, via embedded assertions; no `--check` mode. | Yes: `output/phantom_run.json` and `output/phantom_fractal_run.json`. | `walk_forward_validation`, `deterministic_offline_demos`. |
| Benchmark battery | Runs benchmark scorecards over deterministic synthetic keyholes and methods. | `python examples/benchmark_battery.py` | Per-keyhole and combined scorecards, then `Full report JSON: output/benchmark_report.json`. | No dedicated `--check` mode; regression proof lives in tests/fixtures. | Yes: `output/benchmark_report.json`. | `benchmark_scorecard`, `stress_rank_spearman`. |
| Adapter example | Shows `CallableAdapter` and a custom stateful `CalibrableTarget` wrapper. | `python examples/adapter_example.py` | Two adapter pattern summaries and `Adapter example complete.` | No dedicated `--check` mode. | No persistent audit artifact. | `package_naming_install` for import/package usage. |
| Walk-forward gate case study | Deterministic overfitting case study: naive `best_any` chases slice noise and collapses on holdout; the KC-4 walk-forward gate fails the run; the constraint-gated `best_feasible` transfers. | `python examples/walkforward_gate_demo.py` | Narrative with numbers: `FAIL:KC-4` pipeline status, train-vs-holdout scoreboard, and `Walk-forward gate demo PASSED.` | Yes, via embedded assertions; no `--check` mode. | No persistent audit artifact (in-memory audit trail only). | `walk_forward_validation`, `hard_constraint_compliance`, `feasible_best_vs_absolute_best`, `deterministic_offline_demos`. |
| Optuna bridge | Gates an existing Optuna study's completed trials via the `audit_optuna_study` API (walk-forward gate + `best_any`/`best_feasible` split from `user_attrs["feasible"]` flags) and writes an HTML scorecard. Skips cleanly when optuna is not installed. | `python examples/optuna_audit_demo.py` | KC-4 `FAIL` verdict on the noise-chasing study ranking, feasible-vs-any holdout scoreboard, and `Optuna bridge demo PASSED.` | Yes, via embedded assertions; no `--check` mode. | Yes: `output/optuna_audit_scorecard.html`. | `walk_forward_validation`, `feasible_best_vs_absolute_best`, `optuna_study_audit_bridge`, `html_scorecard_stdlib`. |
| Demo replay | Replays the checked-in Phantom demo capture at recording pace, or checks the capture deterministically. | `python examples/demo_replay.py --check` | Stable JSON summary with marker booleans and `"status": "PASS"`. | Yes: `--check`. | No new artifact; reads `examples/_demo_output.txt`. | `deterministic_offline_demos`, `walk_forward_validation`. |
| SRAM demo case | Runs an offline SRAM bitcell surrogate through `run_p1` and audit reporting. | `python examples/demo_sram.py --check` | Stable JSON summary with best-feasible markers, constraint counts, schema roundtrip, and hash-chain verification. | Yes: `--check`. | `--check` does not write an artifact. Full demo command `python examples/demo_sram.py` writes `output/audit_sram.json`. | `hard_constraint_compliance`, `feasible_best_vs_absolute_best`, `append_only_audit_trail`, `sha256_hash_chain_tamper_detection`, `deterministic_offline_demos`. |

## Notes

- Commands assume the repository root as the working directory.
- The examples are offline and deterministic by construction where fixed seeds
  or checked-in captures are used.
- `benchmark_battery.py` prints mechanically computed metrics. This gallery
  does not claim external benchmark superiority; the ledger-backed claims are
  limited to the existence and reproducibility of the scorecard/regression
  surfaces.
- `demo_replay.py --check` and `demo_sram.py --check` are the self-checking
  demo paths intended for lightweight CI or release-readiness gates.
- Since 0.3.4 the package installs one console command via
  `[project.scripts]`: `omega-lock` (`demo`, `gate`, `report` subcommands).
  `omega-lock demo` prints the same narrative as
  `python examples/walkforward_gate_demo.py`; the `python examples/...`
  commands above continue to work from the source tree. There is still no
  `omega-lock diff` command.

## Claim Ledger Links

The claim IDs referenced above are maintained in
`docs/claims/public_claims.yml` and rendered in
`docs/claims/generated_readme_claims.md`:

- `append_only_audit_trail`
- `benchmark_scorecard`
- `deterministic_offline_demos`
- `feasible_best_vs_absolute_best`
- `hard_constraint_compliance`
- `html_scorecard_stdlib`
- `optuna_study_audit_bridge`
- `package_naming_install`
- `sha256_hash_chain_tamper_detection`
- `stress_rank_spearman`
- `walk_forward_validation`
