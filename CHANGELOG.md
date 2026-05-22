# Changelog

This changelog records local repository release notes only. It is not PyPI
publication proof, GitHub release proof, or release approval.

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
