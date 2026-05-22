# Changelog

This changelog records local repository release notes only. It is not PyPI
publication proof, GitHub release proof, or release approval.

## 0.2.6

- Local package version surfaces use `0.2.6`.
- Fixed `scripts/post_release_verify.py` PyPI JSON fetch timeout handling so
  the timeout is passed to `urllib` as `timeout`, not as request data.
- Preserved the injected opener test interface as `(request, timeout)`.
- Registry publication status is not asserted here.
- Release preparation and verification remain governed by `RELEASE.md` and the
  offline release audit scripts.
