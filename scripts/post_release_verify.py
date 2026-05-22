#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Post-release PyPI verification for omega-lock.

This script verifies a release that should already exist on PyPI. It never
publishes, creates tags, or creates GitHub releases. Tests can use
--offline-fixture to avoid live network access.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
IMPORT_PACKAGE = "omega_lock"
STATUSES = ("PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED")
BLOCKING_STATUSES = frozenset({"FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"})
Status = Literal["PASS", "FAIL", "WARN", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]
UrlOpen = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class VerifyResult:
    name: str
    status: Status
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def _tail_lines(text: str, limit: int = 12) -> tuple[str, ...]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return tuple(lines[-limit:])


def _pypi_url(distribution: str, version: str) -> str:
    dist = urllib.parse.quote(distribution, safe="")
    ver = urllib.parse.quote(version, safe="")
    return f"https://pypi.org/pypi/{dist}/{ver}/json"


def _urlopen_with_timeout(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_pypi_json(
    distribution: str,
    version: str,
    *,
    opener: UrlOpen = _urlopen_with_timeout,
    timeout: float = 10.0,
) -> tuple[dict[str, Any] | None, VerifyResult]:
    url = _pypi_url(distribution, version)
    request = urllib.request.Request(url, headers={"User-Agent": "omega-lock-post-release-verify"})
    try:
        with opener(request, timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, VerifyResult(
                "pypi-release",
                "FAIL",
                "Requested release is not published on PyPI.",
                ("UNRELEASED", url),
            )
        return None, VerifyResult(
            "pypi-release",
            "ENVIRONMENT_BLOCKED",
            f"PyPI returned HTTP {exc.code}; post-release status is not approval.",
            (url,),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, VerifyResult(
            "pypi-release",
            "ENVIRONMENT_BLOCKED",
            "PyPI JSON could not be fetched; post-release status is not approval.",
            (url, str(exc)),
        )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, VerifyResult(
            "pypi-release",
            "FAIL",
            "PyPI JSON response could not be decoded.",
            (url, str(exc)),
        )
    return payload, VerifyResult("pypi-release", "PASS", "PyPI JSON response was fetched.", (url,))


def load_offline_fixture(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, VerifyResult]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None, VerifyResult(
            "offline-fixture",
            "FAIL",
            "Offline fixture file is missing.",
            (str(path),),
        )
    except json.JSONDecodeError as exc:
        return None, None, VerifyResult(
            "offline-fixture",
            "FAIL",
            "Offline fixture file is invalid JSON.",
            (str(path), str(exc)),
        )

    status = fixture.get("fixture_status")
    if status == "UNRELEASED":
        return None, None, VerifyResult(
            "pypi-release",
            "FAIL",
            "Offline fixture marks the release as unpublished.",
            ("UNRELEASED",),
        )
    if status == "ENVIRONMENT_BLOCKED":
        return None, None, VerifyResult(
            "pypi-release",
            "ENVIRONMENT_BLOCKED",
            "Offline fixture marks the PyPI check as environment-blocked.",
        )

    payload = fixture.get("pypi_json")
    install_probe = fixture.get("install_probe")
    if not isinstance(payload, dict):
        return None, None, VerifyResult(
            "offline-fixture",
            "FAIL",
            "Offline fixture must contain a pypi_json object.",
        )
    if install_probe is not None and not isinstance(install_probe, dict):
        return None, None, VerifyResult(
            "offline-fixture",
            "FAIL",
            "offline fixture install_probe must be an object when present.",
        )
    return payload, install_probe, VerifyResult(
        "offline-fixture",
        "PASS",
        "Loaded offline post-release fixture.",
        (str(path),),
    )


def verify_pypi_payload(
    payload: dict[str, Any],
    *,
    distribution: str,
    version: str,
) -> list[VerifyResult]:
    results: list[VerifyResult] = []
    info = payload.get("info", {})
    if not isinstance(info, dict):
        info = {}

    found_name = info.get("name")
    if found_name == distribution:
        results.append(VerifyResult("pypi-metadata-name", "PASS", f"PyPI metadata name is {distribution}."))
    else:
        results.append(
            VerifyResult(
                "pypi-metadata-name",
                "FAIL",
                "PyPI metadata name mismatch.",
                (f"expected: {distribution}", f"found: {found_name!r}"),
            )
        )

    found_version = info.get("version")
    if found_version == version:
        results.append(VerifyResult("pypi-metadata-version", "PASS", f"PyPI metadata version is {version}."))
    else:
        results.append(
            VerifyResult(
                "pypi-metadata-version",
                "FAIL",
                "PyPI metadata version mismatch.",
                (f"expected: {version}", f"found: {found_version!r}"),
            )
        )

    urls = payload.get("urls", [])
    if not isinstance(urls, list):
        urls = []
    files = [item for item in urls if isinstance(item, dict)]
    wheels = [item for item in files if item.get("packagetype") == "bdist_wheel"]
    sdists = [item for item in files if item.get("packagetype") == "sdist"]
    yanked = [str(item.get("filename", "<unknown>")) for item in files if item.get("yanked") is True]

    if wheels:
        results.append(
            VerifyResult(
                "pypi-wheel",
                "PASS",
                "PyPI release includes at least one wheel.",
                tuple(str(item.get("filename", "<unknown>")) for item in wheels),
            )
        )
    else:
        results.append(
            VerifyResult(
                "pypi-wheel",
                "FAIL",
                "PyPI release is missing a wheel artifact.",
                ("MISSING_WHEEL",),
            )
        )

    if sdists:
        results.append(
            VerifyResult(
                "pypi-sdist",
                "PASS",
                "PyPI release includes an sdist.",
                tuple(str(item.get("filename", "<unknown>")) for item in sdists),
            )
        )
    else:
        results.append(
            VerifyResult(
                "pypi-sdist",
                "FAIL",
                "PyPI release is missing an sdist artifact.",
                ("MISSING_SDIST",),
            )
        )

    if yanked:
        results.append(
            VerifyResult(
                "pypi-yanked",
                "FAIL",
                "PyPI release contains yanked artifact(s).",
                tuple(yanked),
            )
        )
    else:
        results.append(VerifyResult("pypi-yanked", "PASS", "PyPI release artifacts are not yanked."))
    return results


def install_command(venv_python: Path, distribution: str, version: str) -> list[str]:
    return [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        f"{distribution}=={version}",
    ]


def create_venv_command(venv_dir: Path) -> list[str]:
    return [sys.executable, "-m", "venv", str(venv_dir)]


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_command(args: Sequence[str], *, cwd: Path, timeout: int = 180) -> CommandResult | VerifyResult:
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        return VerifyResult(
            "subprocess",
            "TOOLING_MISSING",
            "Required executable is unavailable.",
            (str(exc),),
        )
    except subprocess.TimeoutExpired as exc:
        return VerifyResult(
            "subprocess",
            "FAIL",
            "Subprocess timed out.",
            (" ".join(args), str(exc)),
        )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def classify_nonzero(
    name: str,
    completed: CommandResult,
    *,
    fail_message: str,
) -> VerifyResult:
    output = completed.combined_output
    details = _tail_lines(output)
    lower = output.lower()
    network_markers = (
        "temporary failure in name resolution",
        "failed to establish a new connection",
        "connection refused",
        "connection timed out",
        "read timed out",
        "proxy",
        "ssl",
        "network is unreachable",
        "could not fetch url",
    )
    tooling_markers = ("no module named venv", "no module named pip", "ensurepip")
    if any(marker in lower for marker in network_markers):
        return VerifyResult(
            name,
            "ENVIRONMENT_BLOCKED",
            "PyPI install/import check could not complete because network or registry access was blocked.",
            details,
        )
    if any(marker in lower for marker in tooling_markers):
        return VerifyResult(name, "TOOLING_MISSING", f"{name} tooling is unavailable.", details)
    return VerifyResult(name, "FAIL", fail_message, details)


def smoke_probe_code(distribution: str) -> str:
    return f'''
import json
from importlib import metadata

payload = {{}}
try:
    dist = metadata.distribution({distribution!r})
    payload["metadata_name"] = dist.metadata.get("Name")
    payload["metadata_version"] = dist.version
except Exception as exc:
    payload["metadata_error"] = f"{{type(exc).__name__}}: {{exc}}"

try:
    import omega_lock
    payload["import_package"] = "omega_lock"
    payload["import_version"] = getattr(omega_lock, "__version__", None)
except Exception as exc:
    payload["import_error"] = f"{{type(exc).__name__}}: {{exc}}"

print(json.dumps(payload, sort_keys=True))
'''


def read_probe_payload(venv_python: Path, root: Path, distribution: str) -> tuple[VerifyResult, dict[str, Any] | None]:
    completed = run_command([str(venv_python), "-c", smoke_probe_code(distribution)], cwd=root, timeout=60)
    if isinstance(completed, VerifyResult):
        return completed, None
    if completed.returncode != 0:
        return classify_nonzero(
            "pypi-install-import",
            completed,
            fail_message="Installed PyPI package import probe failed.",
        ), None
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        return VerifyResult(
            "pypi-install-import",
            "FAIL",
            "Installed PyPI package import probe did not emit valid JSON.",
            (str(exc), completed.stdout.strip()[:500]),
        ), None
    return VerifyResult("pypi-install-import", "PASS", "Installed PyPI package import probe completed."), payload


def install_from_pypi(root: Path, distribution: str, version: str) -> tuple[list[VerifyResult], dict[str, Any] | None]:
    results: list[VerifyResult] = []
    with tempfile.TemporaryDirectory(prefix="omega-lock-post-release-") as temp:
        venv_dir = Path(temp) / "venv"
        created = run_command(create_venv_command(venv_dir), cwd=root, timeout=120)
        if isinstance(created, VerifyResult):
            results.append(created)
            return results, None
        if created.returncode != 0:
            results.append(
                classify_nonzero(
                    "create-venv",
                    created,
                    fail_message="Temporary virtual environment creation failed.",
                )
            )
            return results, None

        venv_python = venv_python_path(venv_dir)
        installed = run_command(install_command(venv_python, distribution, version), cwd=root, timeout=240)
        if isinstance(installed, VerifyResult):
            results.append(installed)
            return results, None
        if installed.returncode != 0:
            results.append(
                classify_nonzero(
                    "pypi-install",
                    installed,
                    fail_message="PyPI package install failed.",
                )
            )
            return results, None
        results.append(
            VerifyResult(
                "pypi-install",
                "PASS",
                f"Installed {distribution}=={version} from PyPI into a temporary environment.",
            )
        )
        probe_result, payload = read_probe_payload(venv_python, root, distribution)
        results.append(probe_result)
        return results, payload


def validate_install_probe(
    payload: dict[str, Any],
    *,
    distribution: str,
    version: str,
) -> list[VerifyResult]:
    results: list[VerifyResult] = []
    if payload.get("metadata_name") == distribution:
        results.append(VerifyResult("installed-metadata-name", "PASS", f"Installed metadata name is {distribution}."))
    else:
        results.append(
            VerifyResult(
                "installed-metadata-name",
                "FAIL",
                "Installed package metadata name mismatch.",
                (f"expected: {distribution}", f"found: {payload.get('metadata_name')!r}", str(payload.get("metadata_error", ""))),
            )
        )

    if payload.get("metadata_version") == version:
        results.append(VerifyResult("installed-metadata-version", "PASS", f"Installed metadata version is {version}."))
    else:
        results.append(
            VerifyResult(
                "installed-metadata-version",
                "FAIL",
                "Installed package metadata version mismatch.",
                (f"expected: {version}", f"found: {payload.get('metadata_version')!r}"),
            )
        )

    if payload.get("import_package") == IMPORT_PACKAGE and payload.get("import_version") == version:
        results.append(VerifyResult("installed-import", "PASS", f"{IMPORT_PACKAGE} imports with version {version}."))
    else:
        results.append(
            VerifyResult(
                "installed-import",
                "FAIL",
                "Installed package import/version mismatch.",
                (
                    f"expected package: {IMPORT_PACKAGE}",
                    f"expected version: {version}",
                    f"found package: {payload.get('import_package')!r}",
                    f"found version: {payload.get('import_version')!r}",
                    str(payload.get("import_error", "")),
                ),
            )
        )
    return results


def check_readme_version(root: Path, distribution: str, version: str) -> VerifyResult:
    readme = root / "README.md"
    if not readme.exists():
        return VerifyResult("readme-version", "WARN", "README.md is absent; public README version drift was not checked.")
    text = readme.read_text(encoding="utf-8")
    patterns = (
        re.compile(r"(?:release|version)-(\d+\.\d+\.\d+)"),
        re.compile(rf"pypi\.org/project/{re.escape(distribution)}/(\d+\.\d+\.\d+)/"),
        re.compile(rf"{re.escape(distribution)}(?:\[[^\]]+\])?==(\d+\.\d+\.\d+)"),
    )
    stale: list[str] = []
    found_any = False
    for line_no, line in enumerate(text.splitlines(), 1):
        for pattern in patterns:
            for match in pattern.finditer(line):
                found_any = True
                found = match.group(1)
                if found != version:
                    stale.append(
                        f"README.md:{line_no}: expected {version}, found {found}: {line.strip()[:140]}"
                    )
    if stale:
        return VerifyResult(
            "readme-version",
            "FAIL",
            "Public README release/version surface drifts from the PyPI release.",
            tuple(stale),
        )
    if not found_any:
        return VerifyResult(
            "readme-version",
            "WARN",
            "README.md has no static release badge, PyPI version URL, or exact install pin to compare.",
        )
    return VerifyResult("readme-version", "PASS", f"Public README version surfaces match PyPI release {version}.")


def has_blocking_status(results: Sequence[VerifyResult]) -> bool:
    return any(result.status in BLOCKING_STATUSES for result in results)


def approved(results: Sequence[VerifyResult]) -> bool:
    return all(result.status == "PASS" for result in results)


def summarize(results: Sequence[VerifyResult]) -> dict[str, int]:
    return {status: sum(1 for result in results if result.status == status) for status in STATUSES}


def run_post_release_verify(
    root: Path,
    *,
    version: str,
    distribution: str,
    offline_fixture: Path | None = None,
) -> list[VerifyResult]:
    root = root.resolve()
    results: list[VerifyResult] = []

    if offline_fixture is not None:
        pypi_payload, install_probe, fixture_result = load_offline_fixture(offline_fixture)
        results.append(fixture_result)
    else:
        pypi_payload, fetch_result = fetch_pypi_json(distribution, version)
        install_probe = None
        results.append(fetch_result)

    if pypi_payload is None:
        results.append(check_readme_version(root, distribution, version))
        return results

    results.extend(verify_pypi_payload(pypi_payload, distribution=distribution, version=version))
    results.append(check_readme_version(root, distribution, version))
    if has_blocking_status(results):
        return results

    if offline_fixture is not None:
        if install_probe is None:
            results.append(
                VerifyResult(
                    "pypi-install",
                    "FAIL",
                    "Offline fixture is missing install_probe; PyPI install/import cannot be verified.",
                )
            )
        else:
            results.append(
                VerifyResult(
                    "pypi-install",
                    "PASS",
                    "Offline fixture provides mocked PyPI install/import probe.",
                )
            )
            results.extend(validate_install_probe(install_probe, distribution=distribution, version=version))
        return results

    install_results, probe_payload = install_from_pypi(root, distribution, version)
    results.extend(install_results)
    if probe_payload is not None:
        results.extend(validate_install_probe(probe_payload, distribution=distribution, version=version))
    return results


def to_payload(
    results: Sequence[VerifyResult],
    *,
    version: str,
    distribution: str,
    offline_fixture: Path | None,
    root: Path,
) -> dict[str, Any]:
    return {
        "approved": approved(results),
        "distribution": distribution,
        "mode": "offline_fixture" if offline_fixture is not None else "network",
        "results": [
            {
                "details": list(result.details),
                "message": result.message,
                "name": result.name,
                "status": result.status,
            }
            for result in results
        ],
        "root": str(root.resolve()),
        "schema_version": 1,
        "summary": summarize(results),
        "version": version,
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_text(
    results: Sequence[VerifyResult],
    *,
    version: str,
    distribution: str,
    offline_fixture: Path | None,
    root: Path,
) -> str:
    decision = "PASS" if approved(results) else "BLOCKED"
    lines = [
        "Post-release verification",
        f"Root: {root.resolve()}",
        f"Distribution: {distribution}",
        f"Version: {version}",
        f"Mode: {'offline_fixture' if offline_fixture is not None else 'network'}",
        f"Decision: {decision}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status}] {result.name}: {result.message}")
        for detail in result.details[:3]:
            lines.append(f"  - {detail}")
        if len(result.details) > 3:
            lines.append(f"  - ... {len(result.details) - 3} more line(s)")
    counts = summarize(results)
    lines.extend(
        [
            "",
            "Summary: " + ", ".join(f"{status}={count}" for status, count in counts.items() if count),
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="released version to verify")
    parser.add_argument("--distribution", required=True, help="PyPI distribution name, e.g. omega-lock")
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    parser.add_argument("--offline-fixture", type=Path, help="fixture JSON for offline tests")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = run_post_release_verify(
        args.root,
        version=args.version,
        distribution=args.distribution,
        offline_fixture=args.offline_fixture,
    )
    if args.json:
        print(
            render_json(
                to_payload(
                    results,
                    version=args.version,
                    distribution=args.distribution,
                    offline_fixture=args.offline_fixture,
                    root=args.root,
                )
            ),
            end="",
        )
    else:
        print(
            render_text(
                results,
                version=args.version,
                distribution=args.distribution,
                offline_fixture=args.offline_fixture,
                root=args.root,
            )
        )
    return 0 if approved(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
