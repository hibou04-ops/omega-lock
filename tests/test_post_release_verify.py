# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
from __future__ import annotations

from email.message import Message
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Protocol, Sequence


def _load_post_release_verify() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "post_release_verify.py"
    spec = importlib.util.spec_from_file_location("post_release_verify", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load_post_release_verify()


class VerifyResultLike(Protocol):
    name: str
    status: str
    message: str
    details: tuple[str, ...]


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _pypi_payload(
    *,
    version: str = "1.2.3",
    name: str = "omega-lock",
    include_wheel: bool = True,
    include_sdist: bool = True,
    yanked: bool = False,
) -> dict[str, object]:
    urls: list[dict[str, object]] = []
    if include_wheel:
        urls.append(
            {
                "filename": f"omega_lock-{version}-py3-none-any.whl",
                "packagetype": "bdist_wheel",
                "yanked": yanked,
            }
        )
    if include_sdist:
        urls.append(
            {
                "filename": f"omega_lock-{version}.tar.gz",
                "packagetype": "sdist",
                "yanked": yanked,
            }
        )
    return {
        "info": {"name": name, "version": version},
        "urls": urls,
    }


def _install_probe(version: str = "1.2.3") -> dict[str, object]:
    return {
        "metadata_name": "omega-lock",
        "metadata_version": version,
        "import_package": "omega_lock",
        "import_version": version,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_readme(root: Path, version: str) -> None:
    _write(
        root / "README.md",
        f"""
        # Omega-Lock

        [![Release](https://img.shields.io/badge/release-{version}-orange.svg)](https://pypi.org/project/omega-lock/{version}/)

        ```bash
        pip install omega-lock=={version}
        ```
        """,
    )


def _write_fixture(root: Path, payload: dict[str, object], install_probe: dict[str, object] | None) -> Path:
    fixture = root / "fixture.json"
    data: dict[str, object] = {"pypi_json": payload}
    if install_probe is not None:
        data["install_probe"] = install_probe
    fixture.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return fixture


def _by_name(results: Sequence[VerifyResultLike]) -> dict[str, VerifyResultLike]:
    return {result.name: result for result in results}


def test_offline_fixture_success_path_verifies_release_without_network(tmp_path: Path):
    _write_readme(tmp_path, "1.2.3")
    fixture = _write_fixture(tmp_path, _pypi_payload(), _install_probe())

    results = VERIFY.run_post_release_verify(
        tmp_path,
        version="1.2.3",
        distribution="omega-lock",
        offline_fixture=fixture,
    )
    by_name = _by_name(results)

    assert VERIFY.approved(results)
    assert by_name["pypi-metadata-name"].status == "PASS"
    assert by_name["pypi-wheel"].status == "PASS"
    assert by_name["pypi-sdist"].status == "PASS"
    assert by_name["installed-import"].status == "PASS"


def test_fetch_pypi_json_uses_mocked_network_response():
    def opener(request, timeout):
        assert "https://pypi.org/pypi/omega-lock/1.2.3/json" == request.full_url
        assert timeout == 10.0
        return _FakeResponse(_pypi_payload())

    payload, result = VERIFY.fetch_pypi_json("omega-lock", "1.2.3", opener=opener)

    assert result.status == "PASS"
    assert payload is not None
    assert payload["info"]["version"] == "1.2.3"


def test_fetch_pypi_json_distinguishes_unreleased_404():
    def opener(_request, _timeout):
        raise urllib.error.HTTPError(
            url="https://pypi.org/pypi/omega-lock/9.9.9/json",
            code=404,
            msg="Not Found",
            hdrs=Message(),
            fp=None,
        )

    payload, result = VERIFY.fetch_pypi_json("omega-lock", "9.9.9", opener=opener)

    assert payload is None
    assert result.status == "FAIL"
    assert "UNRELEASED" in result.details


def test_fetch_pypi_json_distinguishes_network_blocked():
    def opener(_request, _timeout):
        raise urllib.error.URLError("network blocked")

    payload, result = VERIFY.fetch_pypi_json("omega-lock", "1.2.3", opener=opener)

    assert payload is None
    assert result.status == "ENVIRONMENT_BLOCKED"
    assert "not approval" in result.message


def test_pypi_payload_distinguishes_yanked_release():
    results = VERIFY.verify_pypi_payload(
        _pypi_payload(yanked=True),
        distribution="omega-lock",
        version="1.2.3",
    )
    by_name = _by_name(results)

    assert by_name["pypi-yanked"].status == "FAIL"
    assert "omega_lock-1.2.3-py3-none-any.whl" in by_name["pypi-yanked"].details


def test_pypi_payload_distinguishes_missing_wheel():
    results = VERIFY.verify_pypi_payload(
        _pypi_payload(include_wheel=False),
        distribution="omega-lock",
        version="1.2.3",
    )
    by_name = _by_name(results)

    assert by_name["pypi-wheel"].status == "FAIL"
    assert "MISSING_WHEEL" in by_name["pypi-wheel"].details
    assert by_name["pypi-sdist"].status == "PASS"


def test_validate_install_probe_catches_import_version_mismatch():
    probe = _install_probe(version="1.2.2")

    results = VERIFY.validate_install_probe(
        probe,
        distribution="omega-lock",
        version="1.2.3",
    )
    by_name = _by_name(results)

    assert by_name["installed-metadata-version"].status == "FAIL"
    assert by_name["installed-import"].status == "FAIL"


def test_readme_version_drift_fails_against_pypi_release(tmp_path: Path):
    _write_readme(tmp_path, "1.2.2")

    result = VERIFY.check_readme_version(tmp_path, "omega-lock", "1.2.3")

    assert result.status == "FAIL"
    assert "README.md" in result.details[0]


def test_install_command_uses_pypi_spec_without_publish_actions(tmp_path: Path):
    command = VERIFY.install_command(tmp_path / "venv" / "Scripts" / "python.exe", "omega-lock", "1.2.3")
    flattened = " ".join(command)

    assert "omega-lock==1.2.3" in command
    assert "--no-index" not in command
    assert "twine upload" not in flattened
    assert "git tag" not in flattened
    assert "gh release" not in flattened


def test_json_payload_is_stable_and_nonapproval_when_blocked(tmp_path: Path):
    results = [
        VERIFY.VerifyResult("pypi-release", "ENVIRONMENT_BLOCKED", "network blocked"),
    ]

    payload = VERIFY.to_payload(
        results,
        version="1.2.3",
        distribution="omega-lock",
        offline_fixture=None,
        root=tmp_path,
    )
    rendered_once = VERIFY.render_json(payload)
    rendered_twice = VERIFY.render_json(payload)
    parsed = json.loads(rendered_once)

    assert rendered_once == rendered_twice
    assert parsed["approved"] is False
    assert parsed["summary"]["ENVIRONMENT_BLOCKED"] == 1
