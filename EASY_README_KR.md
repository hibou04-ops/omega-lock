# Omega-Lock 쉬운 설명

현재 로컬 패키지 버전: `0.2.6`.

[![Version 0.2.6](https://img.shields.io/badge/version-0.2.6-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

[Full README](README.md) · [한국어 README](README_KR.md) · [Easy README](EASY_README.md)

Omega-Lock은 배포 전 튜닝 후보를 검증합니다. 후보가 walk-forward validation,
선언된 hard constraints, 검토 가능한 append-only audit trail을 통과하는지
확인합니다.

정답을 채점하거나 correctness를 증명하지 않습니다. dashboard나 web app도
아니며, 현재 설치되는 console command도 없습니다. 특히 `omega-lock diff`
명령은 제공하지 않습니다.

## 핵심 아이디어

fitness가 가장 높은 후보가 항상 가장 안전한 후보는 아닙니다. hard constraint를
위반했다면 audit report는 그 후보를 `best_any`로 보여줄 수 있고,
`best_feasible`은 constraint를 만족하는 후보 중 fitness가 가장 높은 후보를
보여줍니다.

일반적인 audit/CI 실행에는 다음 설정을 권장합니다.

```python
P1Config(constraint_policy="prefer_feasible")
```

## 오프라인 데모 실행

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

두 데모는 deterministic이며 network/API key가 필요 없습니다. replay는 60초 영상과
같은 데모 흐름을 보여줍니다.

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

## 이름 구분

| 구분 | 이름 |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| 설치되는 console executable | 현재 없음 |

사용하는 package index에 `0.2.6`가 게시되어 있는 경우에만 PyPI로 설치하세요.

```bash
pip install omega-lock==0.2.6
pip install "omega-lock[p2]==0.2.6"
```

## 최소 사용 예시

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint("must_be_feasible", lambda params, result: result.metadata["sharpe"] > 0.5),
    ],
)

result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))
```

README claim의 근거는
[docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md)에
정리되어 있습니다.
