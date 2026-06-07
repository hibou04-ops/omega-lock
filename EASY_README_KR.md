# Omega-Lock 쉬운 설명

현재 로컬 패키지 버전: `0.3.1`.

[![Version 0.3.1](https://img.shields.io/badge/version-0.3.1-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

[Full README](README.md) · [한국어 README](README_KR.md) · [Easy README](EASY_README.md)

Omega-Lock은 배포 전 튜닝 후보를 검증합니다. 후보 생성 이후에 동작하며, 후보가
walk-forward validation, 선언된 hard constraints, 검토 가능한 append-only audit
trail을 통과하는지 확인합니다.

## 0.3.1에서 새로워진 점

내부 hardening만 포함합니다 — 사용 방법은 달라지지 않습니다. cross-repo
"docking" guard(contract manifest, `DOCKING.md`, 새 오프라인 presence-lint)를
강화하고, golden audit fixture를 새 버전 문자열에 맞게 재생성합니다. 공개 API나
runtime 동작 변경은 없습니다.

## 언제 쓰나

- 튜닝·캘리브레이션된 후보를 배포하기 전
- 최고 점수 후보가 hard constraint를 깨뜨릴 수 있을 때
- 검토자가 `best_any`와 `best_feasible`을 분리해서 봐야 할 때
- 오프라인에서 재현 가능한 audit artifact가 필요할 때

## 무엇을 검사하나

- 모든 후보에 대해 평가·기록되는 hard constraints
- `best_feasible`(constraint 만족) vs `best_any`(최고 raw 점수)
- test target이 설정된 경우 walk-forward / holdout 전이
- append-only JSON audit trail, 선택적 SHA-256 hash chain 증거
- `KCThresholds.pure_objective()`로 비-action 목적(수학, ML, 시뮬레이션) 지원

## 무엇을 증명하지 않나

- correctness나 root cause를 증명하지 않습니다
- 후보가 전역 최적해라고 증명하지 않습니다
- PyPI/GitHub 게시를 증명하지 않습니다 — registry 상태는 별도의 post-release 검증이 필요합니다
- dashboard나 web app이 아니며 설치되는 console 명령도 없습니다 — Omega-Lock은 `omega-lock diff` 명령을 제공하지 않습니다

## 핵심 아이디어

fitness가 가장 높은 후보가 항상 가장 안전한 후보는 아닙니다. hard constraint를
위반했다면 audit report는 그 후보를 `best_any`로 보여주고, `best_feasible`은
constraint를 만족하는 후보 중 fitness가 가장 높은 후보를 보여줍니다. 일반적인
audit/CI 실행에는 다음 설정을 권장합니다.

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

사용하는 package index에 `0.3.1`이 게시되어 있는 경우에만 PyPI로 설치하세요.

```bash
pip install omega-lock==0.3.1
pip install "omega-lock[p2]==0.3.1"
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
정리되어 있습니다. 신뢰 경계와 각 보증이 다루지 않는 범위는
[docs/TRUST_MODEL.md](docs/TRUST_MODEL.md)를 참고하세요.
