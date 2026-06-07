# Omega-Lock

> 배포 전 튜닝 후보를 검증합니다: walk-forward validation, 선언형 hard
> constraints, feasible-best 선택, append-only JSON 감사 추적.

Omega-Lock은 **후보 생성 이후**에 동작합니다. search·tuning·calibration 방법이
후보를 제안하면, Omega-Lock은 그 후보가 배포되기 전에 사전에 선언한 evidence
gate를 통과하는지 판단합니다.

[![Version 0.3.3](https://img.shields.io/badge/version-0.3.3-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

**README 종류:** [Full README](README.md) · [한국어 README](README_KR.md) ·
[Easy README](EASY_README.md) · [쉬운 한국어 README](EASY_README_KR.md)

현재 로컬 패키지 버전: `0.3.3`. 이 문서는 PyPI 또는 GitHub Release 게시 여부를
주장하지 않습니다. 로컬 버전 메타데이터는 registry 게시의 증거가 아니며,
registry 상태는 별도의 post-release 검증을 거쳐야 합니다.

## 0.3.3에서 새로워진 점

classifier 승격뿐 — 기존 사용자에게 기능 변경은 없습니다.

- `Development Status`가 `3 - Alpha`에서 `4 - Beta`로 승격되었습니다. 0.3.2
  이후 기능 코드 변경은 없습니다. 0.3.2에서 도입된 휴면(default-off) 병렬 실행
  executor seam과 sdist 패키징 수정은 그대로 유지됩니다.
- golden audit fixture는 새 버전 문자열을 담기 위해서만 재생성되며, audit
  report 스키마와 SHA-256 hash chain은 바뀌지 않았습니다.

## 언제 쓰나

- 튜닝·캘리브레이션된 후보를 배포하기 전
- 최고 fitness 후보가 hard constraint를 위반할 수 있을 때
- 검토자가 `best_any`와 `best_feasible`을 분리해서 봐야 할 때
- train/test 또는 holdout 전이에 walk-forward gate가 필요할 때
- 검토·CI를 위한 append-only JSON 감사 추적이 필요할 때
- 결정적이고 오프라인인 release 위생이 중요할 때
- 비-action 목적(수학, ML, 시뮬레이션)을 캘리브레이션할 때 — `KCThresholds.pure_objective()` 참고

## Trust loop / 신뢰 루프

1. 후보 파라미터를 생성하거나 전달받는다
2. `AuditingTarget`을 통해 평가한다
3. 모든 후보에 대해 hard-constraint 결과를 기록한다
4. `best_feasible`을 `best_any`와 분리해서 선택한다
5. 설정된 경우 walk-forward 또는 holdout gate를 적용한다
6. JSON 결과, audit report, scorecard를 출력한다
7. 선택적으로 SHA-256 hash chain 증거와 함께 직렬화한다
8. generated claims와 저장소 일관성을 오프라인으로 검증한다

## 설치

```bash
pip install omega-lock==0.3.3
pip install "omega-lock[p2]==0.3.3"
```

PyPI 명령은 사용하는 package index에 `0.3.3`이 보일 때만 사용하세요. 로컬 버전
메타데이터는 registry 게시의 증거가 아닙니다.

소스에서 설치:

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

## 검증 및 근거

공개 README claim은 generated claim ledger로 추적됩니다. 로컬 검사는
문서/소스 정합성을 검증할 수 있지만, registry 게시는 여전히 별도의 post-release
검증을 거쳐야 합니다.

- Claim ledger (소스): [docs/claims/public_claims.yml](docs/claims/public_claims.yml)
- Generated claim 검토표: [docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md)
- 저장소 surface: [docs/REPO_SURFACE.md](docs/REPO_SURFACE.md)
- Trust model: [docs/TRUST_MODEL.md](docs/TRUST_MODEL.md)
- Toolkit positioning: [docs/TOOLKIT_POSITIONING.md](docs/TOOLKIT_POSITIONING.md)
- Release checklist: [RELEASE.md](RELEASE.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- 오프라인 quality CI: [.github/workflows/quality-ci.yml](.github/workflows/quality-ci.yml)
- Publish workflow: [.github/workflows/publish.yml](.github/workflows/publish.yml)

claim artifact를 오프라인으로 재생성하고 검사:

```bash
python scripts/generate_readme_claims.py
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py --check
```

## 결정적 오프라인 데모 (API·네트워크 불필요)

API key나 네트워크 접근이 필요하지 않습니다.

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

`demo_replay.py`는 체크인된 `examples/phantom_demo.py` 출력의 paced replay이며,
12축 sensitivity, top-K unlock, grid search, walk-forward validation, KC report,
zoom refinement 흐름을 보여줍니다. 두 데모는 deterministic이며 network/API key가
필요 없습니다.

60초 데모 영상은 동일한 로컬 흐름을 보여줍니다.

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

## 무엇이 다른가?

| 능력 | omega-lock | 일반 optimizer | 임시 grid/random search | benchmark 전용 리포트 |
| --- | --- | --- | --- | --- |
| raw winner를 감사 전까지 untrusted로 취급 | ✓ | ✗ | ✗ | 부분적 |
| `best_any`와 `best_feasible`을 분리 | ✓ | ✗ | ✗ | ✗ |
| 후보별로 선언된 hard-constraint 결과를 기록 | ✓ | 다양 | 수동 | ✗ |
| 설정 시 walk-forward / holdout gate 지원 | ✓ | 다양 | 수동 | 다양 |
| 검토 가능한 JSON 감사 artifact 출력 | ✓ | 다양 | 수동 | 리포트 한정 |
| 선택적 SHA-256 hash chain 변조 증거 | ✓ | ✗ | ✗ | ✗ |
| Generated README claim ledger | ✓ | ✗ | ✗ | ✗ |
| 전역 최적해·도메인 정확성 주장 | ✗ | 때때로 | ✗ | ✗ |

포지션: Omega-Lock은 optimizer 대체가 아니라 audit gate가 우선입니다.
optimizer는 "무엇이 가장 높은 점수를 받았는가?"에 답하고, Omega-Lock은
"무엇이 선언된 evidence gate를 통과했는가?"에 답합니다.

## 하지 않는 것

- 정답 채점이나 gold-label 채점이 아닙니다
- correctness 증명이 아닙니다
- root cause 증명이 아닙니다
- production runtime wrapper, dashboard, web app이 아닙니다
- cryptographic signing이나 immutable storage가 아닙니다
- published-registry verifier가 아닙니다 — registry 상태는 별도의 post-release
  검증이 필요합니다
- 설치되는 console 명령은 없습니다 — Omega-Lock은 현재 console `omega-lock diff`
  명령을 제공하지 않습니다

## 무엇을 감사하나

Omega-Lock은 튜닝된 calibration 후보를 위한 audit-first framework입니다. 후보
생성 이후에 붙어, 후보가 선언된 게이트를 통과하는지 확인합니다.

- **Walk-forward gate (KC-4)**: test target 데이터에서 walk-forward 재평가를
  수행하고 Pearson 및 trade-ratio 기준을 확인합니다.
- **Pure-objective 프리셋 (0.3.0)**: `KCThresholds.pure_objective()`는 action-count
  게이트(KC-3, KC-4의 trade-ratio 하위 게이트)를 비활성화하고 도메인-중립
  게이트는 유지하므로, 비-action 목적이 action-count 하한에 강제로 걸리지 않습니다.
- **선언형 hard constraints**: 모든 후보에 대해 constraint를 평가하고
  기록합니다. `constraint_policy="prefer_feasible"`은 선언된 constraint를 모두
  만족하는 후보를 우선 선택합니다.
- **Feasible-best vs absolute-best**: audit report는 `best_feasible`과
  `best_any`를 노출해, 최고 fitness 후보가 hard constraint를 위반했는지 검토자가
  볼 수 있게 합니다.
- **Append-only audit trail**: 모든 평가 후보를 `AuditedRun`으로 append하며,
  phase, role, round, `call_index` 문맥을 남깁니다.
- **선택적 tamper evidence**: `report.to_json(with_hash_chain=True)`로 opt-in
  SHA-256 hash chain을 포함할 수 있고, `AuditReport.verify_hash_chain(...)`으로
  검증할 수 있습니다.

## 왜 feasible-best가 중요한가

absolute-best 후보는 fitness가 가장 높아도 hard constraint를 위반할 수 있습니다.
`best_any`는 "무엇이 가장 높은 점수를 받았는가?"에 답하고, `best_feasible`은
"선언된 constraint를 만족하면서 가장 높은 점수를 받은 후보가 무엇인가?"에
답합니다. 감사와 CI에서는 보통 두 번째 답이 실제로 다음 단계로 갈 수 있는
후보입니다.

일반적인 감사에는 `constraint_policy="prefer_feasible"`을 권장합니다. feasible
candidate가 없으면 즉시 실패해야 하는 경우에는
`constraint_policy="hard_fail"`을 사용합니다. 기본값 `record`는 하위 호환성을
위해 constraint 위반을 기록만 하고 `grid_best` 선택에는 개입하지 않습니다.

## 설치명과 import명

이름은 의도적으로 구분됩니다.

| 구분 | 이름 |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| 설치되는 console executable | 현재 없음 |

Python import:

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard
```

## 최소 감사 예시

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint(
            "must_be_feasible",
            lambda params, result: result.metadata["sharpe"] > 0.5,
        ),
    ],
)

result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))  # feasible best vs absolute best
```

## Benchmark와 claim 증거

`run_benchmark`와 `examples/benchmark_battery.py`는 effective recall,
generalization gap, `stress_rank_spearman` 같은 기계적으로 계산되는 metric으로
objective scorecard를 만듭니다.

체크인된 benchmark regression fixture는 결정적 `stress_rank_spearman` 값을
추적합니다. 이는 regression 신호이며, Omega-Lock이 다른 optimizer보다 우월하다는
주장이 아닙니다.

공개 claim ledger와 근거 링크는 위의 [검증 및 근거](#검증-및-근거) 섹션에
정리되어 있습니다.

## Badge와 download 분석 경계

이 README의 정적 배지는 로컬 메타데이터 surface, 지원 Python 버전, 로컬 quality
gate, methodology positioning을 식별합니다. 배지는 release readiness,
correctness, trustworthiness, 채택, package 품질을 증명하지 않습니다.

download나 star는 가시성을 나타낼 수 있을 뿐, correctness, trustworthiness,
release readiness를 나타내지 않습니다. star/download는 audit 증거나 release
승인으로 사용해서는 안 됩니다. 이 문서는 PyPI/GitHub download 분석을 주장하지
않습니다.

## 범위

Omega-Lock은 CLI/Python package/CI audit tool입니다. 기본 경로는 오프라인이어야
하고, 가능한 검증은 deterministic해야 하며, 공개 claim은 claim ledger의 증거를
따라야 합니다.

## 라이선스

Apache 2.0. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
