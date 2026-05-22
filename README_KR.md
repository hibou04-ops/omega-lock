# Omega-Lock

> 배포 전 튜닝 후보를 검증합니다: walk-forward validation, 선언형 hard constraints, feasible-best 선택, append-only JSON 감사 추적.

[![Version 0.2.6](https://img.shields.io/badge/version-0.2.6-orange.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Quality pytest + pyright + ruff](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-2ea44f.svg)](.github/workflows/quality-ci.yml)
[![Methodology audit gate](https://img.shields.io/badge/methodology-audit--gate-6f42c1.svg)](docs/TRUST_MODEL.md)
[![Trust first](https://img.shields.io/badge/trust-first-0f766e.svg)](docs/TRUST_MODEL.md)
[![Measurement grade audit](https://img.shields.io/badge/measurement--grade-audit-555.svg)](docs/TOOLKIT_POSITIONING.md)

[Full README](README.md) · [Easy README](EASY_README.md) · [쉬운 한국어 README](EASY_README_KR.md)

현재 로컬 패키지 버전: `0.2.6`.

이 문서는 PyPI 또는 GitHub Release 게시 여부를 주장하지 않습니다. 현재 환경에서는
원격 registry 검증이 `ENVIRONMENT_BLOCKED`이므로, 로컬 버전 문자열을 published
release의 증거로 취급하면 안 됩니다.

## 무엇을 감사하나

Omega-Lock은 후보 생성 이후에 붙는 audit-first framework입니다. 가장 빠른
optimizer가 되려는 도구가 아니라, 이미 만들어진 후보가 사전에 선언한 게이트를
통과하는지 확인합니다.

- **Walk-forward gate (KC-4)**: test target에서 walk-forward 재평가를 수행하고
  Pearson 및 trade-ratio 기준을 확인합니다.
- **Declarative hard constraints**: 모든 후보에 대해 constraint를 평가하고
  기록합니다. `constraint_policy="prefer_feasible"`은 hard constraint를 만족한
  후보를 우선 선택합니다.
- **Feasible-best vs absolute-best**: `best_feasible`과 `best_any`를 분리해,
  최고 fitness 후보가 constraint를 위반했는지 검토자가 볼 수 있게 합니다.
- **Append-only audit trail**: 모든 평가 후보를 `AuditedRun`으로 append하며,
  phase, role, round, `call_index` 문맥을 남깁니다.
- **선택적 tamper evidence**: `report.to_json(with_hash_chain=True)`로 opt-in
  SHA-256 hash chain을 포함할 수 있고, `AuditReport.verify_hash_chain(...)`으로
  검증할 수 있습니다.

## 하지 않는 것

- 정답 채점기가 아닙니다. gold label은 target의 fitness 함수가 요구할 때만
  필요합니다.
- root cause를 증명하거나, correctness를 보장하거나, 도메인 검증을 대체하지
  않습니다.
- production runtime wrapper, dashboard, web app이 아닙니다.
- 현재 설치되는 console CLI는 없습니다. 특히 Omega-Lock은 JSON artifact를
  출력하지만, 설치된 `omega-lock diff` 명령은 제공하지 않습니다.
- `0.2.6`가 PyPI에 게시되었다고 주장하지 않습니다. registry 상태는 별도로
  검증해야 합니다.

## 왜 feasible-best가 중요한가

absolute-best 후보는 fitness가 가장 높지만 hard constraint를 위반할 수 있습니다.
`best_any`는 "무엇이 가장 높은 점수를 받았는가?"에 답하고, `best_feasible`은
"선언된 constraint를 만족하면서 가장 높은 점수를 받은 후보가 무엇인가?"에
답합니다. 감사와 CI에서는 보통 두 번째 답이 실제로 다음 단계로 갈 수 있는
후보입니다.

일반적인 감사에는 `constraint_policy="prefer_feasible"`을 권장합니다. feasible
candidate가 없으면 즉시 실패해야 하는 release/CI 게이트에는
`constraint_policy="hard_fail"`을 사용합니다. 기본값 `record`는 하위 호환성을
위해 constraint 위반을 기록만 하고 `grid_best` 선택에는 개입하지 않습니다.

## 결정적 오프라인 데모 실행

아래 60초 데모 영상은 실제 로컬 데모 흐름을 보여주므로 보존합니다.

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

`examples/phantom_demo.py` 출력의 paced replay이며, 12축 sensitivity, top-K
unlock, grid search, walk-forward validation, KC report, zoom refinement 흐름을
보여줍니다. 두 데모는 deterministic이며 network/API key가 필요 없습니다.

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"

python examples/demo_replay.py
python examples/demo_sram.py
```

## 설치명과 import명

이름은 의도적으로 구분됩니다.

| 구분 | 이름 |
| --- | --- |
| GitHub repo | `hibou04-ops/omega-lock` |
| PyPI distribution | `omega-lock` |
| Python import package | `omega_lock` |
| 설치되는 console executable | 현재 없음 |

소스에서 설치:

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

사용하는 package index에 `0.2.6`가 게시되어 있는 경우에만 PyPI 설치:

```bash
pip install omega-lock==0.2.6
pip install "omega-lock[p2]==0.2.6"
```

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

현재 문서가 주장하는 공개 claim은 claim ledger에 묶여 있습니다.

- Source ledger: [docs/claims/public_claims.yml](docs/claims/public_claims.yml)
- Generated review table:
  [docs/claims/generated_readme_claims.md](docs/claims/generated_readme_claims.md)
- Repository surface:
  [docs/REPO_SURFACE.md](docs/REPO_SURFACE.md)

오프라인 검증:

```bash
python scripts/generate_readme_claims.py --check
python scripts/check_repo_consistency.py --check
```

## 범위

Omega-Lock은 CLI/Python package/CI audit tool입니다. 기본 경로는 오프라인이어야
하고, 가능한 검증은 deterministic해야 하며, 공개 claim은 claim ledger의 증거를
따라야 합니다.

## 라이선스

Apache 2.0. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
