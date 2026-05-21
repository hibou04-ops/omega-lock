# Omega-Lock

> 튜닝된 후보를 배포 전에 감사합니다: 스트레스 경계, 하드 제약조건, 워크포워드 검증, 그리고 리뷰어가 diff할 수 있는 append-only JSON 감사 기록.

[![Release](https://img.shields.io/badge/release-0.2.4-orange.svg)](https://pypi.org/project/omega-lock/0.2.4/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

```bash
pip install omega-lock==0.2.4
```

Omega-Lock은 캘리브레이션 결과를 위한 감사 게이트입니다. 가장 빠른 optimizer가 되려는 도구가 아닙니다. 사용 중인 optimizer가 만든 후보가 매번 동일한 기계적 검토를 통과하는지 확인합니다: 선언된 제약조건, train→test 일반화, 스트레스 경계 검증, 리뷰 가능한 증거 기록.

---

### 빠른 핵심 요약 (Quick Diagnostic):
* **무엇인가요?** 튜닝된 캘리브레이션 후보를 위한 구조적 파괴 경계 감사 도구(structural failure-boundary auditor)입니다.
* **왜 중요한가요?** 미리 선언된 감사 게이트(audit gate)를 강제하고 위변조가 불가능한 append-only JSON 감사 추적 기록(audit trail)을 남김으로써, 학습 데이터에 과적합되었거나 제약조건을 위반한 후보가 배포되는 것을 기계적으로 방지합니다.
* **정답 라벨이 필요한가요?** **아니요.** Omega-Lock은 답안 채점기(answer-key evaluator)가 아닙니다. 정답 대신 사용자가 제공하는 스트레스 프로파일, 불변 조건, 실패 기준, 임계값, 워크포워드 슬라이스를 사용해 파괴 경계를 감사합나다.
* **60초 안에 작동 과정을 보려면?** 아래의 데모 동영상을 확인하고 `python examples/demo_replay.py`를 실행하세요.

---

## 데모 (60s)

https://github.com/user-attachments/assets/1012965d-0a01-41b5-96f5-93f87ad751e7

> `examples/phantom_demo.py` 실행에 대한 60초 데모: 12축 감도 측정 → top-K unlock (실제 효과 파라미터 3개, decoy 9개) → 50-combo 그리드 탐색 → 워크포워드 검증 (Pearson 0.879) → KC-1..4 모든 게이트 통과(PASS) → fractal-vise 모드가 `alpha 0.5 → 0.4375`로 정밀화. 실제 `phantom_demo.py` 출력을 기반으로 하며, `python examples/demo_replay.py` 명령으로 동일하게 재현할 수 있습니다.

---

## 정답 주입이 필요 없는 이유 — Omega-Lock은 파괴 경계 감사 도구입니다

Omega-Lock은 답안 채점기가 아니라 구조적 파괴 경계 감사 도구입니다. 많은 도메인에는 주입할 수 있는 단일한 “정답”이 존재하지 않습니다. 

금속 피로파괴 시험은 금속의 “정답”을 요구하지 않습니다. 하중 조건, 반복 조건, 파단 기준, 허용 임계값을 요구합니다. Omega-Lock도 마찬가지입니다.

사용자는 다음을 제공합니다:
- **스트레스 프로파일 (stress profiles)**: 어떤 파라미터, 슬라이스, 콘텍스트, 코너, 혹은 동작 영역(regime)을 교란(perturb)할 것인가
- **불변 조건 (invariants)**: 항상 참이어야 하는 조건 (예: 유효한 JSON 스키마, 필수 키 존재 여부, PVT 마진, 최대 낙폭 등)
- **실패 기준 (failure criteria)**: 어떤 상태를 파괴나 실패로 간주할 것인가
- **임계값 (thresholds)**: 합격과 실패를 가르는 선언된 기준선
- **워크포워드 슬라이스 (walk-forward slices)**: 튜닝 단계에서 후보가 참조하지 않은 별도의 홀드아웃 데이터셋
- **감사 아티팩트 (audit artifacts)**: 리뷰어가 diff로 손쉽게 검증할 수 있는 append-only 증거 추적 기록

일반적으로 다음은 필요하지 않습니다:
- 모든 개별 입력에 대한 정답(gold answer)
- 모든 개별 출력에 대한 인간 선호도 라벨(human preference label)
- 목표 fitness 평가식 자체가 요구하지 않는 의미적 정답 판독기(semantic judge)

요약하면 다음과 같습니다:
```text
omega-lock = 정답 라벨(answer key) 불필요.
omega-lock은 실패 오라클(failure oracle) / 불변 조건(invariant) / 임계값(threshold) / 스트레스 케이스를 필요로 함.
```

| 흔한 오해 | 정확한 해석 |
|---|---|
| “Omega-Lock은 답을 채점한다.” | 튜닝된 후보가 선언한 감사 게이트(audit gate)를 구조적으로 통과하는지 감사합니다. |
| “ground truth 정답 라벨이 필요하다.” | 불변 조건(invariant), 실패 기준(failure criteria), 임계값(threshold), 스트레스 조건이 필요합니다. |
| “실패의 근본 원인(root cause)을 수학적으로 증명한다.” | 후보가 언제 어디서 어떻게 실패했는지에 대한 증거를 감사 추적 기록으로 기록합니다. |
| “기존 옵티마이저를 대체한다.” | 옵티마이저를 감싸거나 보완하여 실행됩니다. 어떤 탐색 엔진을 쓰더라도 감사 규율은 동일합니다. |

---

## 빠른 시작

### 0. 로컬 결정적 데모 실행 (인터넷/API 키 불필요)

```bash
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock && pip install -e ".[dev]"

# 12축 감도 측정 → top-K unlock → 워크포워드 검증 재현 데모
python examples/demo_replay.py

# 6T SRAM 비트셀에 대한 5개 PVT 코너 및 3가지 하드 제약조건 감사 데모
python examples/demo_sram.py
```

### 1. 감사 모듈로 대상(Target) 감싸기

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

# 1. 감사 대상 타겟과 하드 제약조건 정의
audited = AuditingTarget(
    YourCalibrableTarget(),
    constraints=[
        Constraint("must_be_feasible",
                   lambda params, result: result.metadata["sharpe"] > 0.5),
        Constraint("no_drawdown_blowup",
                   lambda params, result: result.metadata["max_dd"] < 0.3),
    ],
)

# 2. P1 파이프라인 실행 (제약조건 만족 우선 정책 적용)
result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

# 3. 감사 점수표 출력 및 저장
report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))
```

---

## 제약조건 정책 (Constraint Policy)

- `record`: 하위 호환 모드입니다. 제약조건 위반을 감사 추적 기록에 저장하지만, 최적 후보(`grid_best`) 선택 단계에서 이를 차단하지는 않습니다.
- `prefer_feasible` (권장): 일반적인 감사/CI에 사용됩니다. 제약조건을 만족한 후보(feasible candidate) 중 최선의 성과를 낸 후보를 우선하여 선택합니다.
- `hard_fail`: 엄격한 배포/CI 게이트용입니다. 모든 후보가 제약조건을 하나라도 위반하는 경우 전체 런을 실패로 표시합니다.

---

## 언제 사용해야 하나요? (When to use)

- **정량적/퀀트 투자 전략 튜닝**: 과거 데이터에서는 훌륭해 보이지만 실제 시장 환경(워크포워드 슬라이스)에서 붕괴하는 과적합 후보를 걸러내고 싶을 때 (KC-4 게이트 활용)
- **하드웨어/회로/시뮬레이션 캘리브레이션**: PVT 스윕, 재료 설계 등 물리적 제약조건이 확실한 시스템에서 고비용 시뮬레이션을 수행할 때 (SRAM 데모 참조)
- **ML / 하이퍼파라미터 거버넌스**: 옵티마이저가 제시한 "최종 결과"가 왜 선택되었는지, 어떤 제약조건을 만족했는지 문서화된 감사 추적 기록(audit trail)을 남기고 싶을 때

## 언제 사용하지 말아야 하나요? (When NOT to use)

- **요청 단위(per-request) 실시간 런타임 검증이 필요할 때**: Omega-Lock은 오프라인 배치/CI 단계용 진단 감사 도구이며, 프로덕션의 실시간 미들웨어가 아닙니다.
- **순수 시맨틱 팩츄얼리티 채점만 필요할 때**: 대상 시스템의 fitness 함수 자체가 시맨틱 평가를 요구하지 않는 한, 단순한 LLM 채점 용도로 쓰기엔 프레임워크가 무겁습니다.
- **일반화 성능(out-of-sample stability)이 중요하지 않을 때**: 단순히 인샘플 튜닝 결과만 뽑고자 한다면 일반적인 그리드 서치로 충분합니다.
- **분산 연산 스케줄링 자체가 메인 문제일 때**: 분산 클러스터 탐색이 목적이라면 Ray Tune을 사용하세요.
- **수십 종의 고도화된 블랙박스 옵티마이저가 필요할 때**: Optuna나 Hyperopt를 직접 사용하고, 이들이 제안한 최종 후보군을 Omega-Lock 감사 게이트로 검증하는 식으로 결합하는 것이 바람직합니다.

---

## 릴리스 기록 (Release History)

**0.2.4** (2026-05-22) — **문서 정합성 릴리스.** README_KR.md, EASY_README.md, EASY_README_KR.md를 메인 README의 파괴 경계 감사 포지션과 일치하도록 복구했습니다. 모든 문서에서 정답 주입이 필수가 아닌 구조를 명확히 설명하고, 상단 설명을 더 직관적으로 다듬었으며, 60초 데모와 PyPI 버전 표기를 정렬했습니다.

**0.2.3** (2026-05-22) — **Structural audit positioning.** Omega-Lock이 답안 채점기가 아니라 failure-boundary auditor라는 점을 명확히 했습니다. 정답 라벨이 일반적으로 필요하지 않다는 guidance를 추가하고, invariant / threshold / failure oracle 언어를 강화했으며, 60초 데모와 긴 audit-first 문서를 보존했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.2** (2026-05-22) — **Badge hardening and release-surface synchronization.** 동적 PyPI version badge를 정적 release badge로 교체해 Shields/PyPI/Camo 캐시 때문에 오래된 버전이 보이는 문제를 피했습니다. 당시 install command와 citation도 0.2.2로 동기화했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.1** (2026-05-22) — **Release sync and badge cache-bust correction.** 동적 PyPI badge URL에 release-specific cache-bust query를 추가하고, 0.2.0 업로드 이후 release metadata와 README/PyPI 표면을 다시 동기화했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.0** (2026-05-22) — **Public README and release-surface polish.** GitHub/PyPI 첫 화면을 더 직접적으로 정리하고, 없는 비디오 placeholder를 제거했으며, `Constraint`를 포함한 quickstart를 실제 API에 맞췄습니다. 영어/한국어 문서와 release checklist를 0.2.0 기준으로 갱신했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.1.9** (2026-05-22) — **README, PyPI metadata, and release hygiene correction.** Cleaned stale README/PyPI long-description text, repaired Korean documentation encoding/content, and added a release checklist so GitHub and PyPI stay synchronized.

**0.1.8** (2026-05-21) — **Audit reliability and static hygiene release.** Establishes a clean baseline across pytest, pyright, and ruff: 289 tests passing, `pyright src tests` at 0 errors, and `ruff check src tests` clean.

**0.1.4** (2026-04-20) — **audit surface as the headline.** New `omega_lock.audit` submodule: `AuditingTarget`, `Constraint`, `AuditReport`, `make_report`, `render_scorecard`.

---

## 라이선스 (License)

Apache 2.0. Copyright (c) 2026 hibou.
