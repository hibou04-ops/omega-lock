# Omega-Lock 쉬운 설명

> 이미 optimizer나 수동 튜닝으로 후보 파라미터가 있다고 가정합니다. Omega-Lock이 묻는 질문은 하나입니다. **이 후보가 실제로 일반화되었고, 실패 경계 감사(failure-boundary audit)를 통과하나요?**

[![Release](https://img.shields.io/badge/release-0.2.4-orange.svg)](https://pypi.org/project/omega-lock/0.2.4/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## 무엇을 하나요?

Omega-Lock은 **검색 우선(search-first)이 아니라 감사 우선(audit-first)** 도구입니다. 후보를 만든 뒤에 검증 엔진으로 붙여서 구조적 파괴 경계를 감사하며, 다음 조건을 검증합니다.

- **선언된 하드 제약조건**: 모든 파라미터 후보에 대해 선언한 제약조건 만족 여부를 자동으로 평가하고 기록합니다.
- **워크포워드 동작 검증**: 사후에 임계값을 낮추어 부적격 후보를 배포하는 것을 막기 위해, 사전 선언된 ship gate(예: KC-4 Pearson 상관계수 및 trade-ratio 임계값)를 검증 슬라이스상에서 엄격히 통과하는지 확인합니다.
- **홀드아웃 타겟 증거**: 탐색 중 전혀 활용되지 않은 홀드아웃 타겟을 마지막에 단 한 번 평가하여, 일반화 가능성을 있는 그대로 측정하고 기록합니다.
- **append-only 감사 추적 기록**: 리뷰어가 Git PR에서 diff로 간편하게 검토하고 검증할 수 있는 JSON 형태의 감사 추적 기록(audit trail)을 남깁니다. (SHA-256 해시 체인 기능 포함)

## 설치

```bash
pip install omega-lock==0.2.4
```

Optuna TPE 기능까지 활성화하려면:

```bash
pip install "omega-lock[p2]==0.2.4"
```

## 먼저 이렇게 시작하세요

일반적인 감사 및 CI 목적이라면, 검증하고자 하는 대상을 `AuditingTarget`으로 감싸고 `constraint_policy="prefer_feasible"` 설정을 활용해 최적 탐색을 수행하세요. 제약조건을 만족한 후보(feasible candidate) 중 최선의 후보를 자동으로 골라줍니다.

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

# 1. 제약조건과 함께 감사 대상 정의
audited = AuditingTarget(
    my_target,
    constraints=[
        Constraint("must_be_feasible",
                   lambda params, result: result.metadata["sharpe"] > 0.5),
    ],
)

# 2. 제약조건 만족 우선 정책으로 탐색 및 감사 수행
result = run_p1(
    train_target=audited,
    config=P1Config(constraint_policy="prefer_feasible"),
)

# 3. 감사 점수표 보고서 생성 및 출력
report = make_report(audited, method="run_p1", seed=42)
print(render_scorecard(report))
```

## 정답 라벨이 필요한가요?

**대부분은 필요 없습니다.** Omega-Lock은 답의 정오를 채점하는 도구가 아니라 튜닝된 후보가 구조적으로 살아남는지 감사하는 도구입니다. 

금속 피로파괴 시험은 금속의 "정답"을 필요로 하지 않으며 하중 조건, 스트레스 프로파일, 파단 기준, 허용 임계값만 요구합니다. Omega-Lock도 동일하게 작동합니다.

사용자는 정답 라벨 대신 제약조건, 임계값, 스트레스 케이스, train/test/holdout 분할을 제공합니다. 의미적 정답성이 목표 함수(fitness function) 자체에 포함되어 이를 판단해야 할 때만 gold label이나 별도 evaluator가 필요합니다.

## 제약조건 정책 (Constraint Policy)

- `record`: 하위 호환 모드입니다. 위반 사항을 감사 추적 기록에 기록하지만 최적 후보 선택을 방해하지는 않습니다.
- `prefer_feasible` (권장): 제약조건을 모두 만족하는 후보(feasible candidate)를 최우선적으로 필터링하여 선택합니다.
- `hard_fail`: 엄격한 릴리스/CI 게이트용입니다. 제약조건을 만족하는 후보가 없는 경우 즉시 실패를 보고합니다.

## 로컬 데모 실행

```bash
# 감도 측정 기반 top-K 그리드 서치 및 감사 프로세스 재현 데모
python examples/demo_replay.py

# PVT 코너별 6T SRAM 비트셀의 하드 제약조건 만족 여부 시뮬레이션 데모
python examples/demo_sram.py
```

두 데모는 로컬에서 결정적으로 실행되며 인터넷 연결이나 별도의 API 키가 필요 없습니다.

## 0.2.4에서 바뀐 점

- 버전 표기와 설치 가이드를 `0.2.4`로 업데이트했습니다.
- `README_KR.md`, `EASY_README.md`, `EASY_README_KR.md` 파일들의 핵심 논조와 구성을 메인 README의 "구조적 파괴 경계 감사 도구(failure-boundary auditor)" 포지션에 맞춰 완전히 재정렬하고 문구를 통일했습니다.
- 정답 주입이 필요 없는 오프라인/CI 배치 감사 도구로서의 역할을 명확히 설명했습니다.
- 버전 metadata 외 런타임 동작 변경은 없습니다.

전체 상세 기술 문서는 [README_KR.md](README_KR.md)를 참고하세요.
