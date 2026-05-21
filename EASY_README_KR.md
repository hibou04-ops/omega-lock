# Omega-Lock 쉬운 설명

> 이미 optimizer나 수동 튜닝으로 후보 파라미터가 있다고 가정합니다. Omega-Lock이 묻는 질문은 하나입니다. **이 후보가 실제로 일반화되었나요?**

[![Release](https://img.shields.io/badge/release-0.2.3-orange.svg)](https://pypi.org/project/omega-lock/0.2.3/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## 무엇을 하나요?

Omega-Lock은 **검색 우선이 아니라 감사 우선** 도구입니다. 후보를 만든 뒤에 붙여서 다음을 확인합니다.

- 선언한 제약조건
- 테스트 데이터에서의 워크포워드 동작
- 홀드아웃 target을 제공했을 때의 홀드아웃 증거
- 증거를 기록만 하고 gate하지 않는 모드의 경고
- 리뷰어가 읽고 diff할 수 있는 JSON 감사 아티팩트

## 설치

```bash
pip install omega-lock==0.2.3
```

Optuna TPE 경로까지 쓰려면:

```bash
pip install "omega-lock[p2]==0.2.3"
```

## 먼저 이렇게 시작하세요

일반적인 감사/CI 용도에서는:

```python
from omega_lock import P1Config, run_p1

result = run_p1(
    train_target=my_target,
    config=P1Config(constraint_policy="prefer_feasible"),
)

print(result.status)
print(result.warnings)
print(result.config_full)
```

제약조건 통과/실패 trail까지 남기려면 `AuditingTarget`과 `Constraint`를 함께 사용하세요.

## 정답 라벨이 필요한가요?

대부분은 필요 없습니다. Omega-Lock은 답이 맞는지 채점하는 도구가 아니라 후보가 구조적으로 살아남는지 감사하는 도구입니다. 사용자는 제약조건, 임계값, stress case, train/test/holdout slice를 제공합니다. 정답 라벨은 목표 fitness 자체가 의미적 정답성을 요구할 때만 필요합니다.

## 제약조건 정책

- `record`: 하위 호환 모드입니다. 위반을 기록하지만 best 후보 선택을 gate하지 않습니다.
- `prefer_feasible`: 일반 사용에 권장합니다. 가능한 경우 제약조건을 만족한 후보를 우선합니다.
- `hard_fail`: 더 엄격한 릴리스/CI 게이트입니다.

## 로컬 데모

```bash
python examples/demo_replay.py
python examples/demo_sram.py
```

두 데모는 결정적이며 네트워크나 API key가 필요 없습니다.

## 0.2.3에서 바뀐 점

- 메인 README를 짧은 landing page가 아니라 긴 기술 문서로 복원했습니다.
- Omega-Lock을 답안 채점기가 아니라 파괴 경계 감사 도구로 명확히 설명했습니다.
- 정답 라벨이 없어도 쓸 수 있는 경우를 더 분명히 적었습니다.
- 버전 metadata 외 런타임 동작은 바꾸지 않았습니다.

전체 문서는 [README_KR.md](README_KR.md)를 보세요.
