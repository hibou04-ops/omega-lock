# Omega-Lock 쉬운 설명

> 이미 어떤 optimizer가 후보 파라미터를 만들었다고 가정합니다. Omega-Lock이 묻는 질문은 하나입니다. **이 후보가 실제로 일반화되었나요?**

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## 무엇을 하나요?

Omega-Lock은 **검색 우선이 아니라 감사 우선** 도구입니다. 옵티마이저 뒤에 붙는 후단 감사 게이트로, 튜닝된 후보를 믿어도 되는지 확인합니다.

기록하는 것:

- 제약조건과 각 후보의 통과/실패 여부
- 테스트 데이터에서의 워크포워드 동작
- 홀드아웃 target을 제공했을 때의 홀드아웃 증거
- 증거를 기록만 하고 선택이나 최종 상태를 gate하지 않는 모드의 경고
- 리뷰어가 읽고 diff할 수 있는 JSON 감사 아티팩트

## 설치

```bash
pip install omega-lock
```

Optuna TPE 경로까지 쓰려면:

```bash
pip install "omega-lock[p2]"
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

## 제약조건 정책

- `record`: 하위 호환 모드입니다. 위반을 기록하지만 best 후보 선택을 gate하지 않습니다.
- `prefer_feasible`: 일반 사용에 권장합니다. 가능한 경우 제약조건을 만족한 후보를 우선합니다.
- `hard_fail`: 더 엄격한 릴리스/CI 게이트입니다.

## 0.1.9에서 바뀐 점

- README와 PyPI long description을 더 명확하게 정리했습니다.
- PyPI badge가 최신 PyPI 버전을 동적으로 따라가도록 했습니다.
- 한국어 문서를 UTF-8 Markdown으로 다시 작성했습니다.
- 버전, 태그, dist 아티팩트, PyPI 업로드가 서로 어긋나지 않도록 릴리스 체크리스트를 추가했습니다.
- 런타임 동작은 바꾸지 않았습니다.

전체 문서는 [README_KR.md](README_KR.md)를 보세요.
