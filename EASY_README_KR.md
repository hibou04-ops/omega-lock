# Omega-Lock 쉬운 설명

> 먼저 큰 그림만 보고 싶은 사람을 위한 짧은 문서입니다.
> 전체 문서: [README_KR.md](README_KR.md) · English: [EASY_README.md](EASY_README.md)

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)

## Omega-Lock은 무엇인가요?

Omega-Lock은 **검색 우선 도구가 아니라 감사 우선 도구**입니다.

Grid search, Optuna, Bayesian search, 직접 만든 optimizer, 사람이 고른 후보까지 어떤 방식으로 후보를 만들었든 상관없습니다. Omega-Lock은 그 후보가 믿을 만한지 확인합니다.

- 학습 데이터에서 좋았던 성능이 테스트/홀드아웃에서도 유지되는가?
- 선언한 제약조건을 만족했는가?
- 워크포워드 검증에서 실패하지 않았는가?
- 나중에 리뷰할 수 있는 감사 아티팩트가 남는가?

즉, 튜닝이 끝난 뒤 **배포해도 되는지, 버려야 하는지, 추가 확인이 필요한지** 알려주는 검증 레이어입니다.

## 설치

```bash
pip install omega-lock
```

Optuna TPE 경로까지 쓰려면:

```bash
pip install "omega-lock[p2]"
```

## 최소 예제

```python
from omega_lock import P1Config, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

wrapped = AuditingTarget(
    my_target,
    constraints=[
        Constraint(
            "score_positive",
            lambda params, result: result.fitness > 0,
            "점수는 양수여야 함",
        ),
    ],
)

result = run_p1(
    train_target=wrapped,
    config=P1Config(constraint_policy="prefer_feasible"),
)

report = make_report(wrapped, method="run_p1", seed=42)
print(result.status)
print(render_scorecard(report))
```

결과는 pass/fail 상태와 모든 평가 기록이 들어 있는 JSON 감사 아티팩트입니다.

## 제약조건 정책

일반적인 감사/CI 용도에서는 `constraint_policy="prefer_feasible"`를 권장합니다.

- `record`: 하위 호환 기본값입니다. 제약조건 위반을 기록하지만 best 후보 선택을 막지는 않습니다.
- `prefer_feasible`: 일반 사용에 더 안전합니다. 가능한 경우 제약조건을 만족한 후보를 `grid_best`로 우선 선택합니다.
- `hard_fail`: 더 엄격한 CI/릴리스 게이트입니다. 만족 가능한 후보가 없으면 run 자체를 실패로 표시합니다.

## 0.1.8에서 바뀐 점

- `pytest`, `pyright`, `ruff` 기준이 모두 깨끗합니다. 현재 테스트 기준은 289 passed입니다.
- 선택 의존성인 `optuna` import typing을 정리했습니다.
- 테스트의 `CalibrableTarget` Protocol typing을 정리했습니다.
- 아티팩트에 재현성 필드가 추가되었습니다: `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, `search_settings`.
- `constraint_policy="record"`일 때 제약조건이 기록만 되고 선택을 gate하지 않는다는 경고가 남습니다.
- `holdout_mode="evidence_only"`일 때 홀드아웃이 최종 상태를 gate하지 않는다는 경고가 남습니다.
- iterative run에서 같은 테스트 slice를 여러 라운드에 재사용하면 KC-4 증거력이 약해진다는 경고가 표시됩니다.
- 워크포워드 아티팩트에 기존 numeric `pearson`과 함께 `pearson_status`, `pearson_computable`이 추가되었습니다.

## 언제 쓸 가치가 있나요?

- 평가가 비쌉니다: 시뮬레이션, 백테스트, LLM 호출, 모델 학습.
- 튜닝된 후보를 배포 전에 검증해야 합니다.
- 제약조건, train/test split, 홀드아웃 증거가 중요합니다.
- 나중에 diff하고 보관할 수 있는 아티팩트가 필요합니다.

## 언제 과한가요?

- 일회성 장난감 문제입니다.
- 목적함수가 매끄럽고 샘플이 사실상 무제한입니다.
- 결과를 아무도 리뷰하거나 재사용하지 않습니다.

## 다음 단계

- 데모 실행: `python examples/phantom_demo.py`
- SRAM 감사 데모: `python examples/demo_sram.py`
- 전체 문서: [README_KR.md](README_KR.md)

License: Apache 2.0. Copyright (c) 2026 hibou.
