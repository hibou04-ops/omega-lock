# Omega-Lock

> 어떤 optimizer가 만든 후보든, 배포 전에 믿을 수 있는지 확인하는 **옵티마이저 후단 감사 게이트**입니다. Omega-Lock은 튜닝된 후보가 일반화되는지, 선언한 제약조건 안에 있는지, 리뷰 가능한 JSON 아티팩트를 남기는지 검사합니다.

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

짧은 설명부터 보고 싶다면 [EASY_README_KR.md](EASY_README_KR.md)를 보세요. English docs: [README.md](README.md)

```bash
pip install omega-lock
```

Optuna TPE 경로까지 쓰려면:

```bash
pip install "omega-lock[p2]"
```

## 첫 화면 요약

Omega-Lock은 **검색 우선(search-first)이 아니라 감사 우선(audit-first)**입니다.

Grid search, Optuna, Bayesian search, 내부 optimizer, 사람이 고른 후보까지 어떤 방식으로 후보를 만들었든 상관없습니다. Omega-Lock은 그 후보를 후단에서 검증합니다.

- 학습 데이터에서 좋았던 후보가 테스트/홀드아웃에서도 일반화되는가?
- 선언한 제약조건을 만족하는가?
- 워크포워드 결과가 무너지지 않는가?
- reviewer가 나중에 읽고 diff할 수 있는 감사 아티팩트가 남는가?

막아야 하는 실패는 단순합니다. "훈련 점수는 최고였지만 실제 배포에서는 깨지는 후보"가 그대로 ship되는 것입니다.

## 최소 예제

```python
from typing import Any

from omega_lock import EvalResult, P1Config, ParamSpec, run_p1


class TinyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec("x", "float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec("risk", "float", low=0.0, high=1.0, neutral=0.5),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        fitness = 1.0 - abs(float(params["x"]) - 0.8) - 0.5 * float(params["risk"])
        return EvalResult(
            fitness=fitness,
            n_trials=100,
            metadata={"risk": float(params["risk"])},
        )


result = run_p1(
    train_target=TinyTarget(),
    config=P1Config(
        unlock_k=2,
        grid_points_per_axis=5,
        constraint_policy="prefer_feasible",
        stress_verbose=False,
        grid_verbose=False,
    ),
)

print(result.status)
print(result.warnings)
print(result.config_full["constraint_policy"])
print(result.search_settings)
```

제약조건별 통과/실패 trail까지 남기려면 `AuditingTarget`과 `Constraint`를 함께 사용하세요.

## 권장 제약조건 정책

일반적인 감사/CI 사용에서는 `constraint_policy="prefer_feasible"`를 권장합니다.

- `record`: 하위 호환 모드입니다. 제약조건 위반을 기록하지만 best 후보 선택을 gate하지 않습니다.
- `prefer_feasible`: 일반 사용에 권장합니다. 가능한 경우 제약조건을 만족한 후보를 `grid_best`로 우선합니다.
- `hard_fail`: 더 엄격한 CI/릴리스 게이트입니다. 만족 가능한 후보가 없으면 run을 실패로 표시합니다.

## 아티팩트가 기록하는 것

- 워크포워드 결과: train ranking이 test 데이터에서도 유지되는지 확인합니다.
- 홀드아웃 증거: 세 번째 target을 제공하면 최종 후보를 홀드아웃에서 한 번 평가합니다.
- 제약조건 결과: 선언한 제약조건의 pass/fail을 기록합니다.
- 재현성 metadata: `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, `search_settings`
- 경고: `constraint_policy="record"`처럼 기록은 하지만 선택을 gate하지 않는 경우, `holdout_mode="evidence_only"`처럼 홀드아웃이 최종 상태를 gate하지 않는 경우를 명시합니다.

## 워크포워드와 홀드아웃

워크포워드(KC-4)는 train fitness와 test fitness의 상관, 그리고 action/trade count 비율을 확인합니다. 아티팩트에는 기존 numeric `pearson`과 함께 `pearson_status`, `pearson_computable`이 들어갑니다. 그래서 "상관이 낮다"와 "상관을 계산할 수 없다"를 구분할 수 있습니다.

홀드아웃은 `run_p1(..., holdout_target=...)` 또는 `run_p1_iterative(..., holdout_target=...)`로 전달합니다. 기본값인 `holdout_mode="evidence_only"`는 하위 호환을 위해 최종 status를 gate하지 않습니다. 릴리스 게이트로 쓰려면 `holdout_mode="gate"`와 threshold를 설정하세요.

## 0.1.9 릴리스

**0.1.9 — README, PyPI metadata, and release hygiene correction**

- README와 PyPI long description을 더 직접적인 제품 설명으로 정리했습니다.
- PyPI badge가 최신 PyPI 버전을 동적으로 따라가도록 했습니다.
- 오래된 version/badge/test count 참조를 정리했습니다.
- 한국어 문서를 자연스러운 UTF-8 Markdown으로 다시 작성했습니다.
- 릴리스 체크리스트를 추가해 GitHub tag, package metadata, fresh `dist/` artifact, PyPI 업로드가 서로 어긋나지 않게 했습니다.
- 런타임 동작은 바꾸지 않았습니다.

## 0.1.8 요약

0.1.8은 audit reliability와 static hygiene 릴리스였습니다.

- `pytest`, `pyright`, `ruff` 기준이 모두 깨끗합니다.
- 선택 의존성인 `optuna` import typing을 정리했습니다.
- `CalibrableTarget` Protocol typing을 정리했습니다.
- hash-chain typing을 JSON shape 변경 없이 정리했습니다.
- 아티팩트에 `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, `search_settings`가 추가되었습니다.
- `constraint_policy="record"` 경고, `holdout_mode="evidence_only"` 경고, iterative test reuse 경고가 추가되었습니다.
- 워크포워드 아티팩트에 `pearson_status`, `pearson_computable`이 추가되었습니다.

## 빠른 실행 경로

```bash
python examples/phantom_demo.py
python examples/demo_sram.py
```

첫 번째는 합성 keyhole 문제이고, 두 번째는 6T SRAM bitcell을 여러 PVT corner에서 평가하는 감사 데모입니다.

## 라이선스

Apache 2.0. Copyright (c) 2026 hibou.
