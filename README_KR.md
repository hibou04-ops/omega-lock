# Omega-Lock

> 어떤 optimizer나 수동 튜닝 과정이 만든 후보를 배포 전에 검증하는 **옵티마이저 후단 감사 게이트**입니다. Omega-Lock은 튜닝된 후보가 일반화되는지, 선언한 제약조건 안에 있는지, 리뷰 가능한 JSON 아티팩트를 남기는지 확인합니다.

[![Release](https://img.shields.io/badge/release-0.2.2-orange.svg)](https://pypi.org/project/omega-lock/0.2.2/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

```bash
pip install omega-lock==0.2.2
```

Omega-Lock은 **검색 우선(search-first)이 아니라 감사 우선(audit-first)** 도구입니다. Grid search, Optuna, Bayesian search, 사내 optimizer, 사람이 직접 고른 후보까지 어떤 방식으로 후보를 만들었든 상관없습니다. Omega-Lock은 그 다음 단계에서 묻습니다.

> 이 후보가 실제로 일반화되었고, 미리 선언한 제약조건 안에 남아 있나요?

막으려는 실패는 명확합니다. 학습 데이터에서 가장 좋아 보이는 후보가 과적합되었거나, 하드 제약조건을 어겼거나, 나중에 리뷰할 수 있는 아티팩트 없이 그대로 배포되는 상황입니다.

일반적인 감사/CI 용도에서는 다음 설정으로 시작하세요.

```python
P1Config(constraint_policy="prefer_feasible")
```

`prefer_feasible`는 제약조건을 만족한 후보 중 fitness가 가장 높은 후보를 우선합니다. 더 엄격한 릴리스 게이트에는 `hard_fail`을 쓰고, 하위 호환 때문에 제약조건 위반을 기록만 해야 하는 경우에만 `record`를 쓰세요.

## 빠른 시작

아래 예제는 공개 API와 맞는 실행 가능한 예제입니다.

```python
from typing import Any

from omega_lock import EvalResult, P1Config, ParamSpec, run_p1
from omega_lock.audit import AuditingTarget, Constraint, make_report


class ToyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec("x", "float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec("risk", "float", low=0.0, high=1.0, neutral=0.5),
        ]

    def evaluate(self, params: dict[str, Any]) -> EvalResult:
        x = float(params["x"])
        risk = float(params["risk"])
        fitness = 1.0 - abs(x - 0.8) - 0.4 * risk
        return EvalResult(
            fitness=fitness,
            n_trials=100,
            metadata={"risk": risk},
        )


def risk_ok(params: dict[str, Any], result: EvalResult) -> bool:
    return float(result.metadata["risk"]) <= 0.6


target = AuditingTarget(
    ToyTarget(),
    constraints=[
        Constraint(
            "risk_ok",
            risk_ok,
            "Risk must stay at or below 0.6.",
        )
    ],
)

result = run_p1(
    train_target=target,
    config=P1Config(
        unlock_k=2,
        grid_points_per_axis=5,
        constraint_policy="prefer_feasible",
        stress_verbose=False,
        grid_verbose=False,
    ),
)
report = make_report(target, method="run_p1", seed=None)

print(result.status)
if result.warnings:
    print(result.warnings)
print(result.config_full["constraint_policy"])
print(result.search_settings)
print(report.best_feasible.params if report.best_feasible else None)
```

`P1Result`는 JSON으로 직렬화할 수 있습니다. 실행 결과를 남기려면 `result.save(path)`를 사용하세요.

## 제약조건 정책

- `record`: 하위 호환 모드입니다. 제약조건 위반을 audit trail에 기록하지만 best 후보 선택은 gate하지 않습니다.
- `prefer_feasible`: 일반적인 감사/CI에 권장합니다. `grid_best`를 고를 때 제약조건을 만족한 후보를 우선합니다.
- `hard_fail`: 엄격한 릴리스/CI 게이트입니다. 만족 가능한 후보가 없으면 run status를 실패로 표시합니다.

## 로컬 데모

```bash
python examples/demo_replay.py
python examples/demo_sram.py
```

두 데모는 네트워크와 API key가 필요 없습니다. Replay 데모는 sensitivity 측정, top-K unlock, grid search, walk-forward 검증, KC gate 결과, 아티팩트 출력을 보여줍니다. SRAM 데모는 6T bitcell을 여러 PVT corner에서 평가하고 선언된 제약조건과 함께 `output/audit_sram.json`을 생성합니다.

## 아티팩트가 기록하는 것

- `schema_version`, `omega_lock_version`
- `config_full`, `kc_thresholds`, `search_settings`
- 선택 후보, grid 결과, stress ranking, KC report
- `pearson_status`, `pearson_computable`을 포함한 워크포워드 증거
- `holdout_target`을 제공했을 때의 홀드아웃 증거
- 기록은 하지만 선택이나 최종 상태를 gate하지 않는 모드의 경고
- `AuditingTarget`이 남기는 feasible 후보와 absolute-best 후보의 분리

## 릴리스 기록

**0.2.2** (2026-05-22) - **Badge hardening and release-surface synchronization.** 동적 PyPI version badge를 정적 release badge로 교체해 Shields/PyPI/Camo 캐시 때문에 오래된 버전이 보이는 문제를 피했습니다. 현재 install command와 citation도 0.2.2로 동기화했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.1** (2026-05-22) - **Release sync and badge cache-bust correction.** 동적 PyPI badge URL에 release-specific cache-bust query를 추가하고, 0.2.0 업로드 이후 release metadata와 README/PyPI 표면을 다시 동기화했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.0** (2026-05-22) - **Public README and release-surface polish.** GitHub/PyPI 첫 화면을 더 직접적으로 정리하고, 없는 비디오 placeholder를 제거했으며, `Constraint`를 포함한 quickstart를 실제 API에 맞췄습니다. 영어/한국어 문서와 release checklist를 0.2.0 기준으로 갱신했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

## 릴리스 체크리스트

[RELEASE.md](RELEASE.md)를 보세요. PyPI는 이미 업로드된 버전을 덮어쓸 수 없습니다. 항상 버전을 올리고, `dist/`를 지운 뒤 fresh artifact를 만들고, 업로드 전에 검증해야 합니다.

## 라이선스

Apache 2.0. Copyright (c) 2026 hibou.
