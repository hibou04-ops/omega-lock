# Omega-Lock

> 어떤 optimizer나 수동 튜닝 과정이 만든 후보를 배포 전에 검증하는 **옵티마이저 후단 감사 게이트**입니다. Omega-Lock은 튜닝된 후보가 일반화되는지, 선언한 제약조건 안에 있는지, 리뷰 가능한 JSON 아티팩트를 남기는지 확인합니다.

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg?cacheSeconds=60&release=0.2.1)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

```bash
pip install omega-lock
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

이 저장소에는 실제 데모 비디오 파일이 없습니다. 혼동을 주던 비디오 placeholder는 제거했습니다.

로컬에서 결정적인 데모를 실행하세요.

```bash
python examples/demo_replay.py
python examples/demo_sram.py
```

두 데모는 네트워크와 API key가 필요 없습니다. Replay 데모는 sensitivity 측정, top-K unlock, grid search, walk-forward 검증, KC gate 결과, 아티팩트 출력을 보여줍니다. SRAM 데모는 6T bitcell을 여러 PVT corner에서 평가하고 선언된 제약조건과 함께 `output/audit_sram.json`을 생성합니다.

자막/대본 파일: [docs/demo/omega-lock-demo.en.srt](docs/demo/omega-lock-demo.en.srt)

## 아티팩트가 기록하는 것

- `schema_version`, `omega_lock_version`
- `config_full`, `kc_thresholds`, `search_settings`
- 선택 후보, grid 결과, stress ranking, KC report
- `pearson_status`, `pearson_computable`을 포함한 워크포워드 증거
- `holdout_target`을 제공했을 때의 홀드아웃 증거
- 기록은 하지만 선택이나 최종 상태를 gate하지 않는 모드의 경고
- `AuditingTarget`이 남기는 feasible 후보와 absolute-best 후보의 분리

이 필드들은 CI나 릴리스 리뷰에서 실행 결과를 재현 가능하고 diff 가능한 아티팩트로 만들기 위한 것입니다.

## 언제 쓰나요?

튜닝된 후보가 실제 의사결정에 영향을 줄 때 사용하세요.

- 모델이나 전략 파라미터 튜닝
- 하드 물리 제약조건이 있는 하드웨어/시뮬레이션 calibration
- 공정 제어 또는 materials discovery
- 리뷰어가 단순 점수가 아니라 감사 가능한 아티팩트를 요구하는 optimizer governance

아무도 결과를 리뷰하지 않는 일회성 toy search에는 과할 수 있습니다.

## API 표면

핵심 프로토콜:

- `CalibrableTarget.param_space() -> list[ParamSpec]`
- `CalibrableTarget.evaluate(params: dict[str, Any]) -> EvalResult`

주요 runner:

- `run_p1`: 표준 audit gate가 붙은 grid/zoom-grid search
- `run_p1_iterative`: effective dimension이 더 큰 경우를 위한 반복 lock-in
- `run_p2_tpe`: `pip install "omega-lock[p2]"`로 사용하는 선택적 Optuna TPE 경로

감사 helper:

- `AuditingTarget`: target을 감싸 모든 평가를 기록합니다.
- `Constraint`: `(params, EvalResult) -> bool` 형태의 이름 있는 제약조건입니다.
- `make_report` / `render_scorecard`: 사람이 읽을 수 있는 요약과 JSON 감사 report를 만듭니다.

## 고급 메모

더 깊게 쓰려면 다음 기능도 사용할 수 있습니다.

- 시간, sensitivity, action count, walk-forward 증거를 보는 KC-1..4 kill criteria
- stress 측정과 top-K parameter unlock
- zooming grid와 iterative coordinate lock-in
- 선택적 random-search baseline과 benchmark scorecard
- 기존 optimizer를 붙이기 위한 adapter pattern

하지만 첫 판단은 단순합니다. 후보 생성 뒤에 Omega-Lock을 붙여 그 후보를 믿어도 되는지 확인하세요.

## 릴리스 기록

**0.2.1** (2026-05-22) - **Release sync and badge cache-bust correction.** 동적 PyPI badge URL에 release-specific cache-bust query를 추가하고, 0.2.0 업로드 이후 release metadata와 README/PyPI 표면을 다시 동기화했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.2.0** (2026-05-22) - **Public README and release-surface polish.** GitHub/PyPI 첫 화면을 더 직접적으로 정리하고, 없는 비디오 placeholder를 제거했으며, `Constraint`를 포함한 quickstart를 실제 API에 맞췄습니다. 영어/한국어 문서와 release checklist를 0.2.0 기준으로 갱신했습니다. 버전 metadata 외 런타임 동작 변경은 없습니다.

**0.1.9** (2026-05-21) - README와 PyPI long description을 정리하고, 동적 PyPI badge와 release hygiene checklist를 추가했으며, 한국어 문서를 UTF-8 Markdown으로 다시 작성했습니다.

**0.1.8** (2026-05-21) - pytest, pyright, ruff clean baseline을 만들고, 선택 의존성 `optuna` import typing, `CalibrableTarget` Protocol typing, hash-chain typing을 정리했습니다. 아티팩트에는 `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, `search_settings`와 여러 warning field가 추가되었습니다.

## 릴리스 체크리스트

[RELEASE.md](RELEASE.md)를 보세요. PyPI는 이미 업로드된 버전을 덮어쓸 수 없습니다. 항상 버전을 올리고, `dist/`를 지운 뒤 fresh artifact를 만들고, 업로드 전에 검증해야 합니다.

## 라이선스

Apache 2.0. Copyright (c) 2026 hibou.
