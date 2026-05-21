# Omega-Lock

> 어떤 optimizer가 만든 후보든, 배포 전에 감사 가능한 후보인지 확인하는 **감사 우선(audit-first) 검증 레이어**입니다. 워크포워드, 제약조건, 홀드아웃, JSON 아티팩트로 튜닝 결과를 리뷰 가능한 상태로 만듭니다.

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20pyright%20%2B%20ruff-brightgreen.svg)](tests/)

처음이라면 짧은 버전부터 보세요: [EASY_README_KR.md](EASY_README_KR.md) · English: [README.md](README.md)

```bash
pip install omega-lock
```

Optuna TPE 경로까지 쓰려면:

```bash
pip install "omega-lock[p2]"
```

## 한 줄 요약

Omega-Lock은 search-first가 아니라 **audit-first**입니다.

Optimizer가 후보를 찾는 일은 기존 도구에 맡겨도 됩니다. Omega-Lock은 후보가 다음 질문을 통과하는지 확인합니다.

- 학습 데이터에서 좋았던 결과가 테스트/홀드아웃에서도 유지되는가?
- 선언한 제약조건을 만족하는가?
- 워크포워드 검증이 통과되는가?
- 나중에 리뷰 가능한 감사 아티팩트가 남는가?

즉, "가장 높은 점수를 찾는 도구"라기보다 "찾은 후보를 믿고 배포해도 되는지 판단하는 도구"입니다.

## 언제 쓰나요?

- 각 평가가 비쌉니다: SPICE류 시뮬레이션, 백테스트, LLM 호출, 모델 학습.
- 튜닝 결과를 배포 전에 기계적으로 검증해야 합니다.
- 제약조건, 워크포워드, 홀드아웃 증거가 중요합니다.
- 감사 로그를 Git에 남기고 diff하거나 재현해야 합니다.

과한 경우도 있습니다. 일회성 toy problem, 샘플이 사실상 무제한인 매끄러운 목적함수, 아무도 결과를 리뷰하지 않는 작업이라면 Optuna나 grid search를 직접 쓰는 편이 낫습니다.

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

결과물은 pass/fail 상태와 모든 평가 기록이 담긴 JSON 감사 아티팩트입니다.

## 권장 제약조건 정책

일반적인 감사/CI 사용에서는 `constraint_policy="prefer_feasible"`를 권장합니다.

- `record`: 하위 호환 기본값입니다. 제약조건 위반을 기록하지만 best 후보 선택을 막지는 않습니다.
- `prefer_feasible`: 일반 사용에 더 안전합니다. 가능한 경우 제약조건을 만족한 후보를 `grid_best`로 우선 선택합니다.
- `hard_fail`: CI/릴리스 게이트에 적합합니다. 만족 가능한 후보가 없으면 run을 실패로 표시합니다.

0.1.8부터 `record` 모드에서는 "제약조건은 기록됐지만 best-candidate selection을 gate하지 않았다"는 경고가 아티팩트에 남습니다.

## 핵심 구성요소

- `AuditingTarget`: 어떤 `CalibrableTarget`이든 감싸서 평가 trail을 남깁니다.
- `Constraint`: hard constraint를 선언하고 각 평가마다 pass/fail을 기록합니다.
- `run_p1`: stress 측정, top-K unlock, grid/zoom grid search, 워크포워드, kill criteria를 실행합니다.
- `run_p1_iterative`: 여러 라운드에 걸쳐 좌표를 잠그는 iterative 모드입니다.
- `run_p2_tpe`: 선택 의존성 `optuna`를 사용하는 TPE 검색 경로입니다.
- `holdout_target`: 최종 후보를 홀드아웃 데이터에서 한 번만 평가합니다.
- `AuditReport`: JSON으로 저장 가능한 리뷰 아티팩트입니다.

## 워크포워드와 홀드아웃

워크포워드(KC-4)는 train fitness와 test fitness의 상관, 그리고 trade/action count 비율을 확인합니다. 0.1.8부터 워크포워드 아티팩트에는 기존 numeric `pearson`과 함께 `pearson_status`, `pearson_computable`이 들어갑니다. 따라서 "상관이 낮다"와 "상관을 계산할 수 없다"를 구분할 수 있습니다.

홀드아웃은 `run_p1(..., holdout_target=...)` 또는 `run_p1_iterative(..., holdout_target=...)`로 전달합니다. 기본값인 `holdout_mode="evidence_only"`는 하위 호환을 위해 최종 status를 gate하지 않습니다. 0.1.8부터 이 경우 아티팩트에 "홀드아웃은 평가됐지만 최종 status를 gate하지 않았다"는 경고가 남습니다. 릴리스 게이트로 쓰려면 `holdout_mode="gate"`와 threshold를 설정하세요.

## 0.1.8에서 바뀐 점

**0.1.8 — Audit reliability and static hygiene release**

- `pytest`, `pyright`, `ruff` 기준이 모두 깨끗합니다.
- 현재 테스트 기준은 289 passed입니다.
- 선택 의존성인 `optuna` import typing을 정리했습니다.
- 테스트의 `CalibrableTarget` Protocol typing을 정리했습니다.
- hash-chain typing을 JSON shape 변경 없이 정리했습니다.
- 아티팩트에 재현성 필드가 추가되었습니다: `schema_version`, `omega_lock_version`, `config_full`, `kc_thresholds`, `search_settings`.
- `constraint_policy="record"`일 때 선택을 gate하지 않는다는 경고가 남습니다.
- `holdout_mode="evidence_only"`일 때 최종 status를 gate하지 않는다는 경고가 남습니다.
- iterative run에서 같은 test slice를 여러 라운드에 재사용하면 KC-4 증거력이 약해진다는 경고가 표시됩니다.
- 워크포워드 아티팩트에 `pearson_status`, `pearson_computable`이 추가되었습니다.

## 빠른 실행 경로

```bash
python examples/phantom_demo.py
python examples/demo_sram.py
```

첫 번째는 합성 keyhole 문제이고, 두 번째는 6T SRAM bitcell을 여러 PVT corner에서 평가하는 감사 데모입니다.

## 라이선스

Apache 2.0. Copyright (c) 2026 hibou.
