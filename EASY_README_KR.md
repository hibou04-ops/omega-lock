# Omega-Lock — 쉬운 설명

> 본 README가 어렵게 느껴지는 분들을 위한 압축 버전.
> 원본: [README_KR.md](README_KR.md) · English easy: [EASY_README.md](EASY_README.md)

## 이 라이브러리가 해결하는 문제

파라미터를 튜닝했습니다. 훈련 데이터에서 점수가 훌륭해요. 배포합니다. **프로덕션에서 망가집니다.**

이게 overfitting. Omega-Lock은 문 앞의 바운서입니다 — "훈련에서 잘 나옴"이 train→test→holdout 검증과 하드 제약조건을 모두 통과하지 못하면 통과시키지 않습니다.

## 60초 멘탈 모델

```
당신의 optimizer (grid / TPE / Bayesian / 자체 구현)
        ↓ 후보 제안
   [ Omega-Lock audit ]
        ↓
   PASS  →  배포 가능, 서명된 trail 포함
   FAIL  →  어느 게이트가 실패했는지 명시 (KC-1..4)
```

Optimizer는 그대로 쓰세요. Omega-Lock은 튜닝 결과가 **믿을 만한지**만 판정합니다.

## 설치

```bash
pip install omega-lock
```

## 최소 동작 예제

```python
from omega_lock import run_p1, P1Config
from omega_lock.audit import AuditingTarget, Constraint, make_report, render_scorecard

# 1. Target을 감싼다 (param_space() + evaluate() 있는 객체)
wrapped = AuditingTarget(
    my_target,
    constraints=[
        Constraint("score_positive",
                   lambda p, r: r.fitness > 0,
                   "점수는 양수여야 함"),
    ],
)

# 2. 아무 optimizer에 넘긴다. run_p1이 기본.
result = run_p1(train_target=wrapped, config=P1Config())

# 3. Audit report 획득
report = make_report(wrapped, method="run_p1", seed=42)
print(render_scorecard(report))
```

Pass/fail 스코어카드 + 모든 평가의 JSON trail이 나옵니다. Trail을 Git에 올려두세요 — 6개월 후의 당신이 고마워할 겁니다.

## 쓸 가치가 있는 경우

- 각 평가가 **비쌈** (SPICE 시뮬, 백테스트, LLM 호출)
- **Train / test** 분리, 가능하면 **holdout**까지 있음
- 누군가 튜닝 결과를 **신뢰**해야 함 (규제기관, 운영팀, 6개월 후의 나)

## 과한 경우

- 일회성 장난감 문제
- 샘플 사실상 무제한 + 목적함수가 부드러움
- 나중에 아무도 결과를 리뷰 안 함

이런 경우엔 Optuna / grid search를 그냥 직접 쓰세요.

## 입문자가 자주 걸리는 3가지

1. **"KC-4가 뭐야?"** Train fitness와 test fitness 간 Pearson 상관의 사전 선언 임계값. Train이 test를 예측 못 하면 overfit이고 run은 `FAIL:KC-4`. 협상 불가. 그게 요점입니다.
2. **"내 optimizer 써도 돼?"** 네. `CallableAdapter`로 아무 callable이나 감싸세요. Audit은 후보 출처를 신경 안 씁니다.
3. **"Optuna 필수야?"** 아니요. 기본 pipeline은 grid search. TPE 원하면 `pip install "omega-lock[p2]"`만 추가.

## 다음 단계

- **일단 돌려보기**: `python examples/phantom_demo.py` — 12-파라미터 synthetic 문제를 end-to-end
- **현실적 데모**: `python examples/demo_sram.py` — 5개 물리 corner + 3 하드 제약의 6T SRAM bitcell
- **전체 문서**: [README_KR.md](README_KR.md)에 벤치마크 수치, 철학, API 레퍼런스 수록

License: Apache 2.0. Copyright (c) 2026 hibou.
