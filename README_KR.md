# Omega-Lock (한국어)

[![PyPI version](https://img.shields.io/pypi/v/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg)](https://pypi.org/project/omega-lock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**민감도 기반 좌표 하강 캘리브레이션 프레임워크.**

"열쇠구멍을 거푸집으로 쓴다" — 모든 파라미터를 단단히 잠그고, 계속 부딪혀서 충격을 받는 곳 (stress 가 높은 곳) 만 풀어준다. 저차원 subspace 에서 grid 탐색 후 walk-forward 로 과적합 검증.

이 패키지는 `Omega_TB_1/research/omega_lock_p1/` (v1 HeartCore target, KC-4 FAIL) 의 방법론을 임의의 파라미터 탐색 문제에 재사용할 수 있도록 **generic 라이브러리로 추출**한 것이다. v1 HeartCore 실험은 방법론이 의도대로 overfitting 을 탐지한 사례로 종결됨 (archive/ 참조).

**English README**: [README.md](README.md)

## 목차

- [철학](#철학)
- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [Kill Criteria](#kill-criteria-사전-명문화)
- [모듈 구조](#모듈-구조)
- [검색 전략 비교](#검색-전략-비교)
- [vs External Alternatives](#vs-external-alternatives)
- [Holdout target](#holdout-target)
- [Fractal-vise 모드](#fractal-vise-모드-multi-scale-refinement)
- [테스트](#테스트)
- [제한 사항](#제한-사항)
- [다음 단계](#다음-단계-이-패키지-범위-밖)
- [Citation](#citation)
- [License](#license)

---

## 철학

대부분의 파라미터 탐색은 **차원의 저주** 에 빠진다. 22 차원 공간을 random search / TPE 로 훑어도 샘플이 부족하고, evaluation 당 비용이 커지면 iteration 이 줄어 Goodhart 국소 최적에 수렴한다.

Omega-Lock 은 가정한다:
- **effective dimension ≪ nominal dimension** (대부분의 파라미터는 결과에 거의 영향이 없다)
- 따라서 **sensitivity 를 먼저 측정** 하고, 상위 K 개만 탐색하는 것이 합리적
- **kill criteria 는 사전 명문화** — 실험자가 fudge 못 하게 함 (Winchester 방지)

이 세 가정이 맞지 않으면 Omega-Lock 은 작동 안 한다. P1 HeartCore 는 가정 1, 2 는 확인했지만 walk-forward 에서 KC-4 FAIL — 즉 3차원으로 줄여도 v1 신호층이 근본적으로 과적합 경향이 있었다. 이것 자체가 유용한 결과였다.

---

## Pipeline

```
target.evaluate(neutral_defaults)        # baseline
    ↓
for each param:                          # stress measurement (KC-2)
    perturb by ±ε, measure |Δfitness|/ε
    ↓
sort stress desc, pick top-K             # unlock set
    ↓
grid search over K-dim subspace          # train fitness
    ↓
walk-forward: top-N on test target       # KC-4 (Pearson + trade ratio)
    ↓
[optional] hybrid validation: top-K with slower judge target
    ↓
KC-1 (time box) + KC-3 (action count floor)
    ↓
P1Result (JSON-serializable)
```

---

## Quick Start

### 1. 설치

```bash
# PyPI (권장)
pip install omega-lock

# Optuna TPE (P2) 선택 기능 포함
pip install "omega-lock[p2]"

# 소스에서 (개발용)
git clone https://github.com/hibou04-ops/omega-lock.git
cd omega-lock
pip install -e ".[dev]"
```

### 2. Toy 예제 실행

```bash
python examples/rosenbrock_demo.py      # 2D Rosenbrock — grid 수렴 sanity check
python examples/phantom_demo.py         # 12-param 합성 keyhole — full P1 end-to-end
```

- `rosenbrock_demo.py` — 2D 정적 함수, walk-forward/KC-4 없음
- `phantom_demo.py` — **`PhantomKeyhole`** (12 params, 3 effective + 9 decoy, seed-driven train/test/validation). stress → top-K unlock → grid → walk-forward → hybrid 전 경로를 KC-1~4 전체 PASS로 시연. 프레임워크의 레퍼런스 열쇠구멍.

### 3. 자신의 target 구현

`CalibrableTarget` 프로토콜 구현:

```python
from omega_lock import CalibrableTarget, EvalResult, ParamSpec, P1Config, run_p1

class MyTarget:
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="threshold", dtype="float", low=0.0, high=1.0, neutral=0.5),
            ParamSpec(name="window",    dtype="int",   low=10,  high=100, neutral=50),
            ParamSpec(name="use_cache", dtype="bool",  neutral=False),
        ]

    def evaluate(self, params: dict) -> EvalResult:
        # ... your logic here ...
        return EvalResult(
            fitness=score,       # scalar to maximize
            n_trials=n_actions,  # for KC-3
            metadata={"mode": ...},
        )

result = run_p1(train_target=MyTarget())
print(result.status)               # "PASS" or "FAIL:KC-..."
print(result.grid_best["unlocked"])
```

### 4. Walk-forward

Time-series 타겟은 train / test target 을 분리해서 전달:

```python
result = run_p1(
    train_target=MyTarget(data=train_slice),
    test_target=MyTarget(data=test_slice),
    config=P1Config(trade_ratio_scale=len(test_slice) / len(train_slice)),
)
```

### 5. Hybrid fitness (A+B 패턴)

빠른 A 로 탐색, 느리지만 정밀한 B 로 top-K 재검증:

```python
# A: fast heuristic (e.g. diversity score from history)
class FastTarget:
    def param_space(self): return SHARED_SPECS
    def evaluate(self, params): return EvalResult(fitness=cheap_score(params))

# B: slow judge (e.g. LLM rubric)
class JudgeTarget:
    def param_space(self): return SHARED_SPECS
    def evaluate(self, params): return EvalResult(fitness=gemini_judge(params))

result = run_p1(
    train_target=FastTarget(),
    validation_target=JudgeTarget(),   # B 가 top-K 만 재평가
    config=P1Config(walk_forward_top_n=5),
)
# result.hybrid_top[0] 은 B 기준 1위
```

---

## Kill Criteria (사전 명문화)

| KC | 체크 시점 | 기본 임계 | 출처 |
|----|---------|--------|------|
| KC-1 | 전체 완료 후 | elapsed ≤ 3일 | time box |
| KC-2 | stress 측정 후 | Gini ≥ 0.2, top/bot ratio ≥ 2.0 | 차별화 보장 |
| KC-3 | 최종 단계 | baseline / train_best / test_best ≥ 50 trades | 통계적 유의성 |
| KC-4 | walk-forward 후 | Pearson ≥ 0.3, trade_ratio ≥ 0.5 | 과적합 방어 |

모든 임계값은 `KCThresholds` dataclass 로 override 가능. Toy 예제는 `trade_count_min=1` 등으로 완화.

---

## 모듈 구조

```
src/omega_lock/
├── target.py         # CalibrableTarget Protocol + ParamSpec + EvalResult
├── params.py         # LockedParams + clip/default_epsilon
├── stress.py         # measure_stress + gini + select_unlock_top_k
├── grid.py           # GridSearch + ZoomingGridSearch + grid_points(_in)
├── random_search.py  # RandomSearch + top_quartile_fitness + compare_to_grid (SC-2 비교)
├── walk_forward.py   # WalkForward + pearson
├── fitness.py        # BaseFitness + HybridFitness
├── kill_criteria.py  # KCThresholds + check_kc1..4
├── orchestrator.py   # run_p1() + run_p1_iterative() (+ holdout 지원)
├── p2_tpe.py         # run_p2_tpe() — Optuna TPE 연속 공간 optimizer (optional dep)
└── keyholes/
    ├── phantom.py        # PhantomKeyhole — effective_dim 3 / nominal 12 (happy-path demo)
    └── phantom_deep.py   # PhantomKeyholeDeep — effective_dim 6 / nominal 20 (iteration 필수)
```

## 검색 전략 비교

| 방법 | 연속성 | 해상도 | 용도 |
|---|---|---|---|
| `GridSearch` | 이산 | 1 round × $n^K$ | 빠른 첫 탐색 |
| `ZoomingGridSearch` | 이산 (기하급수 축소) | $n^K \times r$ rounds | 격자 밖 값까지 refine |
| `RandomSearch` | 이산/연속 혼합 | 동일 예산 랜덤 sampling | SC-2 baseline (grid top-q ≥ 1.5× random) |
| `run_p2_tpe` (Optuna) | 완전 연속 | TPE 적응적 | 진짜 연속 공간 optimizer, optional `pip install "omega-lock[p2]"` |

## vs External Alternatives

| Tool | 접근 | Omega-Lock 과의 차이 |
|---|---|---|
| Optuna / Hyperopt (TPE) | Bayesian 적응 sampling, full-dim | Omega-Lock 은 stress 로 top-K subspace 먼저 고정 후 탐색. `effective_dim ≪ nominal_dim` 가정 맞을 때 sample efficiency 월등. 둘이 상호보완이라 `run_p2_tpe` 로 wrap 가능. |
| Ray Tune / scikit-optimize | 범용 HPO 프레임워크 | 단일 fitness, walk-forward / 과적합 검증 미내장. Omega-Lock 은 KC-4 (Pearson + trade_ratio) 를 필수 gate 로 내장. |
| 단순 grid search | exhaustive | high-dim 폭발 ($n^D$). Omega-Lock 은 stress → top-K unlock 으로 $n^K$ 로 축소. |
| Nelder-Mead / Powell | 국소 연속 search | 저차원 연속 전용, categorical / bool 안됨. Omega-Lock 은 int / bool / continuous 혼합 지원. |

**Omega-Lock 의 USP**: *pre-declared kill criteria + low-dim subspace hypothesis*. 적응 샘플링 optimizer 가 아니라 **방법론 프레임워크**. 이미 훌륭한 optimizer 들 (TPE / Bayesian / Genetic) 위에 얹어 쓰는 것이 이상적 — `run_p2_tpe` 가 그 예.

## Holdout target

`run_p1(..., holdout_target=T3)` 또는 `run_p1_iterative(..., holdout_target=T3)` 에 **round 중 한 번도 건드리지 않는 3번째 target** 을 넘기면, 최종 `grid_best` 또는 `final_baseline` 을 딱 1회 평가해서 `holdout_result` 에 기록. Iterative 모드에서 test_set이 각 라운드의 lock-in 결정에 재사용되는 문제 (KC-4 증거력 약화) 의 honest 보조 검증.

## Fractal-vise 모드 (multi-scale refinement)

프랙탈 바이스 비유: 큰 segment로 물체를 먼저 감싸고 (round 1 lock-in), 더 작은 segment들이 그 좌표계 안에서 세부 형상을 conform (zooming within round, or next round on remaining params).

두 가지 독립 축:

1. **Iterative lock-in** (`run_p1_iterative` + `IterativeConfig`):
   Round 1 에서 top-K unlock 후 grid best 을 lock → Round 2 는 나머지 param 에서 stress 재측정 → 반복. `effective_dim > unlock_k` 인 target 에서 가치 발생.

2. **Zooming grid** (`ZoomingGridSearch`, 또는 `P1Config(zoom_rounds=N)`):
   한 round 내에서 grid 를 이전 winner 중심으로 기하급수 축소. 이산 grid 값(예: alpha=0.5) 대신 finer 값(예: alpha=0.4375) 도달. 2 zoom 라운드마다 오차 ~4× 감소.

둘은 조합 가능. `run_p1_iterative(config=IterativeConfig(rounds=3, zoom_rounds=4))` 가 완전체 프랙탈 바이스. PhantomKeyhole 데모에서 plain grid (alpha=0.5, fitness=12.0) vs fractal (alpha=0.4375, fitness=13.0) 대조 확인 가능.

**주의**: KC threshold 는 매 round 엄격 유지. Winchester 방지. `test_set` 은 rounds 전반에 재사용되므로 `KC-4` PASS 는 round 가 깊어질수록 약한 증거 — 실무에서는 hold-out set 분리 권장.

---

## 테스트

```bash
pip install -e ".[dev]"
pytest tests/                    # 전체
pytest tests/test_stress.py -v   # 개별
pytest --cov=omega_lock          # 커버리지
```

---

## 제한 사항

- **Determinism 가정**: stress 측정은 target 이 결정론적일 때만 정확. 비결정적 target 은 seed 고정 또는 복수 평가 평균 권장.
- **OFI-biased 파라미터**: 특정 파라미터의 stress 가 환경 제약으로 인위적으로 낮게 측정되는 경우, `ParamSpec(ofi_biased=True)` 로 표기. 결과에 flag 되지만 자동 필터링은 안 함 (관찰용).
- **Continuous + int 혼합**: epsilon 은 type-aware (continuous = 10% range, int = 1, bool = flip). Custom epsilon 은 `StressOptions(epsilons={...})` 로 지정.
- **Grid 차원 폭발**: K=3 / 5 points/axis = 125 combos. K 가 커지면 grid 대신 Optuna TPE 같은 adaptive 탐색이 나음 (현재 P2 TPE 는 범위 밖, 향후 추가 가능).

---

## 다음 단계 (이 패키지 범위 밖)

- **Omega_X 어댑터**: `adapters/omega_x/` 에 `SelectorTarget`, `ValidationTarget` 구현 → X thread pipeline calibration
- **P2 Optuna TPE**: `orchestrator.run_p2()` — grid 대신 적응적 탐색
- **P3 enrichment**: bookDepth/aggTrades 에서 faithful OFI 복원 (HeartCore 전용)
- **Random-search baseline**: SC-2 "top-quartile ≥1.5× random" 를 실제로 비교 (P1 에서 누락됨)

---

## Citation

연구 또는 발표 프로젝트에서 Omega-Lock 을 사용한 경우:

```bibtex
@software{omega_lock_2026,
  author  = {hibou},
  title   = {Omega-Lock: Sensitivity-driven coordinate descent calibration framework},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

---

## Archive (private — not in public repo)

프레임워크의 방법론적 출처인 **Omega-Lock P1 HeartCore** 적용 사례 (2026-04-13 ~ 04-14) 는 별도 로컬 보관 (`archive/` 디렉토리, `.gitignore` 처리).

- `P1_HeartCore_SPEC.md` — 21-param v1 HeartCore 대상 원래 설계 문서
- `P1_HeartCore_RESULT.md` — KC-4 FAIL 결과 리포트 (Pearson 0.119, train/test 과적합 탐지 성공)

두 문서는 **불변**이며, 방법론이 의도대로 overfitting 을 탐지한 **최초 사례 기록**으로 보존. 외부에 공개되지 않음 (Omega_TB_1 내부 연구 + BTCUSDT 실데이터 참조 포함).

---

## License

MIT License. 자세한 내용은 [LICENSE](LICENSE) 참조.

Copyright (c) 2026 hibou.
