# Omega-Lock (한국어)

[![PyPI version](https://img.shields.io/pypi/v/omega-lock.svg?v=0.1.2)](https://pypi.org/project/omega-lock/)
[![Python versions](https://img.shields.io/pypi/pyversions/omega-lock.svg?v=0.1.2)](https://pypi.org/project/omega-lock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sensitivity 기반 캘리브레이션 프레임워크 + 재사용 가능한 audit surface.**

2026-04-18 단일 세션에서 Claude Opus 4.7 와 함께 빌드. PyPI 에 `omega-lock` 으로 배포. 149 tests 통과, 30-run 객관 benchmark, CI regression guard.

0.1.2 에 실제로 포함된 것:

- **통합 search pipeline 3개**, 동일한 verification surface 공유: `run_p1` (grid / zooming grid), `run_p1_iterative` (multi-round lock-in), `run_p2_tpe` (Optuna TPE, optional extra).
- **재사용 가능한 audit 컴포넌트**, 모든 pipeline 에 연결: perturbation sensitivity (stress), walk-forward 상관, 사전 명문화 kill criteria (KC-1..4), optional 단일 holdout, Bergstra-Bengio random-search advisory (SC-2), RAGAS-style 객관 scorecard.
- **Bring-your-own-search hook** — `CallableAdapter` 로 임의 외부 optimizer 를 `CalibrableTarget` 으로 wrap 해서 같은 pipeline 에 태움.
- **2개 reference keyhole**, ground-truth 메서드 탑재로 기계적 benchmark 채점 가능.

Origin: KC-4 FAIL 로 종결된 거래전략 캘리브레이션 실험에서 distill — 방법론이 설계대로 overfitting 을 탐지한 그 controlled-failure outcome 이 프레임워크가 만들어내도록 설계된 동작 자체다. 분리된 `omega_lock.audit.run_audit(external_candidate, environments)` API 는 [해커톤 주간 계획](#해커톤-주간-계획-2026-04-21--28) 에 있음, 0.1.2 에 없음.

English README: [README.md](https://github.com/hibou04-ops/omega-lock/blob/main/README.md)

## At a Glance

| | |
|---|---|
| 뭔가 | 3개 통합 search pipeline + 공유 audit surface (stress, walk-forward, KCs, holdout, 객관 scorecard) |
| 왜 중요 | 어떤 pipeline 이 후보를 만들었든 audit 은 동일하게 작동. "뭔가 찾음" 과 "일반화됨" 을 분리 |
| 언제 쓰나 | Fitness 평가 비용 큼, train/test (가능하면 holdout) 분리됨, 단일 점수 아닌 일반화 pass/fail 기계 검증 원할 때 |
| 언제 안 쓰나 | Effective dim ≈ nominal dim, 샘플 무제한, out-of-sample 안정성 무관심 → stock optimizer 로 충분 |
| 설치 | `pip install omega-lock` (기본) 또는 `pip install "omega-lock[p2]"` (Optuna TPE 포함) |
| Core API | `run_p1` · `run_p1_iterative` · `run_p2_tpe` · `run_benchmark` · `CallableAdapter` |
| 상태 | 0.1.2 on PyPI · 149 tests 통과 · 30-run benchmark gold baseline CI regression guard 동결 |
| Built | 2026-04-18 · 단일 세션 · Claude Opus 4.7 |

### Raw benchmark scorecard (30 runs: 2 keyholes × 3 methods × 5 seeds)

`examples/benchmark_battery.py` 의 실제 출력. cherry-pick 없음.

```
keyhole                method          recall  L2err  fit_gap%  gen_gap  pass%
PhantomKeyhole         plain_grid      1.00    0.24    -9.3     1.26     60%
PhantomKeyhole         fractal_vise    0.60    0.50   -16.6     1.13     60%
PhantomKeyhole         optuna_tpe      1.00    0.07   -22.1     1.10      0%
PhantomKeyholeDeep     plain_grid      0.50    1.86   +73.9     0.66     60%
PhantomKeyholeDeep     fractal_vise    0.20    1.51   +45.9     0.51     20%
PhantomKeyholeDeep     optuna_tpe      0.50    1.87   +70.0     0.61     20%
```

수치가 실제로 말하는 것:

- **어떤 단일 search 도 지배적이지 않다.** plain grid 가 pass rate 최고, TPE 가 PhantomKeyhole 에서 최적점 가장 가깝게 (L2err 최저) 찍지만, fractal-vise (iterative lock-in) 는 단일 라운드 grid 대비 엄밀한 개선 아님. 이건 버그가 아니라 실측 결과.
- **Stress ranking 은 method 관계없이 안정적.** 30 runs 전체 Spearman ρ(측정 stress, 실제 importance) ≈ **0.95**. 오래된 lock-by-weight 아이디어에서 여전히 값을 하는 부분 — **저렴하고 정확한 screening**.
- **Search 가 놓친 것을 audit 이 잡는다.** Optuna TPE 가 PhantomKeyhole 에서 진짜 optimum 에 가장 근접했는데 **pass_rate = 0%**: walk-forward 가 정밀 overfit 을 정확히 flag. "뭔가 찾음" 과 "그게 일반화됨" 사이의 이 분리 — 프레임워크가 존재하는 이유.

프레임워크는 3개 통합 search pipeline 을 ship. 각각이 동일한 audit 컴포넌트 (stress / walk-forward / KCs / holdout / benchmark) 를 재사용. 위 benchmark 는 같은 keyhole, 같은 gate 아래에서 세 방법을 비교한 결과.

## 목차

- [철학](#철학)
- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [해커톤 주간 계획 (2026-04-21 ~ 28)](#해커톤-주간-계획-2026-04-21--28)
- [Kill Criteria](#kill-criteria-사전-명문화)
- [모듈 구조](#모듈-구조)
- [검색 전략 비교](#검색-전략-비교)
- [vs External Alternatives](#vs-external-alternatives)
- [Holdout target](#holdout-target)
- [Fractal-vise 모드](#fractal-vise-모드-multi-scale-refinement)
- [객관 Benchmark (RAGAS-style)](#객관-benchmark-ragas-style)
- [Adapter 패턴](#adapter-패턴)
- [테스트](#테스트)
- [제한 사항](#제한-사항)
- [로드맵](#로드맵)
- [Citation](#citation)
- [License](#license)

---

## 철학

대부분의 파라미터 탐색은 **차원의 저주** 에 빠진다. 22 차원 공간을 random search / TPE 로 훑어도 샘플이 부족하고, evaluation 당 비용이 커지면 iteration 이 줄어 Goodhart 국소 최적에 수렴한다.

Omega-Lock 은 가정한다:
- **effective dimension ≪ nominal dimension** (대부분의 파라미터는 결과에 거의 영향이 없다)
- 따라서 **sensitivity 를 먼저 측정** 하고, 상위 K 개만 탐색하는 것이 합리적
- **kill criteria 는 사전 명문화** — 실험자가 fudge 못 하게 함 (Winchester 방지)

이 세 가정이 맞지 않으면 Omega-Lock 은 작동 안 한다. 원 case study 에서 가정 1, 2 는 확인됐지만 walk-forward 에서 KC-4 FAIL. 3차원으로 줄여도 밑단 신호층이 과적합이었다. 프레임워크가 그걸 잡아낸 것이 애초의 기능이다.

---

## Pipeline

두 계층: **inner pipeline** (한 라운드의 stress → unlock → search → verify) 과 **outer loop** (inner pipeline 을 반복하며 매 라운드 winner 를 lock-in 하는 프랙탈 바이스 coordinate descent).

### Inner pipeline (`run_p1`)

```
target.evaluate(baseline_params)              # baseline (neutrals 또는 이전 라운드의 locks)
    ↓
for each unlocked param:                      # stress (KC-2)
    perturb by ±ε, measure |Δfitness|/ε
    ↓
sort stress desc, pick top-K                  # unlock set
    ↓
search over K-dim subspace                    # train fitness
    GridSearch         ─ 1 round × n^K                       (기본)
    ZoomingGridSearch  ─ r rounds, range × zoom_factor       (프랙탈 refinement)
    run_p2_tpe         ─ Optuna TPE, 완전 연속                (옵션)
    ↓
walk-forward on test_target                   # KC-4 (Pearson + trade ratio)
    ↓
[옵션] hybrid re-rank with judge target       # 느리지만 정밀한 B 로 top-K 재평가
    ↓
[옵션] SC-2 advisory                          # grid top-q vs random top-q (Bergstra-Bengio)
    ↓
KC-1 (time box) + KC-3 (action count floor)
    ↓
[옵션] holdout_target 딱 1회 평가              # 정직한 out-of-sample
    ↓
P1Result (JSON-serializable)
```

### Outer loop (`run_p1_iterative`)

```
base_params = neutral_defaults
locked = {}
for round r in 0..max_rounds:
    remaining = all_params - locked
    result = run_p1(target, base_params, subset=remaining)
    if result.status != "PASS":  break   # Winchester 방어
    if improvement < min_improvement:  break
    이번 라운드 winner 를 base_params 에 lock
    ↓
final_baseline (모든 locked 값) + per-round P1Results + holdout_result
```

각 라운드의 KC-1..4 는 독립 적용 — threshold 는 **라운드 간 절대 완화 안 됨** (Winchester 방지). Outer loop 는 첫 실패 라운드에서 정지.

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
python examples/full_showcase.py        # 5-mode 종합: plain / fractal / random / TPE / deep-iteration
python examples/benchmark_battery.py    # RAGAS-style 객관 scorecard (methods × keyholes × seeds)
python examples/adapter_example.py      # 임의 외부 시스템을 CalibrableTarget 으로 wrap
```

- `rosenbrock_demo.py` — 2D 정적 함수, walk-forward/KC-4 없음
- `phantom_demo.py` — **`PhantomKeyhole`** (12 params, 3 effective + 9 decoy). stress → top-K unlock → grid → walk-forward → hybrid 전 경로, KC-1~4 전원 PASS. 레퍼런스 열쇠구멍
- `full_showcase.py` — 모든 search 모드를 두 레퍼런스 키홀에 대해 실행 + 결과 나란히 출력
- `benchmark_battery.py` — 모든 method × keyhole × seed 조합 돌려서 객관 scorecard 출력 (effective_recall, param_L2_error, fitness_gap, generalization_gap, stress_rank_spearman, pass_rate)
- `adapter_example.py` — 외부 시스템 wrap 2 패턴: `CallableAdapter` (순수 함수용 one-liner) + stateful class template

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

### 6. Fractal-vise 모드 (iterative lock-in + zooming grid)

`effective_dim > unlock_k` 인 경우, 단일 라운드 grid 는 K 개만 포착하고 나머지는 neutrals 상태로 방치됨. Iterative orchestrator 는 매 라운드 winner 를 lock 하고 남은 파라미터에서 stress 재측정 → 다음 effectives 가 표면에 떠오름. Zooming 은 각 winner 주위로 격자를 기하급수 축소해서 최종 값이 coarse lattice 에 갇히지 않게 함.

```python
from omega_lock import IterativeConfig, KCThresholds, run_p1_iterative

result = run_p1_iterative(
    train_target=MyTarget(),
    test_target=MyTargetAtDifferentSlice(),
    holdout_target=MyTargetAtThirdSlice(),          # 라운드 중엔 안 건드리고 마지막 1회만 평가
    config=IterativeConfig(
        rounds=3,
        per_round_unlock_k=3,
        zoom_rounds=4,          # 각 라운드 내 기하급수 refinement
        zoom_factor=0.5,        # 매 zoom 패스마다 범위 절반으로
        min_improvement=0.5,
        kc_thresholds=KCThresholds(trade_count_min=50),
    ),
)

print(result.final_status)                # 모든 라운드가 KC-1..4 PASS 면 "PASS"
print(result.locked_in_order)             # [['alpha', 'long_mode', 'beta'], ['window', 'use_ema', 'horizon'], ...]
print(result.round_best_fitness)          # [32.4, 143.6, 143.61]  — 각 라운드의 grid_best
print(result.holdout_result)              # {'fitness': 144.41, 'n_trials': ..., 'params': ...}
```

### 7. Optuna TPE (연속 공간 탐색)

`pip install "omega-lock[p2]"` 로 설치. TPE 가 grid 를 대신해 Bayesian 적응 sampling 수행.

```python
from omega_lock import P2Config, run_p2_tpe

result = run_p2_tpe(
    train_target=MyTarget(),
    test_target=MyTargetAtDifferentSlice(),
    config=P2Config(unlock_k=3, n_trials=200, seed=42),
)
# 동일한 KC-1..4 gate — TPE 는 search 방식 교체일 뿐 threshold 완화 아님.
```

---

## 해커톤 주간 계획 (2026-04-21 ~ 28)

이 repo 는 2026-04-18 하루 세션에 Claude Opus 4.7 와 같이 빌드됐다. 아래는 **오늘 0.1.2 에 실제 포함된 것** vs **Anthropic "Built with Opus 4.7" 해커톤 주간에 추가될 것** 의 정직한 구분.

### 0.1.2 에 이미 ship 중인 것 (오늘)

- `run_p1`, `run_p1_iterative`, `run_p2_tpe` 통합 pipeline
- Stress 측정, walk-forward, KC-1..4, holdout 지원, SC-2 advisory
- `run_benchmark` + 30-run 객관 scorecard + gold baseline regression guard
- `CallableAdapter` (외부 optimizer wrap 용)
- 2개 reference keyhole (`PhantomKeyhole`, `PhantomKeyholeDeep`), ground-truth method 탑재
- 149 tests, PyPI 배포, MIT license

### 해커톤 주간에 추가할 것

1. **`omega_lock.audit.run_audit(candidate, environments, config) -> AuditResult`** — 분리된 audit entrypoint. 임의 소스 (Optuna, 벤더 default, 수동 튜닝, LLM 제안) 에서 온 candidate 를 받아 local stress + per-environment walk-forward + KC gate + holdout 돌림. 반환: pass/fail + per-environment 분해 + trust score + 실패 사유. 기존 stress / walk_forward / kill_criteria / benchmark 코드 재활용; 신규는 thin wrapper + trust-score aggregator.
2. **`omega-lock audit --explain` CLI** — 사람이 읽는 per-corner 출력 + JSON 모드.
3. **`omega_lock.keyholes.sram_bitcell`** — 6T SRAM bitcell 데모 keyhole. 분석식 모델 (SPICE 없음), <1ms/eval, 5 PVT corner (TT / SS / FF / FS / SF). Overfit pathology 는 physics-informed: TT corner 에 최적화된 candidate 가 transistor strength ratio 때문에 SS/FF 에서 systematically 무너짐. **거래전략 캘리브레이션 죽이는 패턴이 실리콘 tape-out 도 죽인다는 것** 의 구체 증명.
4. **`examples/demo_sram.py`** — end-to-end 60초 데모: Optuna TPE 가 TT corner optimum 찾음, `run_audit` 가 SS corner 에서 KC-4 Pearson 0.11 로 reject, radar chart `corners.png` 저장 (TT spike + 4 corner 붕괴 시각화).
5. **PyPI 0.1.3 릴리즈** (~4/25) — 위 전부 포함. shields.io 배지 cache 도 부수적으로 갱신됨.

### 왜 이 데모인가

`omega-lock` 의 origin 은 한 도메인 (거래전략) 의 캘리브레이션 실험이 자기 overfitting check 에서 실패한 것. 같은 방법론을 **전혀 다른 도메인** (low-power SRAM bitcell 사이징) 에 적용하면 두 가지가 구체화된다:

1. Train-corner overfit pathology 는 도메인에 의존하지 않는다. Backtest window 에 과적합한 거래전략을 잡는 그 KC-4 gate 가, typical 공정에서 최적화되어 slow-slow corner 에서 죽는 bitcell 도 잡는다.
2. Audit surface 는 재사용 가능하다. `run_audit` 로 분리하면 프레임워크가 "각자 verification 를 포함한 3개 pipeline" 에서 "임의 소스의 후보를 동일한 기계적 check 로 검증" 으로 전환된다.

데모가 증거. README 는 pointer.

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
├── random_search.py  # RandomSearch + top_quartile_fitness + compare_to_grid (SC-2)
├── walk_forward.py   # WalkForward + pearson
├── fitness.py        # BaseFitness + HybridFitness
├── kill_criteria.py  # KCThresholds + check_kc1..4 (+ KCStatus "ADVISORY" for SC-2)
├── orchestrator.py   # run_p1 + run_p1_iterative (+ holdout + SC-2 wire-in)
├── p2_tpe.py         # run_p2_tpe — Optuna TPE 연속 공간 optimizer (optional dep)
├── adapters.py       # CallableAdapter — 임의 callable 을 CalibrableTarget 으로 wrap
├── benchmark.py      # run_benchmark + BenchmarkReport — RAGAS-style 객관 scorecard
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

## 객관 Benchmark (RAGAS-style)

"PASS 하는가?" (binary KC gate) 는 필요하지만 충분하지 않음. Method 간 비교나 조용한 regression 탐지를 위해 Omega-Lock 은 **기계적으로 계산 가능한 scorecard** 제공 — 모든 지표는 run 출력 + keyhole ground truth 로 자동 계산 (사람 판단 개입 없음).

| 지표 | 정의 | 목표 |
|---|---|---|
| `effective_recall` | \|found ∩ true_effective\| / \|true_effective\| | → 1.0 |
| `effective_precision` | \|found ∩ true_effective\| / \|found\| | → 1.0 |
| `param_L2_error` | 정규화 L2 (found vs true optimum) | → 0.0 |
| `fitness_gap_pct` | `(optimum − found) / |optimum|` | ≤ 0 (found 가 기준 초과) |
| `generalization_gap` | `|train_best − test_best| / |train_best|` | 작을수록 좋음 |
| `stress_rank_spearman` | ρ(측정 stress ranking, 진짜 importance ranking) | → 1.0 |
| `pass_rate` | `status == "PASS"` 인 run 비율 | — |
| `walltime_s` / `n_evaluations` | 효율 | — |

```python
from omega_lock import BenchmarkSpec, CalibrationMethod, run_benchmark
from omega_lock.keyholes.phantom import PhantomKeyhole

spec = BenchmarkSpec("PhantomKeyhole", PhantomKeyhole, seeds=[42, 7, 100, 314, 55])
methods = [
    CalibrationMethod("plain_grid",   runner=lambda t, s: _wrap_p1(run_p1(t, ...))),
    CalibrationMethod("fractal_vise", runner=lambda t, s: _wrap_iter(run_p1_iterative(t, ...))),
]

report = run_benchmark([spec], methods, output_path=Path("bench.json"))
print(report.render_scorecard())
```

샘플 출력 (10 runs 통합):

```
method              recall  prec   L2err  fit_gap%  gen_gap  pass%
plain_grid          0.750   1.000  1.052  32.3%     0.958    60.0%
fractal_vise        0.400   0.217  1.003  14.7%     0.820    40.0%
optuna_tpe          0.750   1.000  0.970  23.9%     0.858    10.0%
```

표 읽는 법: TPE 가 `L2err` 가장 낮고 (진짜 optimum 에 가장 근접) `pass_rate` 도 최저. 프레임워크가 TPE 의 정밀한 overfit 을 잡아내는 증거다. `plain_grid` 는 coarse 한 만큼 overfit 하기 어려워서 pass 가 잦다. `fractal_vise` 는 precision 을 라운드에 걸친 넓은 coverage 와 교환한다.

**CI regression guard**: `tests/test_benchmark_regression.py` 가 현재 run 을 frozen `tests/fixtures/benchmark_gold.json` 과 비교. 결정론적 지표의 drift `1e-6` 초과 시 test FAIL. 의도적 재생성: `OMEGA_LOCK_UPDATE_GOLD=1 pytest tests/test_benchmark_regression.py`.

---

## Adapter 패턴

임의 외부 시스템을 `CalibrableTarget` 으로 wrap. 2 가지 idiomatic 패턴, 둘 다 `examples/adapter_example.py` 에 수록.

### Pattern 1: `CallableAdapter` (순수 함수용 one-liner)

```python
from omega_lock import CallableAdapter, ParamSpec, run_p1

def external_score(params: dict) -> float:
    return -((params["a"] - 3.0) ** 2 + (params["b"] - 7.0) ** 2)

target = CallableAdapter(
    fitness_fn=external_score,
    specs=[
        ParamSpec(name="a", dtype="float", low=0.0, high=10.0, neutral=5.0),
        ParamSpec(name="b", dtype="float", low=0.0, high=10.0, neutral=5.0),
    ],
)

result = run_p1(train_target=target, config=P1Config(unlock_k=2, zoom_rounds=4))
```

### Pattern 2: Stateful class (setup 비용이 있는 시스템)

target 이 내부 상태 (trained model, preloaded data, active session) 를 가질 때는 `param_space()` + `evaluate()` 직접 구현. `examples/adapter_example.py` 의 template 이 전체 모양 보여줌.

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
- **Suppressed-stress 플래그**: 특정 파라미터의 stress 가 환경 제약으로 인위적으로 낮게 측정될 때 (예: 측정 시점에 상위 서브시스템이 mock/비활성화), `ParamSpec(ofi_biased=True)` 로 표기. 결과에 flag 로 남고 자동 필터링은 안 함 (관찰 용도).
- **Continuous + int 혼합**: epsilon 은 type-aware (continuous = 10% range, int = 1, bool = flip). Custom epsilon 은 `StressOptions(epsilons={...})` 로 지정.
- **Grid 차원 폭발**: K=3 / 5 points/axis = 125 combos. K 가 커지면 grid 대신 Optuna TPE 같은 adaptive 탐색이 나음 (현재 P2 TPE 는 범위 밖, 향후 추가 가능).

---

## 로드맵

### 현재 버전에 포함됨

- ✅ **Iterative coordinate descent** — `run_p1_iterative`, multi-round lock-in
- ✅ **Zooming grid** — `ZoomingGridSearch`, 라운드 내 기하급수 refinement
- ✅ **Optuna TPE (P2)** — `run_p2_tpe`, 연속 공간 search (`pip install "omega-lock[p2]"`)
- ✅ **Random-search baseline** — `RandomSearch` + `compare_to_grid`, `run_p1` 에 SC-2 advisory gate wire-in
- ✅ **Holdout target** — 단일 out-of-sample 평가, 라운드 중엔 절대 안 건드림
- ✅ **객관 benchmark** — `run_benchmark` + `BenchmarkReport`, RAGAS-style scorecard + CI regression guard
- ✅ **Adapter 패턴** — `CallableAdapter` + stateful-class template

### 여전히 범위 밖 (application-specific)

- **도메인 특화 adapter** — 특정 외부 시스템 (trading 전략, ML 모델, 시뮬레이션) 을 `CalibrableTarget` 으로 wrap 하는 것은 generic 라이브러리 영역 밖. 일반 패턴은 `CallableAdapter` + `examples/adapter_example.py` 의 stateful-class template 참조
- **Ensemble-averaged `evaluate` helper** — 비결정적 target 용. 현재 `CalibrableTarget` docstring 에 "ensemble averages 권장" 명시만 있고 helper 미탑재. 실제 use case 생기면 추가

---

## Citation

연구 또는 발표 프로젝트에서 Omega-Lock 을 사용한 경우:

```bibtex
@software{omega_lock_2026,
  author  = {hibou},
  title   = {Omega-Lock: Sensitivity-driven coordinate descent calibration framework},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/hibou04-ops/omega-lock}
}
```

---

## License

MIT License. 자세한 내용은 [LICENSE](https://github.com/hibou04-ops/omega-lock/blob/main/LICENSE) 참조.

Copyright (c) 2026 hibou.
