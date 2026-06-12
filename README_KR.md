# omega-lock

**가장 높은 점수가 당신을 속이고 있습니다 — 그리고 당신의 옵티마이저는 그것을 잡아내지 못합니다.** omega-lock은 튜너가 끝난 *뒤에* 실행되는 게이트입니다. 튜너가 고른 "우승" 후보를 받아, 그 점수가 진짜인지 단순한 운인지를 — 배포 **전에** — 알려줍니다.

[![PyPI](https://img.shields.io/pypi/v/omega-lock.svg?cacheSeconds=900)](https://pypi.org/project/omega-lock/)
[![Python](https://img.shields.io/pypi/pyversions/omega-lock.svg?cacheSeconds=900)](https://pypi.org/project/omega-lock/)
[![License](https://img.shields.io/pypi/l/omega-lock.svg?cacheSeconds=900)](https://pypi.org/project/omega-lock/)

```bash
pip install omega-lock
omega-lock demo   # 60초, 오프라인: "우승" 점수가 홀드아웃 데이터에서 -74% 무너지는 것을 직접 보세요
```

> *키워드: hyperparameter overfitting · eval / prompt regression testing · walk-forward validation · validate an Optuna study · holdout transfer check in CI.*

---

## 30초 요약

당신은 하이퍼파라미터 스윕, 프롬프트 탐색, 또는 임계값 튜너를 돌렸습니다. 결과가 자신만만하게 돌아와 우승자 — **튜닝에 사용한 바로 그 데이터에서 가장 높은 점수** — 를 가리킵니다.

그 숫자가 바로 신뢰할 수 없는 숫자입니다. 수백 개의 후보를 시도하고 그중 단 하나의 최고만 남기면, 당신은 가장 실력 있는 것만 남기는 게 아니라 가장 **운 좋은** 것을 남기게 됩니다. 그리고 운은 반복되지 않습니다. 그 우승자를 한 번도 본 적 없는 데이터에서 테스트하는 순간, 운의 흐름은 사라집니다:

```
on the data it was picked from   →   5.967   (real skill  +  a lucky streak)
on brand-new, held-out data      →   1.527   (only the real skill that was left)   ▼ -74.4%
```

이것이 **선택에서 비롯된 과적합(overfitting from selection)** 이며, 어떤 옵티마이저도 이로부터 당신을 보호하지 않습니다 — 최댓값을 찾는 것이 옵티마이저의 본업이기 때문입니다. omega-lock은 당신의 두 번째 의견입니다. 탐색이 한 번도 건드리지 않은 슬라이스에서 우승자를 다시 테스트하고, 단호한 판정을 돌려줍니다: **PASS**(배포) 또는 **FAIL**(차단).

---

## 운 좋은 우승자가 떨어지는 것을 직접 보세요 — 60초, 준비물 없음

```bash
omega-lock demo
```

완전히 오프라인인 사례 연구입니다: 탐색이 학습 데이터에서 훌륭해 보이는 후보를 고르면, omega-lock이 그것을 홀드아웃 슬라이스에서 다시 채점합니다.

```
candidate: best-by-score (selected from 125 trials)
  train score    5.967
  holdout score  1.527     ▼ -74.4%
  walk-forward transfer gate ............ FAIL   (train↔holdout correlation 0.179 < 0.3)
  hard-constraint feasibility ........... FAIL   (best_feasible ≠ best_any)

VERDICT: BLOCK — the winning score did not transfer. Selection concentrated luck.
```

옵티마이저는 `5.967`에 환호했습니다. 현실은 `1.527`이었습니다. omega-lock은 `FAIL`을 찍고, 당신의 파이프라인은 배포를 멈춥니다. 그 붕괴 한 장면이 이 제품의 전부입니다.

---

## CI에 끼워 넣기

omega-lock에 점수 파일 두 개를 가리키게 하세요 — 옵티마이저가 튜닝한 데이터에서 보고한 점수, 그리고 *같은* 후보들을 홀드아웃 슬라이스에서 다시 평가한 점수입니다. 종료 코드는 `0`(배포) 또는 `1`(차단)입니다:

```bash
omega-lock gate --train train_scores.json --holdout holdout_scores.json
```

```yaml
# .github/workflows/overfit-gate.yml
name: overfit-gate
on: [pull_request]

jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install omega-lock

      # your tuner runs here and writes train_scores.json + holdout_scores.json
      - run: python tune.py

      # the gate: a non-zero exit fails the check and blocks the merge
      - run: omega-lock gate --train train_scores.json --holdout holdout_scores.json
```

홀드아웃 점수가 튜닝된 점수를 따라가지 못하면, 그 단계는 빨갛게 실패하고 PR은 머지될 수 없습니다. 모든 실행은 **추가 전용(append-only) 감사 추적**도 함께 기록하므로, 나중에 무엇이, 언제, 왜 게이트를 통과했는지 정확히 증명할 수 있습니다.

파이썬을 선호하나요? 같은 판단을 한 번의 호출로:

```python
from omega_lock.simple import gate_scores

result = gate_scores(train="train_scores.json", holdout="holdout_scores.json")
assert result.passed, result.reason   # fail your test suite on a bad candidate
```

---

## 이미 Optuna study가 있나요? 3줄로 게이팅하세요

```python
import optuna
from omega_lock import audit_optuna_study

study  = optuna.load_study(study_name="my-sweep", storage="sqlite:///sweep.db")
report = audit_optuna_study(study, holdout_evaluate=score_on_holdout)  # walk-forward + feasibility on study.best_trial
print(report.passed, report.gated_best)   # False, and the candidate it WILL certify (or None)
```

새 study도, objective 재작성도, config DSL도 필요 없습니다. 평범한 리스트에서도 동작합니다(Ax, Ray Tune, Hyperopt, `GridSearchCV`, 또는 직접 만든 스윕) — 리더보드 하나면 충분합니다.

---

## 게이트가 실제로 확인하는 것

당신의 탐색이 이미 고른 후보에 대한 세 가지 독립적인 검사입니다. 그중 어느 하나라도 후보를 차단할 수 있습니다.

| Check | Plain English | Blocks when |
|---|---|---|
| **Walk-forward transfer gate** | Does the score earned on the tuned data carry over to a held-out slice it never saw? | The held-out result decorrelates from the tuned ranking — the winner was a fluke. |
| **Hard-constraint feasibility** | Is the highest-scoring candidate also a *valid* one (passes your latency / cost / risk limits), or did you win on a config you can't run? | `best_feasible ≠ best_any` — the top score violates a constraint you declared. |
| **Append-only audit trail** | Can you reconstruct the decision months later? | Never blocks — always records the verdict, inputs, and thresholds, tamper-evident. |

**핵심 통찰:** *가장 높은 점수는 당신이 가진 가장 의심스러운 숫자입니다.* 진짜 실력은 한 번도 보여준 적 없는 슬라이스에서 살아남습니다. 운은 그렇지 못합니다.

---

## omega-lock은 또 다른 옵티마이저가 아닙니다

탐색하지도, 샘플링하지도, 아무것도 제안하지 않습니다. 이미 가지고 있는 탐색에 **볼트로 끼우는 게이트**입니다 — Optuna도, 당신의 스윕도, eval 루프도 그대로 두고, omega-lock이 그 출력을 판정하게 하세요.

| | Your optimizer (Optuna / Ax / sweep) | omega-lock |
|---|---|---|
| Job | **Finds** the best score | **Tells you if** that score deploys |
| Runs | *during* the search | *after* it, on the result |
| Looks at | the data the search consumed | a held-out slice it never saw |
| Output | a leaderboard + a winner | PASS / FAIL + the certified candidate |

### 당신이 아는 도구들 옆 어디에 위치하는가

| Tool | Its job | Overlap with omega-lock |
|---|---|---|
| **Optuna / Ax / Ray Tune** | search the space, return a winner (constrained optimization) | none — omega-lock **audits their winner** |
| **MLflow / Weights & Biases** | track *what* you ran | none — omega-lock is a **pass/fail gate**, not a tracker |
| **promptfoo / DSPy / your eval harness** | score prompt & model outputs | none — omega-lock catches the prompt that aced the eval but won't generalize |

omega-lock이 채우는 빈자리: **출력 측 과적합 게이트(output-side overfit gate)**. 경험칙 — *많은 옵션을 시도하고 그중 최고를 남겨서 고른 숫자라면, 그것은 이 게이트 뒤에 있어야 한다.* omega-lock이 더 넓은 도구 상자 안에서 어디에 맞고 어디에 맞지 않는지는 [docs/TOOLKIT_POSITIONING.md](docs/TOOLKIT_POSITIONING.md)를 보세요.

---

## 설치

```bash
pip install omega-lock

omega-lock demo                 # 60s offline walkthrough — watch a lucky winner collapse
omega-lock gate --help          # the CI gate (exit 0 = ship, 1 = block)
```

`render_html`로 어떤 게이트 실행에서든 공유 가능한 다크 테마 스코어카드를 생성하세요 — PR에 첨부하거나 보관하세요.

---

**README:** [Easy / plain-English README](EASY_README.md) · [한국어 README](README_KR.md) · [쉬운 한국어 README](EASY_README_KR.md) — **Docs:** [전이 게이트 동작 원리](docs/HOW_IT_WORKS.md) · [통합 담당자를 위한 Power API](docs/API.md) · [신뢰 & 감사 모델](docs/TRUST_MODEL.md) · [툴킷 포지셔닝](docs/TOOLKIT_POSITIONING.md) · [CHANGELOG](CHANGELOG.md)

<sub>**배지 및 다운로드 분석 경계 (Badge and download analytics boundaries).** 위 배지들은 정적이거나 레지스트리가 제공하는 링크일 뿐, 릴리스 준비도, 정확성, 신뢰성, 채택도, 패키지 품질을 증명하지 않습니다(they do not prove release readiness, correctness, trustworthiness, adoption, or package quality). 다운로드나 별(star)은 가시성을 나타낼 뿐 실력이 아닙니다(downloads or stars may indicate visibility) — 별/다운로드는 감사 증거나 릴리스 승인에 사용되어서는 안 됩니다(stars/downloads must not be used as audit evidence or release approval). 여기서는 어떤 PyPI 또는 GitHub 다운로드 분석도 주장하지 않습니다(no PyPI or GitHub download analytics are asserted here). 오직 홀드아웃 데이터에서의 게이트 PASS/FAIL만이 증거입니다.</sub>

<sub>**용어 안내.** 이 페이지는 쉬운 언어를 씁니다. 공개 파이썬 API는 하위 호환을 위해 기존 심볼을 유지합니다(다른 레포가 import합니다). 코드에서는 다음을 볼 수 있습니다: `run_p1` / `P1Config`(게이트 실행 + 설정), `check_kc4` / `KCThresholds`(walk-forward 전이 검사 + 통과 임계값, 예: 최소 전이 상관), `measure_stress`(섭동 민감도로 파라미터 순위), `ParamSpec`(튜닝 가능한 파라미터의 범위), `EvalResult`(채점된 후보 하나). `omega-lock demo`, `omega-lock gate`, 또는 `omega_lock.simple.gate_scores()`를 쓰는 데에는 이들이 전혀 필요 없습니다. 전체 레퍼런스는 [docs/API.md](docs/API.md)에 있습니다.</sub>
