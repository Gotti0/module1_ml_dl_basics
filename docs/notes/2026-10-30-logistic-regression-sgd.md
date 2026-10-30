# 학습 노트: 회귀 복습 → 로지스틱 회귀 → 확률적 경사 하강법

- 날짜: 2026-10-30
- 교재: 『혼자 공부하는 머신러닝+딥러닝』 3장 복습, 4-1 로지스틱 회귀, 4-2 확률적 경사 하강법(도입)
- 관련 실습: `3-1.ipynb` ~ `3-3.ipynb`, `4-1.ipynb`, `4-1_extra_practice.ipynb`, `4-2.ipynb`, `4-2_extra_practice.ipynb`
- 표에 적힌 수치는 교재 출력 기준이며, 실행 환경에 따라 소수점 아래가 조금 다를 수 있다.

---

## 1. 큰 그림 (어제 복습)

```
인공지능 ⊃ 기계학습 ⊃ 딥러닝
기계학습 ─┬─ 지도학습   : 정답(타깃)이 있음 → 분류, 회귀
          ├─ 비지도학습 : 정답이 없음 → 군집, 차원 축소
          └─ (강화학습)
```

### 회귀 모델의 한계와 다음 단계

| 모델 | 방식 | 한계 |
|---|---|---|
| KNN 회귀 | 이웃 k개 타깃의 평균 | 훈련 데이터 범위 밖을 예측하지 못함(외삽 불가) |
| 선형 회귀 `y = ax + b` | 데이터에 맞는 직선 | 휘어진 데이터에서 과소적합, 음수 무게 예측 |
| 다항 회귀 `y = ax² + bx + c` | `x²` 같은 특성을 추가해 곡선으로 맞춤 | 차수가 너무 높으면 과대적합 |
| 규제 (Ridge / Lasso) | 계수 크기에 벌점 | `alpha`가 클수록 규제가 강함, 사전에 `StandardScaler` 필요 |

- 다항 회귀도 **선형 회귀**다. `x²`를 새 변수로 보면 각 특성과 계수의 선형 결합이기 때문이다.
- 과소적합은 훈련·테스트 점수가 모두 낮은 상태, 과대적합은 훈련 점수만 높고 테스트 점수가 크게 낮은 상태다.

---

## 2. 분류 문제와 확률

- 회귀와 분류를 가르는 기준은 출력이 숫자인지가 아니라 **무엇을 예측하는가**다. 생선의 종류를 예측하므로 분류다.
- 분류 모델도 **각 클래스에 속할 확률**을 계산한다. 확률이 가장 높은 클래스가 최종 예측이 된다.

### 데이터 준비

```python
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
print(pd.unique(fish['Species']))   # 7종: Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt

fish_input  = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
```

- `random_state=42`는 재현성을 위한 시드다. 클래스 비율을 고르게 나누려면 `stratify=`를 쓴다.

### 표준화: 기준은 하나, `fit`은 훈련 세트에만

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled  = ss.transform(test_input)
```

- 테스트 세트에 따로 `fit`하면 두 세트가 서로 다른 기준으로 변환되고, 테스트 정보가 새어 들어간다(데이터 누수).
- 표준화는 평균 0, 표준편차 1로 맞출 뿐 **−1~1로 제한하지 않는다**. 0~1로 맞추는 것은 정규화(Min-Max)다.

---

## 3. KNN 분류기의 확률

```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.classes_)                   # 알파벳순, predict_proba 열 순서와 같음
print(kn.predict(test_scaled[:5]))
proba = kn.predict_proba(test_scaled[:5])   # 5행 7열

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])         # 예: [['Roach' 'Perch' 'Perch']] → Perch 0.6667
```

- KNN의 확률은 **가까운 이웃 중 해당 클래스가 차지하는 비율**이다.
- 이 값은 "실제로 정답일 가능성"을 보증하지 않는다.
- k=3이면 가능한 확률은 0, 1/3, 2/3, 1의 네 가지뿐이다.
- 확률을 더 세밀하게 표현한다고 예측 성능이 더 높다는 보장은 없다.

---

## 4. 로지스틱 회귀 (이진 분류)

### 3단계 구조

```
표준화된 특성 5개
   │  × 계수 5개 + 절편        ← 모델 파라미터 (훈련으로 학습)
   ▼
점수 z = decision_function()   ← 확률이 아님, −∞ ~ +∞
   │  시그모이드 expit()
   ▼
양성 클래스 확률 = predict_proba()[:, 1]   ← 0 ~ 1
   │  0.5 기준
   ▼
최종 클래스 = predict()
```

$$z = a \cdot \text{Weight} + b \cdot \text{Length} + c \cdot \text{Diagonal} + d \cdot \text{Height} + e \cdot \text{Width} + f$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| z | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| σ(z) | 0.0180 | 0.1192 | 0.5 | 0.8808 | 0.9820 |

- 이름에 "회귀"가 붙은 이유는 선형 회귀와 같은 선형식으로 점수를 계산하기 때문이다. 더 정확히는 **로그 오즈를 선형식으로 회귀**하는 모델이다.
- 입력에서 z까지는 독립된 예측 모델이 아니다. 계수는 시그모이드를 거친 확률이 정답과 맞도록 함께 학습된다.
- 시그모이드는 점수를 확률로 **변환**하고, 판정은 0.5 기준이 한다. σ(z) > 0.5와 z > 0은 같은 조건이다.
- scikit-learn은 확률이 정확히 0.5이면 음성 클래스로 처리한다.

### z의 의미: 로그 오즈

$$z = \log\frac{p}{1-p}$$

- `p / (1−p)`는 오즈다. 시그모이드는 이 식을 p에 대해 푼 역함수다.
- 출력을 확률로 읽는 근거는 0~1 범위라서가 아니다. 모델 가정(로그 오즈가 선형)과 로지스틱 손실(최대 가능도)로 학습한다는 점이 근거다.

### 도미·빙어 이진 분류 실습

```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt  = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.classes_)                           # ['Bream' 'Smelt'] → Smelt가 양성 클래스
print(lr.predict(train_bream_smelt[:5]))     # ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
print(lr.coef_, lr.intercept_)
# [[-0.404 -0.576 -0.663 -1.013 -0.732]] [-2.161]

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)                             # [-6.03  3.57 -5.27 -4.24 -6.06]

from scipy.special import expit
print(expit(decisions))                      # predict_proba()[:, 1]과 정확히 일치
```

- 양성 클래스는 `classes_[1]`이다. 이진 분류에서는 한 클래스의 확률만 알면 나머지는 `1 − p`다.
- 직접 계산한 `X @ coef_[0] + intercept_[0]`도 `decision_function()` 결과와 같다.

### 계수의 의미

| 특성 | 계수 | 표준화된 값이 1 증가하면 |
|---|---|---|
| Weight | −0.404 | z가 0.404 감소 |
| Height | −1.013 | z가 1.013 감소 (영향이 가장 큼) |

- 계수와 절편처럼 훈련 데이터에서 학습하는 값이 **모델 파라미터**다. `C`, `max_iter`, `n_neighbors`처럼 사람이 정하는 값은 하이퍼파라미터다.
- 부호는 방향, 크기는 영향력이다. 계수를 서로 비교할 수 있는 것은 표준화로 단위를 맞췄기 때문이다.
- 계수는 상관계수가 아니다. 손실을 줄이도록 학습된 값이다.
- 계수는 **로그 오즈에는 덧셈으로, 오즈에는 곱셈으로(e^계수 배), 확률에는 비선형으로** 작용한다.

| 출발 z | 출발 확률 | 무게 +1 후 확률 | 확률 변화 |
|---|---|---|---|
| 0 | 0.5000 | 0.4002 | −10.0%p |
| 0 (무게 +2) | 0.5000 | 0.3081 | −19.2%p (정확히 2배 아님) |
| −6 | 0.0025 | 0.0017 | −0.08%p |

- 그래서 입력값이 달라지면 확률은 변해도 최종 분류는 같을 수 있다.

---

## 5. 로지스틱 회귀 (다중 분류)

```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))   # 약 0.933
print(lr.score(test_scaled, test_target))     # 약 0.925
print(lr.predict(test_scaled[:5]))            # ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
print(lr.coef_.shape, lr.intercept_.shape)    # (7, 5) (7,)

decision = lr.decision_function(test_scaled[:5])   # 5행 7열
from scipy.special import softmax
proba = softmax(decision, axis=1)                  # predict_proba와 같음
```

### 규제 파라미터

| | Ridge | LogisticRegression |
|---|---|---|
| 파라미터 | `alpha` | `C` |
| 규제를 강하게 하려면 | 크게 | **작게** |
| 기본값 | 1.0 | 1.0 |

- `C=20`은 기본값보다 규제를 약하게 한 설정이다.
- `max_iter`는 최대 반복 횟수다. 최적화가 먼저 끝나면 중간에 멈춘다.
- 테스트 점수 하나로 "로지스틱 회귀가 더 낫다"고 결론 내리면 안 된다. 하나의 데이터·분할 결과일 뿐이고, 테스트 세트로 모델을 고르면 테스트 세트에 맞춘 셈이 된다(5장 검증 세트·교차 검증).

### 소프트맥스

$$\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

- 절차: 각 z에 exp를 적용하고, 모두 더한 뒤, 각 값을 합으로 나눈다.
- 예: exp 결과가 Perch 4, 나머지 6종이 각각 1이면 합은 10이다. Perch는 0.4, 나머지는 각각 0.1이다.
- exp는 항상 양수를 만들고 점수 차이를 벌리며 순서를 보존한다.
- `axis=1`을 빠뜨리면 샘플별이 아니라 전체 값에 대해 계산된다.

| 샘플 | Bream | Parkki | Perch | Pike | Roach | Smelt | Whitefish | 예측 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.014 | **0.841** | 0.000 | 0.136 | 0.007 | 0.003 | Perch |
| 4 | 0.011 | 0.034 | 0.306 | 0.007 | **0.567** | 0.000 | 0.076 | Roach |

### 이진 분류와 다중 분류 비교

| | 이진 분류 | 다중 분류 |
|---|---|---|
| 샘플 하나의 z 개수 | 1개 | 클래스 수(7개) |
| 계수 shape | (1, 5) | (7, 5) |
| 확률 변환 함수 | 시그모이드 | 소프트맥스 |
| 변환 결과 | 양성 클래스의 확률 | 각 클래스의 확률 |
| 확률의 관계 | 음성 = 1 − 양성 | 합 = 1 |
| 최종 선택 | 양성 확률 > 0.5 | 확률이 가장 큰 클래스(argmax) |
| 최고 확률이 0.5 미만이어도 선택? | 해당 없음 | 예 |

- 0.5 기준은 클래스가 2개일 때의 argmax와 같다.
- 클래스가 2개이면 소프트맥스는 시그모이드가 된다: `softmax = σ(z₁ − z₀)`.

### 증명: 소프트맥스가 확률분포를 반환하는 이유

**(1) 형식.** $e^{z} > 0$이므로 $p_k > 0$이다. 합은 $\sum_k e^{z_k} / S = S/S = 1$이다.

**(2) 유도.** 각 클래스의 로그 오즈가 선형이라고 가정한다: $\log(p_k / p_K) = z_k$, $z_K = 0$.
그러면 $p_k = p_K e^{z_k}$이고, 합이 1이라는 조건에서 $p_K = 1 / \sum_j e^{z_j}$이다.
따라서 $p_k = e^{z_k} / \sum_j e^{z_j}$로 소프트맥스가 된다.

- 모든 점수에 같은 상수를 더해도 결과가 같다. 그래서 scikit-learn은 기준 클래스 없이 7개 클래스 모두에 계수를 두고, 규제가 해를 하나로 정한다.
- 출력이 확률분포라는 것과 참 확률에 맞게 보정(calibration)되었다는 것은 다른 문제다.

---

## 6. 계단 함수와 시그모이드

$$
f(x) =
\begin{cases}
1 & (x \geq 0) \\
0 & (x < 0)
\end{cases}
\qquad
H(x) = \lim_{k \to \infty} \sigma(kx) \quad (x \neq 0)
$$

- 시그모이드는 0과 1에 다가갈 뿐 도달하지 않는다. 그래서 계단 함수와 정확히 같아질 수는 없고, 극한으로만 같아진다. x = 0에서는 항상 0.5다.
- 최종 분류는 `H(z)`와 같다. 그래도 시그모이드를 쓰는 이유는 미분할 수 있어야 경사 하강법으로 학습할 수 있기 때문이다(퍼셉트론에서 신경망으로 넘어간 이유).

---

## 7. 확률적 경사 하강법 (4-2 도입)

### 학습할 때 바뀌는 것

| 바뀌지 않음 | 바뀜 |
|---|---|
| 입력 특성, 정답, 하이퍼파라미터 | 계수, 절편 |

- 얼마나 틀렸나 → **손실 함수**
- 어느 방향으로 고치나 → **경사**(손실을 파라미터에 대해 미분한 값)

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

- 경사의 반대 방향으로 학습률 η만큼 이동한다. η가 너무 크면 최저점을 지나쳐 반대편으로 튄다.
- 목표는 손실이 **최소인 지점**(기울기 0)이다. 손실이 0인 지점이 아니다.
- 미분하는 대상은 데이터가 아니라 손실 함수다. 정확도는 미분할 수 없어서 로지스틱 손실을 쓴다.

### 업데이트 한 번의 예시

로지스틱 손실의 경사는 $\partial L/\partial w = (p - y)x$, $\partial L/\partial b = (p - y)$다.

| 단계 | 계산 (w=0.5, b=0, η=0.1, 빙어 y=1, x=−2) | 결과 |
|---|---|---|
| 점수·확률 | z = −1.0, σ(−1.0) | p = 0.269 |
| 오차 | p − y | −0.731 |
| 경사 | (−0.731) × (−2) | +1.462 |
| 수정 | w ← 0.5 − 0.1 × 1.462 / b ← 0 + 0.0731 | w = 0.354, b = 0.073 |
| 재계산 | z = −0.635 | p = 0.346 (정답 쪽으로 이동) |

### SGD, 미니배치, 배치

| 방법 | 한 번 수정에 쓰는 샘플 | 12개 샘플일 때 1 에포크당 업데이트 |
|---|---|---|
| SGD | 1개 | 12번 |
| 미니배치 | 여러 개 (예: 4개) | 3번 |
| 배치 | 전체 | 1번 |

- **에포크**는 훈련 세트 전체를 한 번 사용한 단위다. 수정 횟수가 아니다.
- 다음 에포크는 이전 계수에서 이어서 수정하고, 보통 샘플 순서를 다시 섞는다.

### 점진적 학습

| 방식 | 방법 | 단점 |
|---|---|---|
| 전체 재학습 | 기존 + 새 데이터로 처음부터 학습 | 데이터가 쌓일수록 느림 |
| 점진적 학습 | 기존 모델을 유지하고 새 데이터만 추가 학습 (`partial_fit`) | 예전 학습 내용이 희미해짐(파국적 망각), 분포 변화에 흔들림, 검증이 어려움 |

- 실무에서는 주기적 전체 재학습을 택하는 경우가 많다.

### 코드 주의 사항

```python
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
```

- 교재 구판의 `loss='log'`는 scikit-learn 1.1에서 `'log_loss'`로 이름이 바뀌었고 1.3에서 삭제되었다.
- `SGDClassifier`의 `max_iter`는 **최대 에포크 수**다. `LogisticRegression`의 `max_iter`는 solver의 반복 횟수다.

---

## 8. 노트에서 바로잡은 내용

| 처음 이해한 내용 | 바로잡은 내용 |
|---|---|
| 훈련·테스트 세트를 각각 다른 StandardScaler로 표준화 | 하나의 스케일러를 훈련 세트에만 `fit`하고 둘 다 `transform` |
| 표준화된 값은 −1~1 사이 | 평균 0, 표준편차 1일 뿐 범위 제한은 없음 |
| `random_state=42`는 데이터를 고루 섞음 | 재현성을 위한 시드. 고른 분할은 `stratify` |
| 시그모이드가 O/X를 판정 | 시그모이드는 확률로 변환, 판정은 0.5 기준 |
| z가 음수이면 빙어 확률 50% | z = 0일 때 50%, 음수이면 50% 미만이라 도미 |
| σ(−4) = 0.001, σ(−2) = 0.011 | σ(−4) = 0.0180, σ(−2) = 0.1192 |
| 무게 계수 약 −0.04045 | 약 −0.4045 |
| "회귀"라는 이름은 절편을 더하기 때문 | 선형식으로 로그 오즈를 계산하기 때문 |
| 계수는 데이터의 상관관계 | 손실을 줄이도록 학습된 파라미터 |
| 경사 하강법은 손실이 0인 지점을 찾음 | 손실이 최소(기울기 0)인 지점을 찾음 |
| 데이터를 미분 | 손실 함수를 파라미터에 대해 미분 |
| 점진적 학습에서 은닉 변수가 퇴색 | 예전 학습 내용이 희미해짐(파국적 망각) |
