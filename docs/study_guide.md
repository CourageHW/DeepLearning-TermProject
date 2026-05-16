# 1-bit BNN 공부 노트 — 기본부터 차근차근

본 프로젝트(`A1W1ResNet18v2`, CIFAR-100, Knowledge Distillation, C inference) 의 모든 개념을 **신경망을 처음 배우는 사람** 이 따라올 수 있도록 정리한다. 각 챕터는:

- **직관**: 비유 + 작은 예제
- **수식**: 정식 정의 + 손계산 예시
- **코드 연결**: 우리 프로젝트의 어디에 나오는지 (`train.ipynb` 셀 번호 + 함수명)

순서대로 읽으면 좋고, 모르는 단어가 나오면 부록 §15 의 용어집을 참조하라.

---

## 목차

0. [읽기 전 안내](#0-읽기-전-안내)
1. [신경망 이전 — 컴퓨터의 숫자](#1-신경망-이전--컴퓨터의-숫자)
2. [신경망의 가장 기본 — 곱하고 더하고](#2-신경망의-가장-기본--곱하고-더하고)
3. [Convolution — 사진을 보는 방식](#3-convolution--사진을-보는-방식)
4. [ResNet18 한 줄로 이해](#4-resnet18-한-줄로-이해)
5. [왜 1-bit 인가 — 두 축의 압축](#5-왜-1-bit-인가--두-축의-압축)
6. [sign 함수와 ±1 표현](#6-sign-함수와-1-표현)
7. [학습이 안 되는 문제 — STE](#7-학습이-안-되는-문제--ste)
8. [ReActNet 의 다섯 컴포넌트](#8-reactnet-의-다섯-컴포넌트)
9. [Knowledge Distillation — 선생과 학생](#9-knowledge-distillation--선생과-학생)
10. [XNOR + POPCNT 마법](#10-xnor--popcnt-마법)
11. [Padding correction — 작은 함정](#11-padding-correction--작은-함정)
12. [전체 학습 파이프라인](#12-전체-학습-파이프라인)
13. [평가 지표](#13-평가-지표)
14. [Loss · Optimizer · LR Schedule](#14-loss--optimizer--lr-schedule)
15. [부록 — 용어집 + FAQ](#15-부록--용어집--faq)

---

## 0. 읽기 전 안내

### 이 노트를 위해 필요한 사전 지식

- **거의 없다**. 곱하기, 더하기, 행렬이 뭔지 정도. 미분은 "기울기 정도" 의 직관만 있으면 충분 (수식이 나오면 차근차근 설명).
- 코딩 경험 0 명도 따라올 수 있게 코드는 "어디 가면 보이는지" 만 알려주고, 핵심 로직은 손계산으로 풀어둔다.

### 약속

- **굵게** 는 새로 도입되는 핵심 용어.
- `monospace` 는 코드 / 변수 / 파일 이름.
- > 인용 박스는 "헷갈리기 쉬운 부분" 의 보충 설명.

---

## 1. 신경망 이전 — 컴퓨터의 숫자

신경망 이야기를 하기 전에, 컴퓨터가 **숫자를 어떻게 저장하는가** 부터 짚는다. 1-bit BNN 의 의미를 이해하려면 이 부분이 진짜 필수다.

### 1.1 비트 (bit), 바이트 (byte)

- **1 bit** = `0` 또는 `1` 둘 중 하나만 저장 가능 = 정보 단위의 최소.
- **1 byte** = 8 bits = `00000000` ~ `11111111` 까지 2⁸ = **256** 가지 패턴.

작은 예:
- `01000001` 이라는 1-byte 가 ASCII 표현이면 → 문자 `A`.
- `00000001` 을 정수로 읽으면 → `1`.
- `11111111` 을 부호 없는 정수로 읽으면 → `255`.

같은 비트 패턴도 **"어떻게 해석하는가" 의 약속** 에 따라 의미가 달라진다.

### 1.2 FP32 (single-precision float) — 컴퓨터 신경망의 기본 숫자

- **FP32** = 32 bits = 4 bytes 짜리 실수 (소수점이 있는 수).
- 표현 가능한 범위: 대략 ±10⁻³⁸ ~ ±10³⁸, 소수점 정확도 약 7자리.
- 예: `3.141592` → 메모리 안에서는 `01000000_01001001_00001111_11011011` 같은 32 비트.

신경망의 weight (가중치) 1 개를 FP32 로 저장하면 → **32 bits** 가 필요.

### 1.3 메모리 = weight 수 × bits-per-weight

ResNet18 의 weight 수: 약 **11.22 M** (M = million, 백만).

| 정밀도 | bits / weight | 총 메모리 |
|---|---|---|
| FP32 | 32 | 11.22M × 32 / 8 / 10⁶ = **44 MB** |
| FP16 | 16 | 22 MB |
| INT8 | 8 | 11.2 MB |
| INT4 | 4 | 5.6 MB |
| **1-bit (BNN)** | **1** | **1.4 MB** |

> 본 프로젝트의 핵심 성과 "44 MB → 1.4 MB · 32× 압축" 이 바로 이 표의 첫 줄 ↔ 마지막 줄 차이다.

### 1.4 왜 32× 가 아니라 31.4× 인가?

- 실제로는 student 모델에도 **FP32 로 남겨두는 부분** 이 있다:
  - 첫 conv (stem): 입력 이미지 ↔ 첫 feature map. 3×64×3×3 = 1 728 weights × 32 bit.
  - 마지막 fc (classifier): 512 → 100. 51 200 weights × 32 bit.
  - BN/RPReLU/RSign 의 작은 학습 파라미터.
- → 이 작은 FP32 부분이 약 200 KB 정도 차지해서 1.4 MB 가 되고, 비율이 정확히 32× 가 아니라 **31.4×** 로 측정된다 (보고서 §6.4).

---

## 2. 신경망의 가장 기본 — 곱하고 더하고

### 2.1 한 줄 정의

신경망 = "입력 벡터에 weight 행렬을 곱하고, 비선형 함수(activation) 통과시키는 것을 여러 번 반복하는 함수".

### 2.2 가장 작은 예제 — 2-입력 1-출력

입력 `x = [x_1, x_2]`, weight `w = [w_1, w_2]`, bias `b` 가 있다. 신경망의 한 "**뉴런 (neuron)**" 은:

```
y = w_1 · x_1 + w_2 · x_2 + b
```

손계산 예시: `x = [2, 3]`, `w = [0.5, -1]`, `b = 0.1`
```
y = 0.5 × 2 + (-1) × 3 + 0.1 = 1 - 3 + 0.1 = -1.9
```

### 2.3 비선형 함수 (activation function)

만약 y 가 끝이면 신경망은 그냥 선형 함수일 뿐 (선형 변환만 쌓아봐야 결국 또 다른 선형). 진짜 강력해지려면 **비선형 함수** 를 통과시킨다.

가장 흔한 것:
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)` — 음수면 0, 양수면 그대로.
- **Sigmoid**: `f(x) = 1 / (1 + e^{-x})` — 0 ~ 1 사이로 압축.

예: 위 `y = -1.9` 에 ReLU 적용 → `max(0, -1.9) = 0`.

### 2.4 여러 뉴런 = 행렬 곱

뉴런이 여러 개면 weight 가 **행렬** 이 된다:

```
y = W · x + b
```

`x` 는 길이 2 벡터, `W` 는 3×2 행렬이면 → `y` 는 길이 3 벡터.

```
W = [[ 0.5,  -1.0],          x = [2,
     [-0.2,   0.3],                3],
     [ 1.1,   0.4]]

y_1 =  0.5·2 + (-1.0)·3 = -2.0
y_2 = -0.2·2 +   0.3 ·3 =  0.5
y_3 =  1.1·2 +   0.4 ·3 =  3.4
```

### 2.5 **MAC** — Multiply-Accumulate

`w_i · x_i + (이전 합)` 같이 한 번의 "곱한 뒤 누적" 을 **1 MAC** (multiply-accumulate) 이라 한다.

위 예시에서 `y_1` 계산 = 2 번의 곱셈 + 1 번의 덧셈 = **2 MAC** (관례상). 보고서의 "Teacher 555.5 M MACs" 라는 표현은 **5억 5천 5백만 번의 곱셈+덧셈** 이 한 장 이미지 추론에 필요하다는 뜻.

> **FLOPs vs MACs**: 1 MAC = 1 곱셈 + 1 덧셈 = 2 FLOPs (floating point operations) 로 세는 관례도 있다. 본 보고서는 MACs 단위로 통일.

---

## 3. Convolution — 사진을 보는 방식

이미지 인식 신경망의 핵심 연산. 위 §2 의 "행렬 곱" 의 특수한 형태다.

### 3.1 왜 그냥 행렬 곱이 아닌가

32×32 RGB 이미지 = 3 × 32 × 32 = **3072 픽셀**. 이걸 한 줄로 펴서 fully-connected layer 에 넣으면, 다음 layer 가 1024 뉴런이라면:
- weight 수 = 3072 × 1024 = **약 314 만 개**.

이미지는 너무 큰데, 이미지의 **국소 패턴 (가까운 픽셀끼리 관련)** 은 한정적이다. → "작은 윈도우만 보고, 같은 weight 를 이미지 전체에서 공유" 하자는 아이디어.

### 3.2 직관 — 작은 도장 찍기

이미지 한 장 위에 **3×3 짜리 작은 도장 (kernel / filter)** 을 올린다. 그 도장 안의 9개 weight 와 이미지의 해당 위치 9개 픽셀을 곱해서 합 하나를 만든다. 도장을 한 칸씩 옮겨가며 반복.

```
입력 (5×5):              필터 (3×3):           출력 (3×3):
1 2 3 0 1                  1 0 1
0 1 2 1 0                  0 1 0       →       (계산은 아래)
1 0 1 2 1                  1 0 1
0 2 1 0 1
1 1 0 1 2
```

좌측 상단 3×3 영역 = `[[1,2,3],[0,1,2],[1,0,1]]`, 필터와 element-wise 곱한 뒤 모두 합:
```
1·1 + 2·0 + 3·1 + 0·0 + 1·1 + 2·0 + 1·1 + 0·0 + 1·1
= 1 + 0 + 3 + 0 + 1 + 0 + 1 + 0 + 1 = 7
```
→ 출력의 좌측 상단 = **7**.

도장을 오른쪽으로 한 칸 옮겨서 다시 계산 → 출력의 다음 위치. 이렇게 5×5 입력 + 3×3 필터 (stride 1, no padding) 로 (5-3+1) × (5-3+1) = **3×3 출력** 이 나온다.

### 3.3 정식 수식

```
y[c_out, i, j] = b[c_out] + Σ_{c_in} Σ_{k_h} Σ_{k_w}
                    w[c_out, c_in, k_h, k_w] · x[c_in, i + k_h, j + k_w]
```

- `c_in`: 입력 채널 (RGB 면 3).
- `c_out`: 출력 채널 (예: 64).
- `k_h, k_w`: kernel 내 위치 (0, 1, 2 for 3×3).
- `i, j`: 출력 spatial 위치.

### 3.4 핵심 매개변수

- **kernel size** (k): 도장 크기. 흔히 3×3.
- **stride** (s): 한 번에 옮기는 칸 수. stride 2 면 출력이 절반 크기.
- **padding** (p): 입력 가장자리에 0 으로 둘러주는 띠 두께. 출력 크기 보존 목적.
- **channels** (c_in, c_out): 깊이 (RGB 의 R, G, B 같은 것). 신경망 안에서는 점점 늘어나 64 → 128 → 256 → 512.

### 3.5 출력 크기 공식

```
H_out = (H_in + 2·p - k) / s + 1
```

예: 입력 32, k=3, s=1, p=1 → (32 + 2 - 3)/1 + 1 = **32**. (즉 같은 크기 유지)
예: 입력 32, k=3, s=2, p=1 → (32 + 2 - 3)/2 + 1 = **16**. (절반)

### 3.6 MAC 수 계산

```
MACs = c_in × c_out × k_h × k_w × H_out × W_out
```

예: 입력 32×32, c_in=64, c_out=128, k=3, stride=2 (출력 16×16):
```
MACs = 64 × 128 × 3 × 3 × 16 × 16 = 18 874 368 ≈ 18.9 M
```

→ 본 프로젝트의 `compute_bitops.py` 가 이 공식을 모든 layer 에 적용해서 teacher 555.5 M MACs / student 547.4 M binary MACs 를 산출.

---

## 4. ResNet18 한 줄로 이해

본 프로젝트의 backbone (뼈대).

### 4.1 깊이 = 학습 능력의 증가 + 학습 난이도의 증가

신경망 layer 를 많이 쌓으면 표현력이 강해지지만, 동시에 학습이 어려워진다 (gradient vanishing 등).

### 4.2 Residual connection — He et al. 2015 의 핵심 아이디어

```
y = F(x) + x          ← residual / skip connection
```

`F(x)` 는 "이 layer 가 계산하는 변화량". 만약 best 가 "x 그대로" 면, `F(x) = 0` 으로 학습되면 됨. → 학습이 쉬워진다.

도식:
```
        x ──────┬─────────────────┐
                │                  │
                ▼                  ▼
            conv1 - BN - ReLU
                │                  │
                ▼                  │
            conv2 - BN             │
                │                  │
                ▼                  │
                + ◀────────────────┘   ← residual add
                │
                ▼
              ReLU
                │
                ▼
               y
```

이게 **Basic Block** 의 기본 구조. ResNet18 은 이런 block 을 8 개 쌓는다.

### 4.3 ResNet18 의 전체 구조

| Stage | Block 수 | Channels | Spatial (CIFAR-32 기준) |
|---|---|---|---|
| Stem (첫 conv) | 1 | 64 | 32 |
| Layer1 | 2 blocks (×2 conv = 4 conv) | 64 | 32 |
| Layer2 | 2 blocks (×2 conv = 4 conv) | 128 | 16 (downsample) |
| Layer3 | 2 blocks (×2 conv = 4 conv) | 256 | 8  (downsample) |
| Layer4 | 2 blocks (×2 conv = 4 conv) | 512 | 4  (downsample) |
| AvgPool + FC | — | 100 | 1 |

총 conv 수 = 1(stem) + 16(blocks) + 3(downsample 1×1 conv, FP32 teacher 만) = 17 or 20.

> **CIFAR vs ImageNet**: 표준 ResNet18 은 ImageNet (224×224) 용으로 stem 이 7×7 stride 2 + maxpool 이다. CIFAR (32×32) 용으로는 stem 을 3×3 stride 1 로 바꾸고 maxpool 을 빼는 게 관례 (보고서 §3.2).

---

## 5. 왜 1-bit 인가 — 두 축의 압축

### 5.1 메모리 축 (앞에서 이미 봤음)

11.22 M × 32 bit → 11.22 M × 1 bit = **32× 압축**.

### 5.2 연산 축 (이게 더 신기하다)

연산도 줄어든다. **FP32 곱셈 → 1-bit XNOR 로 대체**.

- FP32 한 번 곱셈: 약 0.5 nanosecond (modern CPU).
- 1-bit XNOR: 약 0.05 ns. 게다가 64 개를 한 번에 (§10).

→ **이론적으로 1 FP32 MAC ≈ 64 BitOPs** 의 가치.

본 프로젝트의 BitOPs 계산 결과 (보고서 §6.4):
- Teacher 555.5 M MACs vs Student 10.4 M FP32-equivalent FLOPs → **53.8× 이론 연산 압축**.

### 5.3 정리

```
            FP32 model     1-bit BNN     비율
메모리       44 MB         1.4 MB        32×
연산        555.5 M MACs   10.4 M FLOPs  53.8×
```

이 두 축이 **임베디드 / 모바일 / 엣지 디바이스** 에서 BNN 이 매력적인 이유.

---

## 6. sign 함수와 ±1 표현

### 6.1 정의

```
sign(x) = +1   if x ≥ 0
sign(x) = -1   if x < 0
```

(어떤 정의는 sign(0) = 0 이지만 BNN 에서는 ≥ 0 → +1 관례.)

### 6.2 그래프

```
        ↑ sign(x)
       +1 ┼━━━━━━━
          │
   ───────┼───────→ x
          │
          ┃━━━━━ -1
```

**계단 함수** — 0 에서 갑자기 -1 → +1 로 점프.

### 6.3 ±1 = 1 bit

`{-1, +1}` 두 값만 가능 → 1 비트로 표현 충분. 관례: `bit = 1` → `+1`, `bit = 0` → `-1`.

### 6.4 적용 — Binary weight, Binary activation

- **Binary weight**: 학습된 FP32 weight `W` 에 `sign()` 적용 → 추론 시 ±1 만 저장.
- **Binary activation**: feature map 의 값에 `sign()` 적용 → 다음 conv 의 입력이 ±1 만 됨.

본 프로젝트는 weight, activation 둘 다 1-bit → **W1A1** (Weight 1-bit, Activation 1-bit), 약자로 **A1W1**.

---

## 7. 학습이 안 되는 문제 — STE

### 7.1 문제

`sign(x)` 는 모든 곳에서 도함수 (미분값) 가 **0** 이다 (계단이라 평평).

```
            sign(x)              d/dx sign(x)
            
          +1 ┼━━━━                    │
             │                        │
   ──────────┼──→            ─────────┼─────────→
             │                        │
             ┃━━━━━ -1                 0 모든 곳에서
```

신경망 학습은 **gradient descent (경사 하강)** — 도함수가 0 이면 weight 가 한 발짝도 못 움직임 → 학습 불가.

### 7.2 해결: Straight-Through Estimator (STE)

**Forward 와 Backward 를 비대칭으로 정의** 하는 트릭.

- **Forward (예측 시)**: `sign(x)` 그대로 사용 (binary output).
- **Backward (학습 시)**: 도함수를 마치 **identity (그냥 통과)** 인 것처럼 취급.

```
forward:   y = sign(x)
backward:  ∂L/∂x = ∂L/∂y  · 1    ← sign 의 도함수 0 을 무시하고 1 로 가정
```

### 7.3 직관

신경망 안에는 **latent weight** (잠재 weight, 학습 중에는 FP32 로 유지되는 진짜 학습 변수) 가 있다. Forward 시에만 `sign()` 통과시켜 ±1 로 binarize. Backward 시에는 gradient 가 latent weight 로 그대로 흘러들어가 FP32 학습 가능.

```
[latent W: FP32]  ──sign()─→  [binary W: ±1]  ──conv─→  output

backward:    gradient ←──────  (identity, sign 무시)  ←─────  gradient
```

학습 끝나면 latent W 를 버리고 `sign(W)` 만 저장 → 1-bit checkpoint.

### 7.4 코드 (`train.ipynb` cell 6)

```python
class BinaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    @staticmethod
    def backward(ctx, g):
        return g     # ← identity, gradient 그대로 통과
```

8 줄. 이게 1-bit BNN 학습의 가장 핵심 트릭.

### 7.5 Activation 도 마찬가지

```python
class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, ones, -ones)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)   # ← clipped STE
```

Activation 쪽은 **clipped STE**: gradient 를 `|x| ≤ 1` 인 곳에서만 흘리고, 밖에서는 막는다. (학습 안정성 향상, Bi-Real Net 이 제안.)

### 7.6 Latent weight clamp

매 step 후 latent W 를 `[-1, +1]` 로 clamp (`w = w.clamp(-1, 1)`). 이유: latent W 가 너무 커지면 sign 출력 영역 밖에서 gradient 가 무의미해짐.

---

## 8. ReActNet 의 다섯 컴포넌트

기본 1-bit BNN (XNOR-Net 같은 것) 만으로는 정확도가 50% 대에 머물렀다. **ReActNet (Liu et al. 2020)** 이 다섯 가지 트릭으로 W1A1 ImageNet 69.4% 까지 끌어올림. 우리 프로젝트도 이 다섯 가지 모두 사용.

### 8.1 RSign (Reactive Sign) — 학습 가능한 threshold

- 일반 sign: `sign(x)` → 0 에서 분기.
- **RSign**: `sign(x - α)` → α 도 학습. **per-channel** (채널마다 다른 α).

```python
class RSignActivation(nn.Module):
    def __init__(self, channels):
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        return BinaryActivationSTE.apply(x - self.threshold)
```

→ 채널마다 "어느 값부터 +1 로 칠 것인지" 를 학습. 데이터 분포에 맞춰 sign 의 결정 경계가 조정됨.

### 8.2 RPReLU (Reactive PReLU) — 학습 가능한 activation

기본 PReLU: `f(x) = x` if x ≥ 0, `α · x` if x < 0 (α 는 음수 기울기, 학습 파라미터).

**RPReLU** 는 PReLU 에 channel-wise 이동 (γ, β) 까지 더한 형태:

```
RPReLU(x) = PReLU(x - γ) + β
```

```python
class RPReLU(nn.Module):
    def __init__(self, channels):
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.prelu = nn.PReLU(num_parameters=channels)
    def forward(self, x):
        return self.prelu(x - self.gamma) + self.beta
```

학습 파라미터: γ (shift before activation), β (shift after activation), α (PReLU slope) — 모두 채널마다 따로.

> **본 프로젝트 ablation 핵심 결과**: RPReLU 를 제거 (`nn.Identity()` 로 대체) 했을 때 정확도가 **−9.22 %p** 떨어졌다 (E3). KD, double-skip 가 각각 ~2.3 %p 인 것과 비교하면 RPReLU 가 **단독 최대 기여 컴포넌트**. 1-bit activation 의 정보 손실을 RPReLU 가 channel 별로 reshape 해서 복원한다는 ReActNet 의 핵심 주장이 본 데이터로 입증.

### 8.3 Double-skip — 두 개의 residual

기본 ResNet basic block 은 residual 이 1 개:
```
y = ReLU(F(x) + x)
```

**Double-skip** 은 한 block 안의 두 conv 각각에 residual 을 붙임:
```
out  = act1(conv1(rsign1(bn1(x))))  + x        ← residual_a
out2 = act2(conv2(rsign2(bn2(out))))+ out      ← residual_b
```

→ Gradient 가 흐를 수 있는 경로가 늘어남. 1-bit activation 으로 인한 정보 손실을 FP32 residual 이 우회 경로로 보상.

Ablation: double-skip 제거 시 **−2.46 %p** (E2).

### 8.4 Detached α scale

`A1W1BinaryConv2dV2` 의 핵심.

```python
def forward(self, x):
    w_b   = BinaryWeightSTE.apply(self.weight)      # ±1 binary weight
    alpha = self.weight.detach().abs().mean(...)    # per-output-channel scale, FP32
    out   = F.conv2d(x, w_b, ...)
    return out * alpha
```

- `alpha` = latent weight 의 `|w|` 의 mean → 채널별 "원래 weight 의 크기" 를 보존.
- **`.detach()`** = 이 alpha 는 학습 시 gradient 가 안 흐르도록 그래프에서 떼어냄.
- 이유: alpha 가 학습 가능하면 → latent W 가 sign 의 도함수 0 영역으로 끌려가 발산.

→ alpha 는 forward 시에만 "binary output 의 크기 보정" 역할.

### 8.5 STE forward / backward 비대칭 (이미 §7 에서 본 트릭)

종합하면, 위 다섯 가지 모두 **본 프로젝트의 student 가 1-bit 정확도 68.7%** 를 달성하는 핵심 디자인.

---

## 9. Knowledge Distillation — 선생과 학생

### 9.1 직관 — 큰 모델한테 hint 받기

- **Teacher**: FP32 ResNet18 (82.44% accuracy, 44 MB).
- **Student**: 1-bit BNN ResNet18 (68.72% accuracy, 1.4 MB).

Student 가 단독으로 학습하면 정확도가 낮음. Teacher 의 출력을 **soft target** 으로 흉내내도록 가르치면 더 잘 배운다.

### 9.2 Softmax — 확률 분포 만들기

신경망의 마지막 출력 (logits) 은 100 개 실수 (CIFAR-100 100 classes). 그대로는 확률이 아님. **Softmax** 로 변환:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

예: logits = `[2, 1, 0]`
```
exp(2) = 7.39,  exp(1) = 2.72,  exp(0) = 1
합 = 11.11
softmax = [0.665, 0.245, 0.090]      ← 합 1
```

→ "class 1 일 확률 66.5%, class 2 일 확률 24.5%, class 3 일 확률 9.0%".

### 9.3 Temperature — 분포의 부드러움 조절

```
softmax_T(z)_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

`T` 를 키우면 분포가 **더 부드러워짐 (uniform 에 가까워짐)**.

예: logits `[2, 1, 0]`, T = 4
```
z/T = [0.5, 0.25, 0]
exp = [1.65, 1.28, 1.0]
합 = 3.93
softmax_4 = [0.420, 0.326, 0.255]    ← 훨씬 평탄
```

→ Teacher 가 "class 1 일 확률은 높지만 class 2 도 꽤 비슷해" 라는 **풍부한 정보** 를 student 한테 전달.

### 9.4 KL Divergence — 두 분포의 "거리"

두 확률 분포 `p_t` (teacher) 와 `p_s` (student) 가 얼마나 다른지 측정:

```
KL(p_t || p_s) = Σ_i p_t(i) · log(p_t(i) / p_s(i))
              = Σ_i p_t(i) · (log p_t(i) - log p_s(i))
```

- 두 분포가 같으면 = 0.
- 다르면 양수 (KL 은 비음수).
- 비대칭: KL(p_t || p_s) ≠ KL(p_s || p_t).

> **방향이 중요**: Hinton KD (2015) 의 표준은 `KL(p_t || p_s)` — teacher 가 기준. PyTorch 의 `F.kl_div(input, target)` 은 `input = log p_s`, `target = p_t` 형태로 받으며 내부적으로 `KL(target || exp(input))` 을 계산한다. 본 프로젝트도 이 표준을 따른다 (보고서 §4 / decision D11).

### 9.5 최종 KD loss

```
L = (1 - w) · CE(p_s, y_true) + w · T² · KL(p_t || p_s)
```

- `CE` (cross-entropy): student 가 **실제 label** 을 맞추는 표준 loss.
- `KL`: student 가 **teacher 의 분포** 를 흉내내는 loss.
- `w`: 두 loss 의 가중치. 본 프로젝트 `w = 0.7` (보고서 §5.2).
- `T²`: temperature 보정 항 (gradient scale 보존).

> **Ablation**: KD 를 끄면 (`w = 0`) 정확도 **−2.30 %p** (E1).

---

## 10. XNOR + POPCNT 마법

이제 **추론** 시 1-bit conv 가 어떻게 빨라지는지.

### 10.1 곱셈을 XNOR 로 대체할 수 있나?

±1 끼리의 곱셈:
```
(+1) × (+1) =  +1
(+1) × (-1) =  -1
(-1) × (+1) =  -1
(-1) × (-1) =  +1
```

`+1` 을 bit `1`, `-1` 을 bit `0` 으로 인코딩:
```
1 ⊕ 1 = 0      (XOR: 다르면 1)        ¬(1 ⊕ 1) = 1     (XNOR: 같으면 1)
1 ⊕ 0 = 1                              ¬(1 ⊕ 0) = 0
0 ⊕ 1 = 1                              ¬(0 ⊕ 1) = 0
0 ⊕ 0 = 0                              ¬(0 ⊕ 0) = 1
```

비교:
```
±1 곱셈   ↔   XNOR
 +1               1
 -1               0
 -1               0
 +1               1
```

→ **곱셈 = XNOR (same bit)**. 곱셈 한 번 → 비트 연산 한 번으로 대체. CPU 가 곱셈보다 비트 연산을 훨씬 빨리 한다.

### 10.2 합을 POPCNT 로

K 개의 ±1 을 곱한 뒤 더하기:
```
Σ_{k=1}^{K}  a_k × b_k   where a_k, b_k ∈ {-1, +1}
```

XNOR 로 보면 결과 bit 가 `1` 이면 `+1`, `0` 이면 `-1`. 합은:
```
합 = (#1 의 개수) - (#0 의 개수)
    = (#1) - (K - #1)
    = 2 · (#1) - K
```

`#1` = bit 1 의 개수 = **POPCNT** (population count). CPU 의 1-clock 명령어.

### 10.3 최종 공식

K bits 짜리 binary dot product = **XNOR + POPCNT + 선형 변환**:

```
y = 2 · POPCNT(¬(A ⊕ W)) - K
```

`A`, `W` 는 K bits 짜리 packed bit vector.

곱셈 K 번 + 덧셈 K 번 (= 2K FLOPs) → XNOR 1 회 + POPCNT 1 회 + 조정 1 회 (= 3 비트 연산). **K = 64 일 때 약 64× 빠름**.

### 10.4 Bit packing — 64 bits 한 번에

CPU register 는 64-bit. 64 개의 ±1 을 `uint64_t` 한 변수에 packing 하면:
```
uint64_t a = 0b1011_0110_0010_1101_..._1010_1101;
```
→ 64 개의 ±1 이 변수 하나.

XNOR 1 명령 → 64 개 비트 동시에 XNOR. POPCNT 1 명령 → 64 비트 안의 `1` 개수 세기.

본 프로젝트의 C 커널 (`xnor_kernel.c`) 이 이 packing + XNOR + POPCNT 를 모든 conv layer 에 적용 → student 1-thread 22 ms/sample (teacher 547 ms/sample) → **25× speedup**.

### 10.5 한계 — 실측 25× < 이론 53.8×

이론은 BitOPs/MAC × 1 = 53.8×. 실제 25× = 이론의 약 **46%**.

차이의 원인:
- C 커널이 BLAS (highly optimized matrix mul library) 대신 직접 작성한 단순 nested loop.
- AVX-512 의 SIMD POPCNT (`VPOPCNTDQ`) 미사용 — scalar POPCNT 만 사용.
- BN, RSign, RPReLU 의 FP32 fallback overhead.
- im2col (conv → matmul 변환) overhead.

→ 보고서 §7.4 + §8 Limitations 에 정직하게 명시.

---

## 11. Padding correction — 작은 함정

### 11.1 문제 — bit-packed 표현에 "0" 이 없다

일반 conv 에서 padding 은 입력 가장자리에 `0` 을 둘러주는 띠. **하지만 binary activation 은 {-1, +1} 두 값뿐** — `0` 이 없다. bit-packed 표현에서:
- bit `1` → `+1`
- bit `0` → `-1`

"padding 위치는 0" 을 어떻게 표현하나?

### 11.2 해결 — `-1` 로 packing 한 뒤 보정항 **더하기**

본 프로젝트의 4-step 트릭 (보고서 §7.1, 실제 구현은 `xnor_kernel.c::binary_conv2d` + `compute_pad_correction_one`):

**Step 1 — Encoding**: padded 위치를 `-1` (bit 0, sign encoding 상 `-1`) 로 packing. (코드 주석: "padded -> leave bit 0 (= -1 in sign encoding)")

**Step 2 — Naive POPCNT**: 위 §10.3 의 공식 그대로 적용. padded 위치는 실제 `0` 대신 `(-1) · w` 만큼 결과에 기여하므로, naive 결과는 정확한 값보다 `α · Σ_{pad} w` 만큼 **부족하다**.

**Step 3 — Correction**: 각 output 픽셀의 padding pattern (kernel × boundary 조합) 별로 사전 계산한 값을 결과에 **더한다**:
```
correction = α_co · ( 2 · popcount(w_co & pad_mask) - n_pad )
           = α_co · Σ_{pad positions} w_co
```

> **계수 2 의 의미**: `2·popcount - n_pad` 는 `(#w=+1 위치) - (#w=-1 위치)` 와 같다 (POPCNT 의 표준 ±1 변환). 이는 정확히 `Σ w_co` (padded 위치들에 대해) 와 동치.

**Step 4 — 결과**: padded 위치의 기여가 정확히 0 으로 보정되어 PyTorch `F.conv2d(..., padding=1)` 과 **bit-exact 일치** (max diff = 0).

### 11.3 왜 정확히 일치하나

PyTorch 의 zero-padding 은 padded 위치의 input × weight = 0 (실제 0 이라). 본 트릭은 padded 를 `-1` 로 인코딩해 `-w` 를 contribute 한 뒤, correction 으로 `+w` 를 더해 net 0 을 만든다. `-w + w = 0`, 양쪽이 수학적으로 동치.

> 이 padding correction 이 없으면 stage 마다 누적 오차가 발생해 정확도가 수 %p 떨어진다. 본 프로젝트의 max diff = 0 (보고서 §7.2) 은 이 4-step 의 정확한 구현 덕분.

---

## 12. 전체 학습 파이프라인

본 프로젝트의 학습 순서 (`train.ipynb` `main_pipeline()`).

### 12.1 Phase A — Teacher fine-tuning (80 epoch)

1. `torchvision.models.resnet18(pretrained=True)` 로 ImageNet 사전학습 모델 로드.
2. Stem 변경: 7×7 stride 2 + maxpool → 3×3 stride 1, no maxpool (CIFAR adapt).
3. fc 변경: 1000 classes → 100 classes.
4. CIFAR-100 train 45 000 장으로 80 epoch FT (fine-tuning).
5. Augmentation: RandomCrop + Flip + RandAugment + Mixup + RandomErasing.
6. Optimizer: SGD + momentum 0.9 + cosine LR schedule.
7. EMA (Exponential Moving Average) shadow 유지 — best val 시점의 shadow checkpoint 저장.
8. 최종 정확도: **EMA shadow 82.44%** (raw 약 81%).

### 12.2 Phase B — Student + Knowledge Distillation (220 epoch)

1. `A1W1ResNet18v2` instantiate (use_rprelu=True, use_double_skip=True).
2. Teacher 의 학습된 weight 중 **shape 가 일치하는 것** 만 student 에 복사 (stem, block conv 의 weight). RPReLU 의 γ, β, threshold 등은 student 초기값 유지.
3. Phase A teacher 를 fixed teacher 로 두고 KD 활성.
4. 220 epoch 학습:
   - Epoch 0–14: warmup, KD off (CE loss only).
   - Epoch 15–219: KD on (CE + KL 혼합), AT off.
5. Optimizer: Adam + cosine LR. Student EMA **disable** (binary weight 의 평균은 무의미).
6. 매 step 후 latent W clamp `[-1, +1]`.
7. 최종 정확도: **best raw 68.72%** (val).

### 12.3 Phase 결과 파일

- `.agents/best_teacher.pth` — 82.44% checkpoint.
- `.agents/best_student.pth` — 68.72% checkpoint (latent FP32 weights).
- `.agents/avx_demo/best_student.bnn` — packed 1-bit weights (binary serialization, 추론용).

---

## 13. 평가 지표

### 13.1 Top-1, Top-5 accuracy

- **Top-1**: 모델의 가장 높은 확률 class 가 정답과 일치한 비율.
- **Top-5**: 모델의 상위 5 개 후보 안에 정답이 있는 비율.

예: 정답 = `cat`. 모델 예측 확률 top-5 = `[dog, cat, lion, fox, tiger]`.
- top-1: dog 가 1 등이라 cat ≠ → **틀림**.
- top-5: 상위 5 개 안에 cat 있음 → **맞음**.

본 프로젝트 보고:
- Student top-1: 68.72%
- Student top-5: 약 91%
- Teacher top-1: 82.44%
- Teacher top-5: 약 96%

### 13.2 EMA (Exponential Moving Average) shadow

학습 중 weight 가 매 step 흔들리는 걸 부드럽게 만든 사본:
```
W_shadow_new = decay · W_shadow_old + (1 - decay) · W_current
```

`decay = 0.999` → 매 step 의 1‰ 만 반영, 천천히 따라감.

→ 일반적으로 **EMA shadow 의 정확도가 raw 보다 1~2 %p 높음**. Teacher 에서는 EMA shadow 의 best epoch 을 사용 (82.44%).

> Student 는 binary weight (sign 함수 출력) 의 평균이 random 에 가까워지므로 EMA shadow 가 의미 없음 → **disabled** (ReActNet / Bi-Real 모두 동일).

### 13.3 Class-wise accuracy

class 별 정확도 = (그 class 의 맞춘 sample 수) / (그 class 의 총 sample 수).

본 프로젝트 §6.7 분석:
- Overall: 67.86% (10K standard CIFAR-100 test set).
- Worst class: boy 36%, seal 36%, lizard 37%, ...
- Best class: road 95%, orange 94%, wardrobe 93%, ...

→ 1-bit BNN 의 정확도 손실이 **fine-grained intra-cluster** (포유류 fine-grained, 사람 age/gender, 나무 종 구별) 에 집중됨.

---

## 14. Loss · Optimizer · LR Schedule

### 14.1 Cross-Entropy (CE) loss

100-class 분류의 표준 loss:
```
CE(p_s, y) = - log p_s[y]
```

- `p_s` = student 의 softmax output (100 차원).
- `y` = 정답 class index (예: 42).
- 정답 class 의 확률이 1 에 가까울수록 loss → 0.

예: `p_s = [0.05, 0.80, 0.10, 0.05]`, 정답 `y = 1` →
```
CE = - log(0.80) = 0.223
```

### 14.2 Optimizer — SGD vs Adam

**SGD** (Stochastic Gradient Descent):
```
W_new = W_old - lr · ∇L
```

**SGD + Momentum**:
```
v_new = β · v_old + ∇L         (β = 0.9 보통)
W_new = W_old - lr · v_new
```
→ 이전 step 의 방향성을 일부 이어받음 → 진동 줄이고 안정적.

**Adam** (Adaptive Moment Estimation): 각 weight 의 step size 를 그 weight 의 gradient 통계에 맞춰 자동 조정.
- Phase A (teacher): SGD + momentum 0.9, weight decay 1e-4 (보고서 §3.3 ablation).
- Phase B (student): Adam (BNN literature 관례, latent weight 의 불안정성 흡수).

### 14.3 LR (Learning Rate) Schedule — Cosine

학습 진행에 따라 lr 을 점진적으로 줄여 final convergence 의 정밀도 향상.

```
lr(t) = lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π · t / T))
```

- 초반: lr_max 근처 (큰 폭으로 탐색).
- 말기: lr_min 근처 (정밀 수렴).
- `t` = 현재 epoch, `T` = 총 epoch.

본 프로젝트:
- Teacher: lr_max = 1e-3, lr_min = 1e-6, T = 80.
- Student: lr_max = 1e-3, lr_min = 1e-6, T = 220.

### 14.4 Warmup — 초반 lr 을 천천히 올리기

```
epoch 0–14:   lr_warmup_t = (t + 1) / 15 · lr_max
epoch 15–:    cosine decay
```

학습 초반 weight 가 무작위 초기화 → 큰 lr 로 시작하면 발산 위험. warmup 으로 안전하게 시작.

본 프로젝트 student warmup = 15 epoch. KD 도 같은 시점에 활성화.

---

## 15. 부록 — 용어집 + FAQ

### 15.1 용어집 (한 줄 정의)

| 용어 | 의미 |
|---|---|
| **BNN** (Binary Neural Network) | weight & activation 이 1-bit (±1) 인 신경망 |
| **W1A1** / **A1W1** | Weight 1-bit + Activation 1-bit 의 약자 |
| **STE** (Straight-Through Estimator) | forward 는 sign, backward 는 identity 의 비대칭 gradient 트릭 |
| **Latent weight** | 학습 중 FP32 로 유지되는 진짜 학습 변수. inference 시 버림 |
| **MAC** | Multiply-Accumulate, 1 곱셈 + 1 누적 |
| **FLOPs** | Floating Point Operations. 흔히 2 × MACs |
| **BitOPs** | 1-bit 연산 (XNOR, POPCNT). 1 MAC ≈ 64 BitOPs 의 가치 |
| **KD** (Knowledge Distillation) | teacher 의 출력을 student 가 흉내내는 학습 방식 |
| **Soft target** | teacher 의 softmax output (확률 분포) |
| **CE** (Cross-Entropy) | 분류 표준 loss |
| **KL Divergence** | 두 확률 분포 간 비대칭 거리. KD loss 의 핵심 |
| **Temperature T** | softmax 의 sharpness 조절. KD 시 T > 1 |
| **EMA** | weight 의 천천히 따라가는 사본. teacher 정확도 보조 |
| **Residual / Skip connection** | `y = F(x) + x` 형태의 우회 경로 |
| **Double-skip** | ReActNet 의 block 당 2 개 residual |
| **RSign** | 학습 가능 threshold 가 있는 sign — `sign(x - α)` |
| **RPReLU** | 학습 가능 shift (γ), bias (β), slope 가 있는 activation |
| **XNOR** | NOT XOR. 두 bit 가 같으면 1 |
| **POPCNT** | bit vector 안의 `1` 개수 세기. CPU 명령어 1 clock |
| **Padding correction** | binary conv 의 zero-padding 을 +1 packing + 보정으로 흉내내는 트릭 |
| **AVX-512** | Intel CPU 의 SIMD 명령어 집합. 512-bit 단위 |
| **VPOPCNTDQ** | AVX-512 의 SIMD POPCNT. 본 프로젝트는 미사용 |

### 15.2 자주 헷갈리는 것 (FAQ)

**Q1. 왜 STE 의 backward 가 identity 인데 학습이 되나요?**
→ Latent weight 가 FP32 로 살아있고, gradient 는 그쪽으로 흘러들어 학습됨. binary 는 forward 의 출력일 뿐, 학습 대상은 아님.

**Q2. KL divergence 의 방향 `KL(p_t || p_s)` 가 왜 `KL(p_s || p_t)` 가 아닌가요?**
→ KL 은 비대칭. `KL(p_t || p_s)` 는 "teacher 가 기준, student 가 그 분포를 흉내내라" 의 뜻. 반대는 student 가 기준이 되어 학습 신호가 약해짐. Hinton (2015) + PyTorch `F.kl_div` 의 표준.

**Q3. EMA 가 왜 teacher 만 적용되고 student 는 안 되나요?**
→ Student 의 weight 는 학습 중에는 FP32 latent W 인데, 실제 model 의 weight 는 `sign(W)` 의 ±1. EMA shadow 는 `sign(W_shadow)` 가 random 모델에 가까워짐 → 의미 없음. (Bi-Real, ReCU, ReActNet 모두 동일.)

**Q4. 25× 가 PyTorch 대비인가요, C teacher 대비인가요?**
→ **동일 C 런타임 안의 student vs teacher**. PyTorch FP32 (BLAS 가속) 와 비교하면 25× 가 안 나옴. "알고리즘 차이의 정직한 측정" 으로 동일 환경 비교 (보고서 §7.4, §8).

**Q5. "메모리 28× vs 31.4× vs 32×" 어느 게 맞나요?**
→ 셋 다 측정값:
- **28×** = bit 예산 (FP32 32-bit vs 1-bit + overhead).
- **31.4×** = 디스크 `.pth` 파일 크기 비율.
- **32×** = 이론 상한 (overhead 무시).

**Q6. RPReLU 가 9.22 %p 단독 기여인데, 천장 71% 도 RPReLU 덕인가요?**
→ 부분적으로. RPReLU 는 **단일 최대 기여 컴포넌트** 이지만, 천장에 도달하려면 KD + double-skip + α detach 등 다른 컴포넌트도 필요. Cross-component interaction 은 미측정 (보고서 §6.6).

**Q7. Padding correction 이 없으면 어떻게 되나요?**
→ Stage 마다 누적 오차 → 정확도 수 %p 손실 + PyTorch 와 bit-exact 불일치. 본 프로젝트의 max diff = 0 (§7.2) 의 핵심 enabler.

**Q8. AT (Attention Transfer) 는 왜 안 썼나요?**
→ 본 환경에서 weight 1000 → 10 ramp-up 도 BN running stats 불안정으로 실패. Logit KD only 가 최종 디자인. (보고서 §5.3 / D9.)

**Q9. T4 GPU 가 INT1 지원하면 안 되나요?**
→ NVIDIA Turing 세대 (T4) 는 **INT4 까지만 native**. INT1 native 는 A100/H100 (Ampere 이상). T4 에서 binary 모델을 돌려도 fp16 conv 비용 그대로라 GPU 가속의 의미가 없음 → 본 프로젝트는 **CPU AVX demo 로 우회**.

**Q10. 코드 어디서부터 봐야 하나요?**
→ 추천 순서:
1. `train.ipynb` cell 6 — `BinaryWeightSTE`, `BinaryActivationSTE`, `RSignActivation`, `RPReLU`, `A1W1BinaryConv2dV2`, `A1W1BasicBlockV2`, `A1W1ResNet18v2` (model 정의)
2. `train.ipynb` cell 11 — `_run_phase()` (학습 루프, KD loss 가 어디서 적용되는지)
3. `.agents/avx_demo/inference_c.py` — `BinaryInferenceEngine.forward()` (C 추론 호출)
4. `.agents/avx_demo/xnor_kernel.c` — `xnor_gemm()` (XNOR+POPCNT 의 실제 C 구현)

---

## 끝맺음

이 노트의 모든 개념이 `report.md` / `slides.md` / `train.ipynb` 의 어디에 등장하는지 한 줄 요약:

| 개념 | 코드 위치 | 보고서 위치 |
|---|---|---|
| Binary weight / activation STE | `train.ipynb` cell 6 | §4.1 |
| RSign | cell 6 `RSignActivation` | §4.1 |
| RPReLU | cell 6 `RPReLU` | §4.2 |
| Double-skip | cell 6 `A1W1BasicBlockV2` | §4.2 |
| Detached α | cell 6 `A1W1BinaryConv2dV2` | §4.1 |
| KD logit loss | cell 11 `train_epoch` | §5.2 |
| Phase A/B split | cell 11 `main_pipeline` | §5.1, §5.2 |
| XNOR+POPCNT C kernel | `xnor_kernel.c::xnor_gemm` | §7.1 |
| Padding correction | `xnor_kernel.c::binary_conv2d` | §7.1, §7.2 |
| Class-wise eval | `eval_class_wise.py` | §6.7 |
| BitOPs 계산 | `compute_bitops.py` | §6.4 |
| Ablation curves | `plot_curves.py` | §6.6 |

추가로 모르는 게 있으면 본 노트의 해당 챕터로 돌아오거나, 부록 §15 에서 용어 찾아보면 됨.
