---
marp: true
theme: default
paginate: true
size: 16:9
math: mathjax
header: '**Team 16** · 1-bit BNN on CIFAR-100'
footer: 'KAU Deep Learning Term Project · 2026'
style: |
  section { font-size: 24px; line-height: 1.3; padding: 50px 60px 70px; }
  h1 { color: #1a3a8a; }
  h2 { color: #1a3a8a; border-bottom: 2px solid #1a3a8a; padding-bottom: 4px; margin-top: 0; }
  table { font-size: 20px; }
  code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  .small { font-size: 18px; color: #555; }
  .green { color: #1a8a3a; font-weight: bold; }
  .red { color: #b22222; font-weight: bold; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# CIFAR-100 1-bit Binary Neural Network with Knowledge Distillation

### Team 16 · KAU Deep Learning Term Project

**1-bit weight + 1-bit activation BNN** with logit distillation
& **AVX-CPU inference demo** (25× speedup)

<br>

<span class="small">
ReActNet-lite ResNet18 · CIFAR-100 · Knowledge Distillation · POPCNT-based C inference
</span>

<!--
Speaker note: 30초.
"안녕하세요, Team 16입니다. CIFAR-100 분류를 1-bit BNN으로 학습하고, AVX CPU에서 실제로 25배 추론 가속을 측정한 프로젝트를 발표하겠습니다."
-->

---

## 1. Motivation & Goal

- **문제**: 클라우드 의존 없는 **on-device AI inference** 수요가 커지지만, 모바일·노트북·엣지의 메모리·전력 예산은 모델 크기를 못 따라감
  - ResNet18 (CIFAR-100 head) ≈ 11.2M params, **44 MB FP32**
- **해결 아이디어**: Binary Neural Network (BNN)
  - Weight·Activation 을 ±1 (1-bit) 로 양자화 → 이론 **32× 메모리 압축**
  - MAC 연산을 **XNOR + POPCNT** 로 대체 → 정수·HW 가속 친화적
- **본 프로젝트 목표**
  1. CIFAR-100 W1A1 BNN 학습 → literature SOTA 영역 (~69%) 도달
  2. Teacher 를 empirical study 로 강화 후 KD 로 student 정확도 push
  3. **POPCNT 기반 C 추론 커널** 로 inference 구현해 CPU 환경 (Intel Core i5-1135G7) 에서 speedup 실측

<!--
Speaker note: 50초.
"BNN은 32배 메모리 압축과 XNOR-POPCNT 기반 정수 연산이 핵심. 본 프로젝트는 정확도와 HW 추론 두 가지 목표를 동시에 진행했습니다. 70% literature SOTA 영역 도달을 목표로 잡았습니다."
-->

---

## 2. Method Overview

### Why **ReActNet-lite**?

- 기존 XNOR-Net / Bi-Real 대비, 학습 가능한 **RSign + RPReLU + double-skip** 로 1-bit 정보 손실을 안정적으로 보완 → **W1A1 ImageNet 69.4% (현 SOTA 영역)**
- 본 프로젝트는 ResNet18 backbone 에 ReActNet 의 핵심 트릭만 이식한 "lite" 변형

### 모델 구성

```
99% Binary (8 binary residual blocks, RSign + BinConv + RPReLU + double-skip)
 1% FP32  (Conv1 stem + final FC)   ← BNN 표준: 200 KB로 5-10%p 정확도 보존
```

### Distillation

- **Teacher**: ResNet18 FP32 (82.44%)
- **Student**: A1W1ResNet18 (W1A1)
- **Loss**: $L = 0.3 \cdot \mathrm{CE}(\hat y_s, y) + 0.7 \cdot T^2 \cdot \mathrm{KL}(p_t^{T} \,\Vert\, p_s^{T}),\ T=4$

<!--
Speaker note: 60초.
"ReActNet-lite를 선택한 이유는 RSign과 RPReLU, double-skip이 1-bit 정보 손실을 가장 효과적으로 보완하기 때문입니다. 99%를 binary로, stem과 분류기 1%만 FP32로 남겼고, KD는 logit KL을 weight 0.7로 적용했습니다."
-->

---

## 3. Student Architecture: `A1W1ResNet18`

<style scoped>
section { font-size: 21px; }
table { font-size: 18px; }
</style>

> ReActNet 의 핵심 트릭 (RSign + RPReLU + double-skip + detached α) 차용 + 두 가지 단순화: **backbone** ReActNet-A → **표준 ResNet18**, **학습** ReActNet 의 2-stage W32A1→W1A1 KD → **단일 stage** W1A1 KD. → 그래서 "lite".

### Binary Residual Block (double-skip + RPReLU)

```
x --+-- BN - RSign - BinConv - BN - RPReLU --+-- +res_a --+
    |                                        |            |
    +----------------------------------------+            |
                                                          |
   -+-- BN - RSign - BinConv - BN - RPReLU --+-- +res_b --+
    |                                        |
    +----------------------------------------+
```

### 핵심 수식

$$
y_{c_o,\,i,\,j} \;=\; \underbrace{\alpha_{c_o}}_{\text{detached}}
\cdot \Bigl( 2 \cdot \mathrm{popcount}\bigl( \lnot ( A_{i,j} \oplus W_{c_o}) \bigr) \;-\; K_{\text{bits}} \Bigr)
$$

| 구성요소 | 역할 | 학습 가능 |
|---|---|---|
| `RSign` (per-channel β) | threshold 학습 sign | ✅ |
| `BinaryConv (sign W, sign x)` | XNOR + POPCNT | weight ✅ |
| α = E[\|W\|] **detached** | per-channel scale | ❌ graph 차단 |
| `RPReLU` (γ, slope, β) / shortcut | shift + PReLU + bias / Bi-Real | ✅ / — |

<!--
Speaker note: 70초.
"Block은 Bi-Real의 double-skip을 가져왔고, 활성에 RSign과 RPReLU를 추가했습니다. 핵심 수식 한 줄로: 2 곱하기 popcount 마이너스 K_bits에 채널별 스케일 알파를 곱합니다. 알파는 forward만 보정하도록 graph에서 detach합니다."
-->

---

## 4. Teacher: Empirical Study (CSV 7-step ablation)

<style scoped>
section { font-size: 22px; }
table { font-size: 19px; }
</style>

체계적 ablation 으로 baseline ResNet18 을 **75.74% → 83.12%** 까지 끌어올림.

| Step | 변경 사항 | Top-1 | 변화 |
|---|---|---|---|
| 0 | Bilinear resize baseline | 75.74 % | — |
| 1 | Stem: Conv1 7×7 stride=1, **maxpool → Identity** | **81.82 %** | **+6.08 %p** |
| 2 | Optimizer: **AdamW (wd=1e-4)** | 82.38 % | +0.56 |
| 3 | Batch 1024, LR 1e-3 | 82.38 % | — |
| 4 | Scheduler: CosineAnnealingLR | 82.38 % | — |
| 5 | Activation: **GELU** | 82.46 % | +0.08 |
| 6 | Mixup decay: **step** | 82.54 % | +0.08 |
| 7 | Aug: **RandAugment num_ops=2, magnitude=10** | **83.12 %** | **+0.58** |

> **Step 1 (stem 변경)** 이 +6.08 %p 단일 기여 — 32×32 입력에 ImageNet stem 의 downsampling 이 손실 요인.
> Phase-A: pretrained FT 80 ep + **ReLU + mag 9** → **82.44 %** (ablation best GELU + mag 10 은 from-scratch 환경에서만 검증, pretrained 호환성으로 채택 안 함).

<!--
Speaker note: 70초.
"Teacher 학습은 7단계 ablation으로 진행했습니다. baseline 75.7%에서 시작해 최종 83.1%까지 끌어올렸고, 단일 변경 중 stem 어댑테이션이 +6%p로 가장 큰 영향을 줬습니다. 32×32 입력에 ImageNet의 7×7 stride 2와 maxpool은 다운샘플링이 과합니다."
-->

---

## 5. Training Pipeline

<style scoped>
section { font-size: 23px; }
table { font-size: 19px; }
</style>

### Single Kaggle session (T4, 12h budget)

| Phase | 내용 | Epoch | Time |
|---|---|---|---|
| **A** | Teacher (ImageNet pretrained → CIFAR FT) | 80 | ~15 min |
| **B** | Student (A1W1 + Logit KD T=4, w=0.7) | 220 | ~80 min |
| | **Total** | | **~1h35** |

### Phase B 핵심 hyperparam

- **Optimizer**: AdamW (wd = 0, base lr = 5e-4, warmup 15 ep + cosine 205 ep)
- **Mixup α = 0.4 (step decay)** · **CutMix** · **RandomErasing p = 0.2**
- **Grad clip norm = 5.0** (binary STE jump 안정화)
- **AMP autocast fp16** (Tensor Core 활용, sign/STE 는 fp16 에서 안정)
- **Student init**: teacher binary-호환 layer 가중치로 warm-start

<!--
Speaker note: 55초.
"단일 Kaggle 세션에서 phase A 80 epoch, phase B 220 epoch을 약 1시간 반에 마칩니다. KD는 logit KL, T=4, weight 0.7. Mixup은 step decay, grad clip 5.0, AMP fp16을 적용했습니다."
-->

---

## 6. Quantitative Results

<style scoped>
section { font-size: 23px; }
table { font-size: 20px; }
</style>

### CIFAR-100 Top-1

| Model | Params | Memory | Top-1 |
|---|---|---|---|
| Teacher (FP32 ResNet18, stem adapted) | 11.22 M | 44.0 MB | <span class="green">82.44 %</span> |
| **Student (A1W1 ReActNet-lite)** | **11.22 M** | **1.4 MB** | <span class="green">**68.72 %**</span> ¹ |

### Literature comparison (W1A1)

| Method | Dataset | Top-1 |
|---|---|---|
| XNOR-Net (ECCV'16) | ImageNet | 51.2 % |
| Bi-Real (ECCV'18) | ImageNet | 56.4 % |
| **ReActNet (ECCV'20)** | **ImageNet** | **69.4 %** |
| ReCU (ICCV'21) | ImageNet | 66.4 % |
| RBNN | CIFAR-100 | ~ 67.1 % |
| **Ours** | **CIFAR-100** | **68.72 %** |

> 📦 **44 MB → 1.4 MB · 32× 압축** &nbsp;·&nbsp; ⚠️ **ReActNet 격차 –0.7 %p** (1-bit 천장 ~ 71 %)
>
> ¹ Ablation baseline rerun 은 68.74 % (best, ep 215) — main result 와 0.02 %p 차이로 noise 범위. 자세한 ablation 은 Backup F.

<!--
Speaker note: 70초.
"Student 68.7%는 literature SOTA 영역입니다. CIFAR-100 RBNN 67.1% 대비 +1.6%p이고, ReActNet ImageNet 69.4%와 0.7%p 차이입니다. 메모리는 44MB에서 1.4MB로 32배 압축, 이론 상한에 매우 근접합니다."
-->

---

## 7. Design Choices & Discussion

<style scoped>
section { font-size: 23px; }
</style>

- **Student EMA 비활성**: binary weight 평균은 `sign()` 부수므로 random 모델이 됨 → ReActNet, Bi-Real, ReCU 모두 동일
- **FP32 stem + classifier (1%)**: 200 KB 만 차지하면서 +5–10 %p 정확도 보존 (BNN 표준)
- **α 를 graph 에서 detach**: 안 하면 weight 가 sign 미분 0 영역으로 끌려 발산
- **AMP fp16 binary STE**: ±1 sign 은 fp16 동적 범위 내 안정, Tensor Core 활용 → 학습 1.5–2 ×
- **Ablation 으로 검증된 컴포넌트 기여도** (Backup F):
  - **RPReLU 가 단일 최대 기여 (−9.22 %p when removed)** > Double-skip (−2.46) ≈ KD (−2.30)

<!--
Speaker note: 50초.
"몇 가지 디자인 결정을 강조합니다. Student EMA는 sign을 부수므로 비활성, stem과 분류기 1%만 FP32 유지, 알파 스케일은 detach. 그리고 ablation 결과 RPReLU가 9.22%p로 단일 최대 기여 컴포넌트임을 검증했습니다."
-->

---

## 8. C Inference: Design + Speedup

<style scoped>
section { font-size: 22px; }
table { font-size: 19px; }
</style>

### C kernel (`xnor_kernel.c`, ~500 LOC, POPCNT + AVX-512 VPOPCNTDQ + OpenMP)

- `xnor_gemm`: packed uint64 + scalar `__builtin_popcountll` + cache-blocked, M-axis OpenMP
- `xnor_gemm_avx512`: `_mm512_popcnt_epi64` 인트린식 (AVX-512 VPOPCNTDQ) → **scalar 대비 GEMM 코어 2.2× 추가 speedup** (Ice Lake+ CPU)
- **Padding correction**: zero-pad ↔ binary {-1,+1} 인코딩 불일치를 패딩 패턴별 lookup 으로 보정 → **bit-exact 달성의 핵심 trick**
- 기타: `fp32_conv2d` (stem), `batch_norm_2d`, `rsign`, `rprelu`, `linear_fp32`

### Bit-exact verification

> **PyTorch ↔ C max diff = 0.0000** ✅ (test set 10 000 sample 모두)

### Wall-clock latency (CPU: Intel Core i5-1135G7, 4C / 8T)

| 모드 | Student (W1A1) | Teacher (FP32) | **Speedup** | 의미 |
|---|---|---|---|---|
| **Single-thread** | **22 ms** | **547 ms** | <span class="green">**25 ×**</span> | 알고리즘 차이 |
| 4-thread (OpenMP) | 15 ms | 173 ms | <span class="green">12 ×</span> | 실용 환경 |

> <span class="small">Ratio 감소 이유: Teacher 큰 GEMM 으로 multi-thread 이득 3× / Student 작은 binary conv 16 개로 1.5×. 두 수치 모두 정직한 측정값.</span>

<!--
Speaker note: 75초.
"C 커널은 약 350줄, XNOR 곱셈과 POPCNT, padding correction, BN, RSign, RPReLU, FP32 stem과 분류기까지 포함합니다. PyTorch와 bit-exact 일치를 검증했고, Intel Core i5-1135G7 노트북 CPU에서 single-thread 25배, 4-thread 12배 speedup을 측정했습니다. 두 수치 모두 정직한 측정값입니다."
-->

---

## 9. Limitations & Honest Discussion

<style scoped>
section { font-size: 23px; }
</style>

- **W1A1 SOTA 격차 ~0.7 %p**: ReActNet 69.4 % vs Ours 68.7 %. 1-bit 의 구조적 천장 (~71 %)
- **T4 GPU INT1 native 미지원** (Turing 은 INT4 까지) → GPU 에서 binary 추론 가속 불가, **CPU AVX demo 로 우회**
- **Naive C 비교 한계 + BLAS 보강 측정**: 메인 C 는 VPOPCNTDQ / SIMD-GEMM 미사용. 보강으로 BLAS-backed teacher (OpenBLAS, im2col + sgemm) 와 비교 시 student/teacher 비율은 **1-thread 3.15×, 4-thread 6.12×** (정직한 production 비교). 25× 는 알고리즘 차이, 3.15× 는 production-grade 차이 — 둘 다 측정값.
- **Ablation 범위 한계** (§6.5 / Backup F 참조):
  - 단일 컴포넌트 ablation (KD/double-skip/RPReLU) 은 수행 — RPReLU 가 최대 기여 (−9.22 %p)
  - cross-component interaction (예: E3 + E1), KD weight sweep, AT 안정화는 미수행
- **AT loss 실패**: layer-wise LR 분리, BN sync, weight scheduling 등 추가 작업 필요

<!--
Speaker note: 50초.
"솔직한 한계를 정리합니다. SOTA 격차 약 0.7%p, T4 GPU INT1 미지원으로 CPU demo로 우회, naive C 구현의 한계, ablation 미실시. 그리고 AT loss는 안정화에 실패했습니다."
-->

---

## 10. Conclusion + Future Work

<style scoped>
section { font-size: 24px; }
</style>

### 핵심 성과 (3줄 요약)

1. **CIFAR-100 W1A1 BNN 68.72 % top-1** — literature SOTA 영역 (ReActNet 69.4 % 대비 –0.7 %p)
2. **메모리 32× 압축** (44 MB → 1.4 MB) 와 **체계적 teacher empirical study** (75.7 → 83.1 %)
3. **AVX-CPU C 추론 demo** — PyTorch 와 bit-exact, single-thread **25× speedup** 실측

### Future Work

- Cross-component ablation (E1 × E2 × E3 의 2-/3-way interaction), KD T/w sweep
- SAM optimizer 도입 (BNN × SAM 효과 검증)
- Bit-width Pareto (W8A8 / W4A4 LSQ) 비교
- AT loss 안정화 (layer-wise LR, BN sync)
- AVX-512 VPOPCNTDQ + BLAS-style GEMM → production-grade C 커널 확장

---

<!-- _class: lead -->
<!-- _paginate: false -->

## Q & A

<br>

**Team 16** · KAU Deep Learning Term Project

<span class="small">
report.md · slides.md · .agents/avx_demo/xnor_kernel.c<br>
</span>

---

<!-- Backup slide -->
## Backup A — Phase A Teacher Hyperparam

<style scoped>
section { font-size: 23px; }
table { font-size: 20px; }
</style>

| 항목 | 값 |
|---|---|
| Backbone | ResNet18 (ImageNet pretrained) |
| Stem | Conv1 7×7 stride 1, maxpool → Identity |
| Optimizer | AdamW (wd = 1e-4) |
| Scheduler | LinearWarmup (8 ep, 0.1→1.0) + Cosine (72 ep, η_min = 1e-6) |
| LR / Batch / Epoch | 1e-3 / 1024 / 80 |
| Augmentation | RandAugment (num_ops = 2, magnitude = 9), RandomErasing p = 0.2 |
| Regularization | Label smoothing 0.05, Mixup α = 1.0 (step decay → 0.0625), CutMix |
| EMA | shadow decay 0.999 (teacher 만) |
| Final top-1 | **82.44 %** |

---

<!-- Backup slide -->
## Backup B — Phase B Student Hyperparam

<style scoped>
section { font-size: 21px; }
table { font-size: 17px; }
</style>

| 항목 | 값 |
|---|---|
| Model | `A1W1ResNet18` (8 binary residual blocks, double-skip) |
| Activation / scale | `RSign` (per-ch β) + `RPReLU` (per-ch γ, slope, β) ; α = E[\|W\|] **detached** |
| Stem / FC | FP32 (~ 1 % params) |
| Optimizer / LR / Batch / Epoch | AdamW wd = 0 / 5e-4 / 1024 / 220 |
| Scheduler / Grad clip | warmup 15 ep + cosine 205 ep / norm 5.0 |
| Mixed precision | autocast fp16 |
| KD Temperature / weight | T = 4.0 / w = 0.7 |
| Attention Transfer | **disabled** (§7) |
| EMA | **disabled** (binary weight avg 무의미) |
| Student init | teacher binary-호환 weight warm-start |

---

<!-- Backup slide -->
## Backup C — Data Augmentation Strategy

<style scoped>
section { font-size: 20px; }
table { font-size: 16px; }
</style>

GPU-side pipeline (`get_gpu_transforms()`), Phase A & B 공통.

| 분류 | 기법 | 설정 |
|---|---|---|
| Image-level | **RandomCrop** | size 32, padding 4 (reflect) |
| Image-level | **RandomHorizontalFlip** | p = 0.5 |
| Image-level | **RandAugment** | num_ops = 2, magnitude = 9 |
| Image-level | **RandomErasing** | p = 0.2, default scale `(0.02, 1/3)`, random |
| Label-mix | **Mixup** | α = 1.0 (T) / 0.4 (S), step decay |
| Label-mix | **CutMix** | random patch 교체, batch 별 50 % 확률 |

**RandAugment ops** (num_ops=2 sample 마다 random 선택) — Geometric (Rotate, Shear, Translate) + Color (Brightness, Contrast, Color, Sharpness) + Pixel-level (Posterize, Solarize, AutoContrast, Equalize, Invert).

**Mixup / CutMix** &nbsp;&nbsp; $\hat{x} = \lambda x_A + (1-\lambda) x_B,\ \hat{y} = \lambda y_A + (1-\lambda) y_B,\ \lambda \sim \mathrm{Beta}(\alpha, \alpha)$

→ Mixup 픽셀 blending / CutMix 공간 patch 교체. 서로 다른 invariance.

---

<!-- Backup slide -->
## Backup D — C Kernel Detail

<style scoped>
section { font-size: 20px; }
table { font-size: 16px; }
</style>

### `xnor_gemm` (core primitive)

```c
y[c_o, n] = α[c_o] * (2 * popcount(~(A[n] ^ W[c_o])) - K_bits)
```

- Packed `uint64_t` (64 bits / word) · `_mm_popcnt_u64` scalar POPCNT (AVX-512 VPOPCNTDQ 미사용)
- Cache-blocked N-axis, M-axis OpenMP 병렬화

### 함수 목록 (`xnor_kernel.c`, ~350 LOC)

| 함수 | 역할 | OpenMP |
|---|---|---|
| `xnor_gemm` / `xnor_gemm_padded` | Binary GEMM + K_bits padding 보정 | ✅ M-axis |
| `fp32_conv2d` | FP32 stem & teacher | ✅ C_out |
| `binary_conv2d` | 1-bit conv + im2col + padding correction | ✅ 출력위치 |
| `batch_norm_2d` / `rsign` / `rprelu` | Inference BN / RSign / RPReLU | ✅ 채널축 |
| `shortcut_downsample` / `linear_fp32` / `adaptive_avgpool_1` | AvgPool + zero-pad / Classifier 등 | — |

---

<!-- Backup slide -->
## Backup E — Threading 25× → 12× 의 이유

<style scoped>
section { font-size: 23px; }
table { font-size: 20px; }
</style>

| Threads | Student (W1A1) | Teacher (FP32) | Ratio |
|---|---|---|---|
| 1 (default) | 22 ms | 547 ms | **25 ×** |
| 4 | 15 ms | 173 ms | **12 ×** |
| 8 (HT 포함) | 21 ms | 200 ms | 9 × |

**왜 thread 수에 따라 ratio 가 변하는가**

- Teacher (FP32): 큰 GEMM 루프 → multi-thread 로 ~3× 가속
- Student (binary): 작은 op 이 많아 (16 binary conv) thread spawn overhead 로 ~1.5× 만 가속
- → 4-thread 시 teacher 가 더 큰 이득 → ratio 25 → 12 × 감소
- **두 숫자 모두 정직한 측정값**. 발표에서는 single-thread (알고리즘 차이) 와 multi-thread (실용) 모두 제시

> Default = single-thread (Codex review 의 "fair comparison" 가이드 준수)

---

<!-- Backup slide -->
## Backup F — Student Design Ablation

<style scoped>
section { font-size: 22px; }
table { font-size: 19px; }
img { max-height: 320px; }
</style>

220 epoch 동일 schedule, baseline 에서 단일 컴포넌트만 제거

| Variant | Best Val | Δ |
|---|---|---|
| **Baseline** (KD + double-skip + RPReLU) | **68.74 %** | — |
| E1: KD off | 66.44 % | −2.30 |
| E2: Single-skip | 66.28 % | −2.46 |
| **E3: No RPReLU** | **59.52 %** | **−9.22** |

![Δ Top-1 vs baseline](ablation_bars.png)

- **RPReLU 가 단일 최대 기여 (−9.22 %p)** → activation reshape > weight precision
- KD ≈ double-skip (~2.3 %p) → 둘 다 gradient flow 보조
- Baseline 만 후반 fine refinement (ep 215 vs 196–203)

<!--
Speaker note (질문 받으면):
"RPReLU는 channel-wise learnable shift β로 1-bit activation 정보 손실을 보상하는데, 단독으로 9%p 차이를 만들었습니다. KD와 double-skip은 각각 2.3%p로 거의 같은 크기이고, 둘 다 gradient flow를 보조한다는 공통점이 있습니다."
-->
