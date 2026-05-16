# CIFAR-100 텀프로젝트 Plan v1 — A1W1 BNN 메인 + Pareto/KD/HW

## 0. 한 줄 요약
**ReActNet-lite (W1A1) ResNet18** 을 메인 제출 모델로 학습해 **CIFAR-100 Top-1 ≥ 70%** 를 노리고, 부수실험으로 **(a) bit-width Pareto frontier (b) Knowledge Distillation 효과 분리 (c) 하드웨어 inference 추정**을 함께 제시한다.

## 1. 목표 / 채점 포인트
| 항목 | 목표 |
|---|---|
| Submission accuracy | A1W1 student Top-1 ≥ 70% (Top-5 ≥ 90%) |
| FP32 teacher accuracy | ≥ 76% (안정적 KD source) |
| 독창성 1 | bit-width Pareto frontier 그래프 (FP32 / W8A8 / W4A4 / W2A2 / W1A1) |
| 독창성 2 | KD on/off, T sweep, teacher 강도(ResNet18 vs ResNet34) ablation |
| 독창성 3 | HW inference 추정 — BitOPs 이론치 + 가능하면 latency/메모리 실측 |
| 안전망 | A1W1 70% 미달 시 Plan B = W2A2 LSQ 메인으로 스위치 (목표 70~72%) |

## 2. 데이터셋 / 환경
- CIFAR-100 (Kaggle competition `26-deep-learning-course`)
- 입력: 32×32 PNG, train+val 100% / test 별도. 현재 코드 기준 train_val_split=0.90.
- 환경: Kaggle T4×2 여러 세션 + 로컬 GPU.
- 시드 42 고정, AMP 사용.

## 3. 모델 라인업

### 3.1 메인: ReActNet-lite (W1A1) ResNet18
- Backbone: ResNet18, CIFAR stem (conv1 3×3 stride 1, maxpool 제거).
- Binary block (BasicBlock 변형):
  ```
  x → BN → RSign(learnable per-ch threshold) → Conv1×1or3×3(W1) → BN
                              ↘ skip (avg-pool downsample, channel-pad) ↗
  → RPReLU(learnable γ, β, ζ) → next block
  ```
  - **RSign**: `sign(x − θ_c)` (이미 구현됨 — `RSignActivation`).
  - **RPReLU (신규 추가)**: `PReLU(x − γ_c) + β_c` 형태로 distribution reshape. 본 noteboook에는 미구현 → 추가 필요.
  - **Double-skip**: BasicBlock 출력 후 RPReLU와 함께 다시 한 번 identity residual을 더한다 (ReActNet 기법).
- Weight scaling: per-channel `α = mean|w|` (이미 있음). bias 사용 X.
- 첫 conv (stem)와 마지막 fc는 **FP32 유지** (관례, 정확도 하락 큰 두 레이어 보호).
- Stem 입력 quant 모드: `none` (BNN 메인 학습 단계에선 FP 유지). 발표용 HW 추정 시에만 int8 stem 가정.

### 3.2 Teacher: FP32 ResNet18 (필요시 ResNet34 추가 ablation)
- 현재 노트북 레시피 사용 (AMP+AdamW+Cosine+EMA+SWA+Mixup/CutMix+RandAug+RandomErasing+label smoothing 0.05).
- 목표: Top-1 76~78%.

### 3.3 부수실험 라인업
- **QATW4A4ResNet18** (이미 있음, W4 per-channel / A4 per-tensor)
- **QATW8A8ResNet18** (W4A4 코드의 비트 인자만 8로 변경)
- **QATW2A2ResNet18** *LSQ* 버전 추가 — 핵심 차별점 (learned step-size)
- **A1W1ResNet18** (메인)

## 4. 학습 레시피

### 4.1 Phase A — FP32 Teacher (300 epochs, ~6h on T4×2)
- Optimizer AdamW (lr 1e-3, wd 1e-4), Cosine, warmup 10% (lr×0.1→×1.0).
- Batch 1024, label smoothing 0.05.
- Aug: Pad+RandomCrop, HFlip, RandAugment(N=2, M=10), RandomErasing(p=0.2), Mixup+CutMix 0.5/0.5 prob, α step decay (1.0→0.0625).
- EMA 0.999, SWA 시작 epoch 240 (0.8×), SWA lr = min_lr.
- 산출물: `teacher_fp32.pth` (val acc EMA가 자기 자신보다 좋으면 SWA, 아니면 EMA 저장).

### 4.2 Phase B — A1W1 Student (2-stage, 총 ~300 epochs)

**Stage B1: W32A1 (활성만 binary, 가중치 FP32) — 120 epochs**
- 활성만 RSign으로 binary. 가중치는 FP32 그대로.
- Init: teacher의 ResNet18 FP32 가중치 그대로 복사 (`init_xnor_from_teacher` 활용).
- Optimizer AdamW (lr 5e-4, wd 0). Cosine + warmup 10%.
- KD: T=4, distill weight 0.7 (warmup 끝나면 시작).
- Mixup α 0.4→0.0 step decay, CutMix 같이 사용.
- 목표: 71~73% Top-1.

**Stage B2: W1A1 (가중치도 binary) — 180 epochs**
- Stage B1 가중치를 그대로 가져가 binary화. latent weight는 FP32로 유지 후 sign 적용.
- Optimizer AdamW (lr 1e-4, wd 0). Cosine + warmup 10 epochs.
- Weight clip `[-1, 1]` 매 step (이미 `clip_xnor_weights` 있음).
- KD: T=4, distill weight 0.8.
- Mixup α 0.2 fixed, label smoothing 0.0 (BN과 충돌 회피).
- gradient clip global norm 1.0 (수치 안정성).
- 목표: 68~70% Top-1.

> **선택지**: Phase B를 한 번에 W1A1로 학습하는 single-stage도 baseline으로 1회 돌려서 “2-stage가 효과 있다”를 보일 것.

### 4.3 Phase C — 부수실험 학습
- W8A8, W4A4, W2A2 (LSQ): 각각 200 epochs, FP32 teacher로 KD (T=4, w=0.5).
- W2A2는 가장 까다로움. LSQ는 `s`를 learnable scalar per-tensor(act)/per-channel(weight)로 두고 gradient scale = `1/sqrt(N×Q)`.
- 동일 augmentation/scheduler, 같은 random seed로 비교 공정성 유지.

### 4.4 Phase D — KD ablation (메인 A1W1 모델 기준, 각 200 epochs)
1. KD off (baseline)
2. KD on, T=2, w=0.7
3. KD on, T=4, w=0.7  ← 메인 레시피
4. KD on, T=4, w=0.7, teacher=ResNet34 (더 강한 teacher)
- 같은 seed/recipe. 결과는 Top-1, Top-5, convergence epoch 함께 비교.

### 4.5 Phase E — HW inference 추정
**이론 BitOPs** (가장 확실, 발표 그래프용):
- 레이어별 FLOPs를 `thop`/`fvcore.nn.FlopCountAnalysis`로 측정.
- `BitOPs = FLOPs × W_bit × A_bit` (관례). FP32는 32×32 가정.
- Weight memory = `Σ (num_params × bit / 8)` bytes.
- 표: 모델별 Top-1 / Params(bits) / BitOPs / Weight memory.

**실측 latency (가능 범위 내)**:
- PyTorch `torch.profiler` 로 FP32 vs A1W1 forward latency 측정 (CPU/GPU).
- 1-bit XNOR-popcount 가속은 vanilla PyTorch에 없으므로, “이론 가속비”를 함께 명시. `larq` 또는 `bitorch` 라이브러리 OR 단순 numpy로 popcount 시뮬레이션 가능.
- Optional: ONNX export + ONNX Runtime CPU에서 FP32 vs INT8(W8A8) 실제 latency 비교.

**에너지 추정** (옵션):
- 논문 표준값 사용: 45nm CMOS 기준 mult 비용 ≒ INT8 0.2pJ vs FP32 3.7pJ 등. 인용 출처 첨부.

## 5. 평가 / 제출 프로토콜
- 모든 실험 동일 split (seed 42, 90/10).
- Test set 제출은 메인 A1W1 student의 SWA가 아닌 **best val acc 시점 checkpoint** 기준 (binary는 SWA 효과 미미하다는 보고 있음 → A/B 확인 후 결정).
- Inference 시 BN을 conv에 fold 가능 (BN-fold). 발표 표 별도.

## 6. 위험요소 / Mitigation
| 위험 | Mitigation |
|---|---|
| A1W1 < 70% 가능성 (현실적으로 65~70%가 SOTA 근처) | ① W2A2-LSQ로 메인 교체 (목표 70~72%) ② “1-bit 한계 정직 보고”를 스토리로 살림 |
| BNN 수렴 불안정 | weight clip, low lr, mixup α 낮춤, 2-stage |
| Kaggle 세션 12h 끊김 | 매 epoch checkpoint, resume flag 사용 (`xnor_resume` 이미 있음) |
| Teacher 성능 부족 → KD 효과 미미 | RandAug N/M 증강, ResNet34 teacher 옵션 ablation 포함 |
| Augmentation 과도로 1-bit student 수렴 실패 | Stage B에서 augmentation 약화 (RandAug N=1,M=5), Mixup α 0.2 |

## 7. 일정 (가정: 가용 시간 2주, 일 평균 6h)
| Day | 작업 |
|---|---|
| D1 | 환경/데이터 sanity check, plan 확정 |
| D2-D3 | FP32 teacher 300ep 학습 + EMA/SWA |
| D4-D5 | Stage B1 (W32A1) 120ep |
| D6-D7 | Stage B2 (W1A1) 180ep |
| D8 | W8A8, W4A4 학습 |
| D9 | W2A2 LSQ 학습 |
| D10 | KD ablation (4 runs, 가능하면 epoch 축소 150ep) |
| D11 | HW 추정 (BitOPs/메모리/latency 측정) |
| D12 | 결과 정리, plot, 표 |
| D13-D14 | 발표 자료, 리허설 |

## 8. 산출물
1. `train.ipynb` — phase별 cell 정리.
2. `best_a1w1_resnet18.pth` (메인 제출).
3. `pareto_results.csv` + `pareto_plot.png`.
4. `kd_ablation.csv`.
5. `hw_estimate.csv` (BitOPs, memory, latency).
6. 발표 슬라이드.

## 9. 검토 요청 사항 (codex/gemini)
1. **A1W1 70% 달성 현실성** — 우리 레시피로 충분한가? 누락된 트릭은?
2. **Stage B1 (W32A1)이 정말 필요한가** — single-stage W1A1과 비교한 보고 사례.
3. **KD weight 0.8/T=4** — 1-bit 학습에 더 좋은 설정 있는가?
4. **LSQ vs PACT vs DSQ** for W2A2 — 어떤 것이 30일 내 안전한가?
5. **HW 추정 방법론** — BitOPs로 충분한지, 별도로 cycle/energy 모델 추천?
6. **발표 스토리텔링** — Pareto + KD + HW 셋 다 vs 둘로 줄이기. 어떤 조합이 가장 임팩트?
7. **잠재적 buggy 부분 미리 짚어주기** (RSign 위치, BN order, mixup-binary 충돌 등).
