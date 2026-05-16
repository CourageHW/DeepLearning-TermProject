# 보고서 & PPT 아키텍처 v1 (검토용)

## 0. 메타 정보

| 항목 | 값 |
|---|---|
| 과목 | 항공대(KAU) 3-1 딥러닝 텀프로젝트 |
| 팀 | Team 16 |
| 발표 시간 | 10 분 |
| 결과물 | (1) 보고서 `report.md` (PDF 변환 전제, ~10-15p) (2) 슬라이드 `slides.md` (Marp, ~12-13장) |
| 메시지 비중 | **BNN 정확도 7 : HW demo 3** |
| 데이터셋 | CIFAR-100 (Kaggle competition `26-deep-learning-course`) |

## 1. 핵심 사실 (확정)

- **Teacher**: ResNet18 (ImageNet pretrained → CIFAR-100 FT) = **82.44%**
- **Student**: A1W1ResNet18v2 (ReActNet-lite, 1-bit W + 1-bit A) = **68.72%**
- **Literature 비교**: ReActNet 69.4% / ReCU 69% / RBNN ~68% — SOTA 영역
- **메모리 압축**: 44 MB → 1.4 MB (≈ 32×)
- **C 추론 demo (AVX + POPCNT)**:
  - bit-exact 검증 통과 (PyTorch vs C max diff = 0)
  - Single-thread: Student 22 ms / Teacher 547 ms → **25× speedup**
  - 4-thread: 15 ms / 173 ms → **12× speedup**
- **Teacher empirical study** (Sheet1.csv 기반 7-step ablation):
  - Step 0 Resize Strategy (bilinear/bicubic/lanczos): 거의 동일 (~75.7%)
  - Step 1 Resize/Stem: 7×7 stride=1 + maxpool→Identity가 best (**81.82%**)
  - Step 2 Optimizer: AdamW (wd=1e-4) best (82.38%)
  - Step 3 Batch Size: 1024 + LR 1e-3 best
  - Step 4 Scheduler: CosineAnnealingLR best
  - Step 5 Activation: GELU best (82.46%) vs PReLU 81.4 / Swish 80.6
  - Step 6 Mixup Decay: step best (82.54%)
  - Step 7 Augmentation: **RandAugment num_ops=2, magnitude=10 → 83.12%**

## 2. 발표 슬라이드 구성 (Marp, 12장 + 백업)

10분 / 12장 → 슬라이드 당 약 50초.

| # | 슬라이드 제목 | 핵심 메시지 (1줄) | 시간 |
|---|---|---|---|
| 1 | Title | CIFAR-100 / 1-bit BNN + KD / Team 16 | 30s |
| 2 | Motivation & Goal | Edge/모바일 추론을 위한 32× 메모리 압축의 필요성 | 45s |
| 3 | Method Overview | ReActNet-lite (99% binary, 1% FP32) + KD | 45s |
| 4 | Student Architecture — `A1W1ResNet18v2` | RSign → BinConv → BN → RPReLU + **double-skip**, 핵심 수식 | 60s |
| 5 | Teacher Empirical Study (CSV) | 7-step ablation 표 + 75.7% → **83.12%** 추이 | 60s |
| 6 | Training Pipeline | Phase A 80ep / Phase B 220ep + KD (T=4, w=0.7) | 45s |
| 7 | Quantitative Results | Teacher 82.44% / **Student 68.72%** + Literature 비교표 | 60s |
| 8 | Design Choices & Discussion | EMA 비활성·FP32 stem/FC·AT 폭발 사례·AMP fp16 STE | 60s |
| 9 | HW Demo: C Inference Pipeline | `xnor_kernel.c` 구조 + bit-exact (max diff = 0) | 45s |
| 10 | Speedup Measurement | Single-thread 25× / Multi-thread 12× 막대그래프 | 45s |
| 11 | Limitations & Honest Discussion | W1A1 SOTA 격차 / T4 INT1 미지원 / naive C 비교의 한계 | 45s |
| 12 | Conclusion + Future Work | 핵심 성과 3줄 + 후속 작업 | 30s |
| B1+ | 백업 (Q&A 대응) | Mixup/CutMix, KD weight 폭발 사례, EMA shadow 결과 | — |

**총 합 ≈ 9분 30초** — Q&A 여유 30초.

## 3. 보고서 구성 (Markdown, ~12-14 페이지)

```
# CIFAR-100 1-bit Binary Neural Network with Knowledge Distillation
   — Team 16, KAU Deep Learning Term Project

## Abstract                                                       (0.5p)
## 1. Introduction                                                (1p)
   1.1 Motivation
   1.2 Contributions
## 2. Related Work                                                (1p)
   2.1 Binary Neural Networks (XNOR-Net, Bi-Real, ReActNet, ReCU)
   2.2 Knowledge Distillation (Hinton, FitNets, AT)
## 3. Teacher: Empirical Study on ResNet18                        (2.5p)
   3.1 Baseline & resize strategy
   3.2 Stem adaptation (3×3 vs 7×7 stride=1, maxpool→Identity)
   3.3 Optimizer & weight decay sweep
   3.4 Scheduler comparison
   3.5 Activation function (ReLU/PReLU/GELU/Swish)
   3.6 Mixup decay schedule
   3.7 Data augmentation (TrivialAugment vs RandAugment, magnitude sweep)
   3.8 Summary table + final config
## 4. Student: A1W1 ReActNet-lite                                 (2p)
   4.1 Binary primitives (BinaryActivationSTE, RSign, BinaryConv)
   4.2 Block design (double-skip, RPReLU, detached α)
   4.3 FP32 stem & classifier policy
   4.4 Numerical: AMP fp16 + grad clip
## 5. Training Pipeline                                           (1.5p)
   5.1 Phase A: Teacher fine-tuning
   5.2 Phase B: Student + Logit KD (T=4, w=0.7)
   5.3 Discussion: EMA disabled, AT exploded → disabled
   5.4 Hyperparameter table
## 6. Experiments                                                 (1.5p)
   6.1 Setup (CIFAR-100, Kaggle T4×2)
   6.2 Quantitative results & literature comparison
   6.3 Compression analysis (params, memory)
   6.4 Training cost
## 7. Hardware Inference Demo                                     (1p)
   7.1 C kernel design (XNOR + POPCNT, padding correction)
   7.2 Bit-exact verification
   7.3 Latency benchmark (single/multi-thread)
## 8. Limitations & Honest Discussion                             (0.5p)
## 9. Conclusion                                                  (0.5p)
## References                                                     (0.5p)
```

비중: §3 (Teacher empirical) + §4 (Student) + §5 + §6 = **약 7.5p (BNN 정확도)**, §7 + §8 = **약 1.5p (HW demo)**. 7:3 비율 일치.

## 4. 핵심 그림/표 목록

| ID | 내용 | 사용처 |
|---|---|---|
| Fig 1 | A1W1ResNet18v2 block diagram (double-skip) | 슬라이드 4 / 보고서 §4.2 |
| Fig 2 | Teacher empirical study line chart (Step 0~7 정확도 추이) | 슬라이드 5 / 보고서 §3.8 |
| Fig 3 | Training loss/acc 곡선 (Phase A, Phase B) | 보고서 §6.2 |
| Fig 4 | Memory footprint 비교 막대 (44 MB vs 1.4 MB) | 슬라이드 7 / 보고서 §6.3 |
| Fig 5 | C 추론 latency 막대 (single/multi-thread) | 슬라이드 10 / 보고서 §7.3 |
| Tab 1 | Teacher ablation 요약 (7 step, before/after acc) | 슬라이드 5 / 보고서 §3.8 |
| Tab 2 | Quantitative results + Literature 비교 | 슬라이드 7 / 보고서 §6.2 |
| Tab 3 | Hyperparameter table | 보고서 §5.4 |

## 5. 메시지 우선순위 (왜 7:3인가)

- **7 (BNN 정확도)**: 텀프로젝트 평가단이 "다른 팀과 동일 데이터셋으로 비교"하는 게 본질. 학생 정확도 68.72% (literature SOTA 영역)와 teacher 튜닝 디테일이 평가 핵심.
- **3 (HW demo)**: 차별화 포인트지만 모든 팀이 보유한 것은 아님 → 임팩트는 있으나 "정량 평가"는 어려움. 그래서 강하지만 짧게.

## 6. 발표 스토리라인 (10분 흐름)

1. "왜 1-bit BNN인가" (모바일/엣지, 32× 압축, 1분)
2. "어떻게 했나" (ReActNet-lite + 2-phase KD, 4분)
3. "결과" (Teacher empirical study → Student 68.7%, 3분)
4. "추가로 보여줄 것" (C 추론 demo, speedup 실측, 1.5분)
5. "솔직한 한계" + 결론 (0.5분)

## 7. 검토 요청 사항 (Codex / Gemini 양쪽 동일 질문)

다음 항목을 평가해주세요:

**(A) 슬라이드 구성 (12장)**
- 10분 발표에 슬라이드 12장은 적정한가? (50초/장)
- 비중 7:3 (정확도:HW) 이 슬라이드 분배(2-8번 정확도, 9-11번 HW)에 잘 반영되어 있는가?
- 빠진 핵심 메시지가 있나? (e.g. KD ablation, 학습 곡선, 실패 사례)
- 슬라이드 12장 중 cut 가능한 것 / merge 가능한 것이 있는가?

**(B) 보고서 섹션 (9 + abstract/refs)**
- §3 Teacher empirical에 2.5p 배정한 게 과한가 / 부족한가?
- 빠진 섹션이 있나? (e.g. Ablation of student design, Failure cases section)
- Related Work를 1p로 압축할 수 있는가?

**(C) 메시지 우선순위 (7:3)**
- 학부 텀프로젝트 발표 맥락에서 7:3 비중이 맞는가?
- 만약 평가단이 BNN에 익숙치 않다면 비중 조정 필요한가?

**(D) 기타**
- "솔직한 한계" 슬라이드(#11)를 넣는 게 점수에 유리한가 불리한가?
- 발표 도입(#1-2)에서 student 정확도(68.7%)를 미리 공개할까 vs 결과 슬라이드(#7)에서 reveal?

답변 형식: 각 항목당 한두 줄로 (1) 동의/이견 (2) 근거 또는 대안.
