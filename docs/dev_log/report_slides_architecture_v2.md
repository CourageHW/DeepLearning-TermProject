# 보고서 & PPT 아키텍처 v2 (Codex/Gemini cross-check 반영 후 확정)

## 0. 메타 정보

| 항목 | 값 |
|---|---|
| 과목 | 항공대(KAU) 3-1 딥러닝 텀프로젝트 |
| 팀 | Team 16 |
| 발표 시간 | 10 분 |
| 결과물 | (1) `report.md` (PDF 변환 전제, ~12-14p) (2) `slides.md` (Marp, **11장 + 백업**) |
| 메시지 비중 | **BNN 정확도 7 : HW demo 3** |
| 데이터셋 | CIFAR-100 (Kaggle competition `26-deep-learning-course`) |

## 1. v1 → v2 변경 요약

1. 슬라이드 12장 → **11장** (#9 C Pipeline + #10 Speedup → merge)
2. 슬라이드 3에 "왜 ReActNet-lite인가" 한 줄 추가 (BNN 미숙 평가단 대응)
3. 슬라이드 8(Design Choices)에 **AT weight 폭발(1000→10)** 사례 정직하게 포함
4. 보고서 Related Work **1p → 0.7p**로 압축
5. 보고서 §8 Limitations에 **"Student/KD ablation 미실시"** 항목 추가
6. 학습 곡선은 **제외** (데이터 있으나 사용자 결정으로 보고서·슬라이드 모두 미수록)

## 2. 핵심 사실 (확정)

- **Teacher**: ResNet18 (ImageNet pretrained → CIFAR-100 FT) = **82.44%**
- **Student**: A1W1ResNet18v2 (ReActNet-lite, 1-bit W + 1-bit A) = **68.72%**
- **Literature 비교**: ReActNet 69.4% / ReCU 69% / RBNN ~68%
- **메모리 압축**: 44 MB → 1.4 MB (≈ 32×)
- **C 추론 demo (AVX + POPCNT)**:
  - bit-exact 검증 (PyTorch vs C max diff = 0)
  - Single-thread: Student 22 ms / Teacher 547 ms → **25× speedup**
  - 4-thread: 15 ms / 173 ms → **12× speedup**
- **Teacher empirical study** (CSV 7-step ablation):
  - Step 1 Stem(7×7 stride=1 + maxpool→Identity): **81.82%**
  - Step 2 Optimizer (AdamW wd=1e-4): 82.38%
  - Step 5 Activation (GELU): 82.46%
  - Step 6 Mixup Decay (step): 82.54%
  - Step 7 Augmentation (RandAugment num_ops=2, magnitude=10): **83.12%**
  - Phase A에 사용한 최종 setting 은 epoch 80, batch 1024, AdamW wd 1e-4, CosineAnnealingLR, GELU(=teacher 변경 없음, ResNet18 기본 ReLU 유지)

> 📌 **주의**: CSV는 ResNet18 backbone ablation을 150 epoch로 돌린 사전 실험. 노트북의 Phase A teacher는 80 epoch + ImageNet pretrained → CIFAR FT 흐름이라 다소 차이가 있음. 보고서에서는 "사전 ablation으로 hyperparam 후보를 좁힌 뒤 80 epoch 단축 FT로 옮겼다"고 서술.

## 3. 발표 슬라이드 구성 (Marp, 11장 + 백업)

총 10분 / 11장 → 평균 약 55초/장.

| # | 슬라이드 제목 | 핵심 메시지 | 시간 |
|---|---|---|---|
| 1 | Title | CIFAR-100 / 1-bit BNN + KD / Team 16 / 발표자명 | 30s |
| 2 | Motivation & Goal | Edge/모바일 추론에서의 1-bit BNN 필요성 + 32× 압축 목표 | 50s |
| 3 | Method Overview | ReActNet-lite (99% binary, 1% FP32) + **"왜 ReActNet-lite? RSign+RPReLU+double-skip로 학습 안정화"** + Logit KD | 60s |
| 4 | Student Architecture — `A1W1ResNet18v2` | RSign → BinConv → BN → RPReLU + double-skip 도식 + 핵심 수식 `y = α·(2·popcount(¬(A⊕W)) − K_bits)` | 70s |
| 5 | Teacher Empirical Study (CSV) | 7-step ablation 표 + 정확도 추이 75.7% → **83.12%** | 70s |
| 6 | Training Pipeline | Phase A 80ep / Phase B 220ep + KD (T=4, w=0.7), 핵심 hyperparam | 55s |
| 7 | Quantitative Results | Teacher 82.44% / **Student 68.72%** + Literature 비교표 (ReActNet/ReCU/RBNN) + 메모리 32× | 70s |
| 8 | Design Choices & Discussion | Student EMA 비활성 / FP32 stem+FC / AMP fp16 STE / **AT weight 1000→10 폭발 사례** | 60s |
| 9 | C Inference: Design + Speedup (**merge**) | xnor_kernel.c 구조 + bit-exact (max diff=0) + single-thread 25× + multi-thread 12× | 75s |
| 10 | Limitations & Honest Discussion | W1A1 SOTA(~71%) 격차 / T4 INT1 미지원 / Student·KD ablation 미실시 / naive C 비교 한계 | 50s |
| 11 | Conclusion + Future Work | 핵심 성과 3줄 + KD ablation·SAM·bit-pareto 등 후속 작업 | 30s |
| B1+ | 백업 (Q&A 대응) | Mixup/CutMix, AT 폭발 상세, EMA shadow 검증, ReActNet 원논문 차이 | — |

**합계 ≈ 8분 40초 + 발표 흐름 buffer 약 1분** = 9분 40초.

## 4. 보고서 구성 (Markdown, ~12-14p, v2)

```
# CIFAR-100 1-bit Binary Neural Network with Knowledge Distillation
   — Team 16, KAU Deep Learning Term Project

## Abstract                                                       (0.5p)
## 1. Introduction                                                (1p)
   1.1 Motivation
   1.2 Contributions

## 2. Related Work                                                (0.7p)  ← 1p → 0.7p
   2.1 Binary Neural Networks (XNOR-Net, Bi-Real, ReActNet, ReCU — 각 1줄)
   2.2 Knowledge Distillation (Hinton 2015, AT — 각 1줄)

## 3. Teacher: Empirical Study on ResNet18                        (2.5p)
   3.1 Baseline & resize strategy (Step 0)
   3.2 Stem adaptation (Step 1): 7×7 stride=1, maxpool→Identity
   3.3 Optimizer & weight decay sweep (Step 2)
   3.4 Batch & scheduler (Step 3-4)
   3.5 Activation function (Step 5)
   3.6 Mixup decay (Step 6)
   3.7 Augmentation (Step 7): RandAugment magnitude sweep
   3.8 Summary table + Phase A 최종 config

## 4. Student: A1W1 ReActNet-lite                                 (2p)
   4.1 Binary primitives (BinaryActivationSTE, RSign, BinaryConv)
   4.2 Block design (double-skip, RPReLU, detached α)
   4.3 FP32 stem & classifier policy + 이론적 근거
   4.4 Numerical: AMP fp16 STE + grad clip

## 5. Training Pipeline                                           (1.5p)
   5.1 Phase A: Teacher fine-tuning
   5.2 Phase B: Student + Logit KD (T=4, w=0.7)
   5.3 Failure case: AT weight 1000 → 10 ramp-up
   5.4 Discussion: EMA disabled (binary weight avg 무의미)
   5.5 Hyperparameter table

## 6. Experiments                                                 (1.5p)
   6.1 Setup (CIFAR-100, Kaggle T4×2)
   6.2 Quantitative results (Teacher 82.44% / Student 68.72%)
   6.3 Literature comparison
   6.4 Compression analysis (params, memory)
   6.5 Training cost

## 7. Hardware Inference Demo                                     (1.5p)
   7.1 C kernel design (XNOR + POPCNT, padding correction)
   7.2 Bit-exact verification
   7.3 Latency benchmark (single/multi-thread)
   7.4 Threading 관찰 (25× → 12×의 의미)

## 8. Limitations & Honest Discussion                             (0.7p)
   - W1A1 SOTA(~71%) 와의 1-2%p 격차
   - T4에서 INT1 native 가속 미지원 → CPU demo로 우회
   - C 구현이 naive (BLAS·VPOPCNTDQ 미사용)
   - **Student/KD ablation 미실시** (시간 제약)

## 9. Conclusion                                                  (0.5p)
## References                                                     (0.5p)
```

분량 합산: 0.5 + 1 + 0.7 + 2.5 + 2 + 1.5 + 1.5 + 1.5 + 0.7 + 0.5 + 0.5 ≈ **12.9p**.

비중: §3+§4+§5+§6 = **7.5p (BNN 정확도)**, §7+§8 = **2.2p (HW demo + 한계)**. 약 **7.5 : 2.2 ≈ 7.7 : 2.3** → 7:3 비중 부합.

## 5. 핵심 그림/표 목록

| ID | 내용 | 사용처 | 데이터 소스 |
|---|---|---|---|
| Fig 1 | A1W1ResNet18v2 block diagram (double-skip + RPReLU) | 슬라이드 4 / 보고서 §4.2 | 직접 작도 (ASCII 또는 Mermaid) |
| Fig 2 | Teacher empirical study 정확도 추이 (Step 0~7) | 슬라이드 5 / 보고서 §3.8 | CSV 직접 |
| Fig 3 | Memory footprint 비교 막대 (44 MB vs 1.4 MB) | 슬라이드 7 / 보고서 §6.4 | PROJECT.md |
| Fig 4 | C 추론 latency 막대 (1T / 4T) | 슬라이드 9 / 보고서 §7.3 | PROJECT.md / inference_c.py 실측 |
| Tab 1 | Teacher ablation 요약 (7 step, before/after acc) | 슬라이드 5 / 보고서 §3.8 | CSV |
| Tab 2 | Quantitative results + Literature 비교 | 슬라이드 7 / 보고서 §6.3 | 문헌 + PROJECT.md |
| Tab 3 | Hyperparameter table (teacher + student) | 보고서 §5.5 | train.ipynb Cell 2 |

> **참고**: 학습 곡선(loss/acc per epoch)은 사용자 결정으로 보고서·슬라이드 모두 미수록.

## 6. 발표 스토리라인 (10분 흐름)

1. **0:00-1:00** "왜 1-bit BNN인가" (slide 1-2) — 모바일/엣지, 32× 압축 동기
2. **1:00-4:00** "어떻게 했나" (slide 3-6) — ReActNet-lite + double-skip + KD pipeline
3. **4:00-7:00** "정량 결과" (slide 7-8) — Teacher 82.4% / Student 68.7% + design choices/실패 사례
4. **7:00-8:30** "추가 차별화 — C 추론 demo" (slide 9) — bit-exact + 25×/12× speedup
5. **8:30-9:30** "솔직한 한계 + 결론" (slide 10-11)

## 7. [3] 단계 작성 순서 (다음 작업)

1. `report.md` 작성 (Markdown 12-14p, 위 §4 구조대로)
2. `slides.md` 작성 (Marp 11장, 위 §3 구조대로)
3. 그림·표를 ASCII 또는 Mermaid로 임베드 (Marp/Markdown 호환)
4. 작성 후 자체 검토:
   - 사실관계 (PROJECT.md / CSV / 노트북과 대조)
   - 비중 7:3 유지 여부
   - 시간 분배 (10분 추정)
   - 학부 평가단 입장에서의 가독성

## 8. 검토 단계 ledger (CLAUDE.md §5 준수)

- **Risk markers touched**: schema(보고서/슬라이드 구조), 새로운 아카이브 산출물 — minor
- **External call used**: yes — `codex-consultant` + `gemini-consultant` 병렬 호출 (v1 → v2 cross-check)
- **Reason for call**: 사용자 명시 요청 + memory `feedback_review_workflow.md` 룰 (큰 작업 = 플랜 단계 cross-check)
