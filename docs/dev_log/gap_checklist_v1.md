# 보고서·발표 자료 보충/수정 체크리스트 v1

**작성일**: 2026-05-14
**범위**: train.ipynb / CSV / report.md / slides.md cross-reference 결과
**총 39개 항목 발견 → 5 카테고리로 분류** (Critical / High / Medium / Low / 추가 실험)

---

## 🔴 A. CRITICAL — 사실관계 불일치, 즉시 수정 필요

| ID | 위치 | 문제 | 권장 조치 |
|---|---|---|---|
| **A1** | report.md §5.6 / slides.md Backup C, RandomErasing 행 | 보고서·슬라이드에 `scale = (0.02, 0.2)` 로 적혀 있으나 **실제 노트북 cell 5 는 `scale` 미지정 → torchvision v2 default `(0.02, 1/3) ≈ (0.02, 0.333)` 사용**. PDF 자료(`v2&bit_flip.pdf`) 의 값을 가져오면서 코드와 불일치 발생. | 둘 중 택1: (a) 보고서 표기를 `scale=default (0.02, 0.333)` 또는 단순히 `default scale` 로 수정 (안전), (b) 노트북에 `scale=(0.02, 0.2)` 명시 후 재학습 (재학습 비용 큼 → 비권장) |
| **A2** | report.md §6.1 normalize 부분 (없음) | 노트북 cell 2 의 `CIFAR100_MEAN/STD` **변수명이 misleading** — 실제 값은 **ImageNet 표준 mean/std** (`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`). CIFAR-100 의 실제 통계는 `[0.5071, 0.4865, 0.4409] / [0.2673, 0.2564, 0.2762]` 와 다름. 이는 의도된 선택 (pretrained ResNet18 호환) 이지만 **보고서에 미설명**. | §6.1 (Setup) 또는 §5.5 hyperparam table 에 한 줄 추가: "Normalize stats: ImageNet (pretrained backbone 호환). CIFAR-100 native stats 미사용 — pretrained conv1 의 activation 분포 가 ImageNet 기준으로 학습되어 있어 호환성 우선." |

---

## 🟠 B. HIGH — 논리적 비약 / 핵심 디테일 누락

| ID | 위치 | 문제 | 권장 조치 |
|---|---|---|---|
| **B1** | report.md §5.2 / slides.md slide 6 | **KD warm-up 시작 epoch 명시 안 됨**. 실제 노트북 cell 11 line 788: `distill_start_ep=student_warmup_epochs=15` → student 학습 epoch 0–14 는 KD 없이 CE 만, epoch 15 부터 KD 활성. | §5.2 에 한 줄 추가: "KD 는 warm-up 15 epoch 이후 (`distill_start_ep=15`) 부터 활성화하며, 그 전 epoch 은 CE-only 로 student stem/binary blocks 의 초기 적응 단계." |
| **B2** | report.md §4 / §5 어디에도 없음 | **Binary weight clamping** (clamp(-1, 1)) 이 노트북 cell 6 line 272-276 + cell 11 line 789 (`clip_binary=True`) 로 매 step 후 적용되는데 보고서 미언급. 이는 ReActNet / Bi-Real 표준 trick — **학습 안정성 의 핵심 요소**. | §4.4 (Numerical considerations) 또는 §5.2 에 한 줄: "매 optimizer step 후 binary conv 의 latent weight 를 [-1, +1] 로 clamp 하여 sign STE 의 dynamic range 안정화." |
| **B3** | report.md §5.5 Hyperparameter table | **AdamW 의 weight decay group split** 누락. 노트북 cell 7 `make_optimizer` 가 1-D params (BN γ/β, PReLU/RPReLU γ/β, RSign θ, bias) 에 weight_decay=0 적용. 보고서는 "wd=1e-4 (teacher) / wd=0 (student)" 만 명시. | §5.5 표에 footnote 추가: "AdamW weight decay 는 4-D conv/linear weight 에만 적용, 1-D parameter (BN, PReLU, RPReLU, RSign threshold, bias) 는 항상 no-decay (별도 param group)." |
| **B4** | report.md §5.2 / slides.md backup B | **Student init from teacher 의 정확한 범위 미명시**. 노트북 cell 9 `init_student_from_teacher` 는 **shape 일치 텐서만** 복사. teacher conv1 (7×7, stride 1) 과 student conv1 (3×3, stride 1) 은 shape 불일치 → stem 은 random init. binary block 의 conv1/conv2/bn1/bn2 만 복사됨. | §5.2 update: "teacher 의 state-dict 중 shape 가 일치하는 텐서만 student 로 복사 (binary block 의 conv·BN). Student stem conv (3×3) 는 teacher stem (7×7) 과 shape 불일치로 random init 유지." |
| **B5** | report.md §6.2 / §3.9 / slides.md slide 7 | **Teacher 82.44 % 가 EMA shadow 결과인지 raw 결과인지 미명시**. 노트북 cell 11 line 712-713: `target = shadow if (use_ema and epoch >= warmup_eps) else base` 로 EMA shadow 가 best ckpt 로 저장됨. 즉 **best_teacher.pth = EMA shadow weights**. | §3.9 또는 §6.2 한 줄 보강: "Teacher 82.44 % 는 EMA shadow (decay = 0.999) evaluation 의 best epoch. live model 의 raw val 은 약 1 %p 낮음 (typical EMA gap)." 또는 §5.5 표에 "Best checkpoint criterion: EMA val (teacher) / raw val (student)" footnote. |
| **B6** | report.md §3 ablation / CSV | **CSV 컬럼 "SWA enable: TRUE" 와 실제 노트북의 EMA 사용 불일치**. CSV 모든 ablation 행이 SWA=TRUE 로 표시되어 있으나, 노트북 코드에는 SWA 없음 — `torch.optim.swa_utils` 미사용, EMA (cell 10 의 custom class) 만. CSV 가 작성된 ablation 시점과 현재 노트북이 다를 가능성. | 두 가지 중 택1: (a) §3 도입부에 한 줄 — "본 ablation 의 'SWA enable' 컬럼은 EMA shadow (decay = 0.999) 사용을 의미하며, 실제 코드에서는 torch SWA API 가 아닌 custom EMA class 로 구현." (b) CSV 의 SWA 컬럼을 EMA 로 rename. |

---

## 🟡 C. MEDIUM — 디테일 보강, 발표 임팩트 강화

| ID | 위치 | 문제 | 권장 조치 |
|---|---|---|---|
| **C1** | report.md §6.4 / slides.md slide 7 | **BitOPs (binary operations) 분석 부재**. BNN 표준 metric. 32× 메모리만 있고 연산량 비교 누락. 발표에서 "32× 메모리"는 강한데 "연산량은?" 질문 가능. | §6.4 에 표 추가: ResNet18 FP32 ≈ 1.82 GFLOPs / Binary FLOPs ≈ (FP32 stem + FC) + (binary blocks 의 BitOPs / 64). 또는 "Binary GEMM 은 K_bits × M × N 의 popcount → FP32 multiply 대비 ~ 1/64 cost (uint64 packing)" 한 줄. |
| **C2** | report.md §6.2 / §8 / slides.md slide 7 | **단일 run, 단일 seed 결과 명시 안 됨**. variance 측정 없음. 학부 수준에선 OK 이나 정직성 +1. | §8 Limitations 또는 §6.1 에 한 줄: "본 결과는 seed=42 단일 run (시간 제약). seed sweep 미실시 — 학습 variance 는 ±0.3 %p 영역 예상 (BNN literature 일반값)." |
| **C3** | report.md §6 / slides.md slide 7 | **Kaggle leaderboard test accuracy 미보고**. §6.2 의 68.72 % 는 internal val (5000 samples). Kaggle submission test accuracy (10000 samples) 가 있다면 별도 보고 가능. | (a) Kaggle 점수가 있으면 §6.2 에 "Kaggle leaderboard public test: XX.XX %" 추가, (b) 없으면 미언급 OK. |
| **C4** | report.md §6.3 Literature comparison | **ImageNet vs CIFAR-100 비교의 한계 명시는 있으나 발표 슬라이드에는 미명시**. slide 7 표에 ImageNet 결과와 CIFAR-100 결과가 한 표에 나열되어 청중 혼란 가능. | slide 7 표에 "*" 또는 footnote: "ImageNet 결과는 참고용 (직접 비교 불공정, 도메인 다름). CIFAR-100 직접 비교 baseline 은 RBNN ~ 67.1 %." |
| **C5** | report.md §6.4 / slides.md slide 7 | **Param count 11.22 M 동일 → 미스리딩 위험**. teacher = student = 11.22 M, 하지만 student 의 99 % 가 1-bit. 청중이 "왜 1-bit 인데 같은 11.22 M?" 의문 가능. | slide 7 표 컬럼 추가 (`bit-width` 또는 `Storage`) 또는 footnote: "Params 동일 (구조 동일), 차이는 **bit-width** (FP32 32-bit vs binary 1-bit) → Storage 32× 압축". |
| **C6** | report.md §3.9 / slides.md slide 5 | **Backbone choice rationale 부재**. 왜 ResNet18 인가? CSV 마지막에 ResNet34/50 비교 계획되었으나 미실시. | §3 도입부 또는 §3.9 에 한 줄: "Backbone 으로 ResNet18 을 선택한 이유 — (1) BNN literature 표준 (XNOR-Net / Bi-Real / ReActNet 모두 ResNet18 baseline), (2) Kaggle T4 단일 세션 12h budget 안에서 student 220 epoch 학습 가능 한 최소 크기, (3) ResNet34/50 ablation 은 시간 제약상 미실시 (Future Work)." |
| **C7** | report.md §5.6 | **Mixup/CutMix 적용 확률 (각 50 %) 가 §5.6 표 외 본문에 부각 안 됨**. 노트북 cell 6 line 321: `if random.random() < 0.5:`. 보고서 §5.6 끝 단락 "각 batch 마다 50 % 확률" 한 줄만 있음. | §5.6 표 의 Mixup / CutMix 행 "역할" 칼럼에 "(batch 별 50 % 적용)" 명시. |
| **C8** | report.md §5.3 / slides.md slide 8 | **AT loss weight 1000 → 10 ramp 사례의 출처 불분명**. 노트북 코드 (cell 6 line 343-349) 에는 ramp-up 만 있고, "1000 → 10" 의 초기 시도 흔적은 코드에 없음. PROJECT.md §3.2 에 "AT weight 10 (낮춤 — 1000에서 폭발했음)" 언급. 발표 시 "정말 1000 으로 해봤어요?" 질문 가능. | §5.3 wording 약간 보강: "초기 AT weight = 1000 으로 시도 시 (노트북 별도 실험), epoch 0 에서 학습 발산. weight 10 + 10-epoch linear ramp-up 으로 수정 (현재 코드 cell 6 line 343-349), 그러나 KD switch 시점 BN running stats 불안정 → 최종 AT disable." — 별도 실험이라는 점 명시. |

---

## 🟢 D. LOW — 선택적 보강 (시간 남으면)

| ID | 위치 | 문제 | 권장 조치 |
|---|---|---|---|
| **D1** | report.md §6 | **TTA (test-time augmentation) 미사용 명시 없음**. cell 12 는 단순 `argmax`, val_tf 적용 (normalize only). | §6.1 또는 §7.2 에 한 줄: "Test inference: single-pass, no TTA — 발표 demo 의 fair comparison 일관성." |
| **D2** | report.md §6.1 | **Val split = random (non-stratified)** 명시 없음. cell 5 line 220: `random_split(...)`. 100 class × 50 samples/class 의 validation 이지만 실제 distribution variance 약간 있을 수 있음. | §6.1 에 한 줄: "Val split = random 90:10 (seed=42), non-stratified — class 별 약 50 samples/class 평균." |
| **D3** | report.md §3.2 | **64×64 / 128×128 resize 결과 (CSV step 1) 미언급**. CSV 에 64/maxpool=82.70 %, 128/maxpool=82.76 % 있으나 보고서는 32×32 의 4 가지만. | §3.2 끝에 한 줄: "Higher-resolution (64×64, 128×128) 변형도 ablation 했으나 (82.70 % / 82.76 %), 학습 시간 ×1.5 ~ 4 증가로 비용 대비 효과 작음 → 32×32 native 채택." |
| **D4** | report.md §4.1 / slides.md slide 4 | **학습 시 binary conv 는 popcount 사용 안 함, F.conv2d(sign(w), sign(x)) * α 형식**. 발표의 핵심 수식 `y = α·(2·popcount(¬(A⊕W))−K_bits)` 는 **inference (C kernel) 의 식**. 학습 시는 등가의 conv 연산. | §4.1 에 한 줄 또는 footnote: "학습 시 cell 9 line 510: `F.conv2d(x, sign(w)) · α` 로 표현 — popcount 식과 수학적 동등, GPU autograd 호환성 위해 일반 conv 사용. **Inference (C kernel) 에서만 packed bit + popcount 변환**." |
| **D5** | report.md §5.2 | **CE label_smoothing 0.05 (teacher) / 0 (student) 차이 명시 약함**. §5.6 마지막에 다루지만 §5.2 (Phase B 본문) 에서는 미언급. | §5.2 한 줄: "Student 는 KD soft label 이 label smoothing 역할을 하므로 CE 의 `label_smoothing = 0` (teacher 는 0.05)." |
| **D6** | slides.md slide 4 | **수식 변수의 정의가 슬라이드 안에 없음**. $A_{i,j}, W_{c_o}, K_\text{bits}$ 의 의미 (input patch, weight kernel, bit count). 보고서 §4.1 에는 있음. | slide 4 표 아래 한 줄 (small font): "$A_{i,j}$: input patch (binary), $W_{c_o}$: output-channel weight (binary), $K_{bits}$: total bit count." |
| **D7** | report.md §6.5 | **Phase B 80 분의 GPU 활용률 / 효율 분석 없음**. fp16 + binary forward 의 효율 (TensorCore 활용) 언급 가능. | minor — 시간 남으면 §6.5 에 한 줄: "fp16 autocast 로 binary conv 가 TensorCore 활용, FP32 대비 ~1.6 × 학습 가속." |

---

## 🔵 E. 추가 실험 후보 (Future Work 또는 발표 직후 수행)

각 실험의 **ROI (effort vs presentation impact)** 추정.

| ID | 실험 | 예상 학습 시간 | 보고서 임팩트 | 권장 우선순위 |
|---|---|---|---|---|
| **E1** | **KD on/off ablation** — student 1회 (logit KD weight=0, ~80 min) | +80 min | 매우 큼 (§5/§6 KD 효과 정량화, 발표 narrative 강화) | ⭐⭐⭐ |
| **E2** | **Double-skip on/off ablation** — student 1회 (use_double_skip=False) | +80 min | 큼 (§4.2 design rationale 정량화) | ⭐⭐ |
| **E3** | **RPReLU vs PReLU ablation** — student 1회 (use_rprelu=False) | +80 min | 중 (§4.1 design rationale) | ⭐⭐ |
| **E4** | **Class-wise accuracy / confusion matrix** — `inference_c.py --compare` 실측 캡처 | <10 min (학습 X) | 중 (§6 정성 분석, 백업 슬라이드) | ⭐⭐⭐ (학습 불필요) |
| **E5** | **BitOPs 계산** — pure 분석, 학습 X | <30 min | 중 (§6.4 보완) | ⭐⭐⭐ |
| **E6** | **Multi-seed variance** — student 3-seed run | +240 min | 작음 (학부 수준 over-engineering) | ⭐ |
| **E7** | **Mixup α sweep** — Phase B 의 α=0.2, 0.4, 0.8 비교 | +240 min | 작음 (§5.5 hyperparam search) | ⭐ |
| **E8** | **KD temperature T sweep** — T=2, 4, 8 비교 | +240 min | 중 (§5.5 + §6.3 KD rationale) | ⭐⭐ |
| **E9** | **Bit-width Pareto** — W8A8 LSQ, W4A4 LSQ 학습 후 비교 | +5-10 시간 | 매우 큼 (별도 발표 chapter) | ⭐ (시간 부족) |

**즉시 가능 (학습 불필요) — 우선순위 최상**:
- E4 (class-wise / confusion) — `inference_c.py --compare` 실행만 하면 됨
- E5 (BitOPs 분석) — 종이 계산

**1회 추가 학습으로 큰 임팩트 — 시간 있으면 강력 권고**:
- E1 (KD on/off)
- E2 (double-skip on/off)

---

## 📋 적용 가이드 — 발표까지 시간별 권장

### 시나리오 X: 발표 직전 (30분 이내)
- **A1, A2 즉시 수정** (사실관계 오류) → 보고서 재 컴파일
- **B1, B2, B3, B4, B5, B6 적용** (모두 텍스트 한두 줄 보강)
- **C1 (BitOPs)** 종이 계산해 §6.4 추가
- **E4, E5** 도 시간 되면

### 시나리오 Y: 1–2 시간
- 위 + **C2–C8** 모두 적용
- **E4 (class-wise/confusion)** `inference_c.py --compare` 실행 → 캡처
- **E5 (BitOPs)** 종이 계산 표 추가

### 시나리오 Z: 반나절 ~ 1 일
- 위 + **D1–D7** 모두 적용
- **E1 (KD on/off ablation)** student 1회 추가 학습 → §5/§6 정량화
- 또는 **E2 (double-skip on/off)**

### 시나리오 W: 며칠+
- 위 + **E8 (KD T sweep)** + **E9 (Bit-width Pareto)** — 별도 보고서·발표 가능

---

## 🤝 두 모델 cross-check 제안

본 진단표는 단독 작성. 사용자 memory `feedback_review_workflow.md` 기준 큰 작업이므로 Codex / Gemini cross-check 권장.

검토 요청 사항:
1. A/B 카테고리에 빠진 critical 항목이 있는가?
2. E 카테고리 우선순위 — 발표 직전 시점에서 어떤 실험을 강력히 추천하나?
3. 본 진단표의 항목 중 학부 텀프로젝트 맥락에서 over-engineering 인 것은?
