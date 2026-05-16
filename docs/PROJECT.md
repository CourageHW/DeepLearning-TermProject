# CIFAR-100 Term Project — 1-bit BNN + C Inference Demo

딥러닝 텀프로젝트. **A1W1 ReActNet-lite ResNet18**로 CIFAR-100 분류 + AVX CPU 기반 binary inference demo.

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| **데이터셋** | CIFAR-100 (Kaggle competition `26-deep-learning-course`) |
| **메인 모델** | `A1W1ResNet18v2` — ReActNet-lite, **1-bit weights + 1-bit activations** |
| **Teacher (KD source)** | ResNet18 (ImageNet pretrained → CIFAR-100 fine-tune, 80 epochs) |
| **목표** | Top-1 ≥ 70% (literature SOTA 영역) |
| **현재 결과** | Teacher 82.44%, Student (1차) **68.72%** ← 진행 중 (A1+A2 적용 재학습) |
| **차별화 포인트** | (a) 99% binary inference, (b) AVX CPU 실측 speedup demo |

**핵심 트레이드오프 결정**
- Stem conv1 + 마지막 FC만 FP32 유지 (BNN 분야 표준; 200KB만 차지하면서 +5-10% acc 보존)
- Knowledge Distillation 사용 (T=4, w=0.7, **logit KL + Attention Transfer**)
- Student EMA 비활성 (binary 가중치 평균은 sign() 부수므로 BNN literature도 안 씀)
- AMP fp16 binary forward (ReActNet 원 구현도 mixed precision)

---

## 2. 디렉토리 구조

```
Termp/
├── train.ipynb              ★ 메인 학습 노트북 (Kaggle 단일 세션)
├── best_teacher.pth         ★ FP32 teacher 가중치 (82.44%)
├── best_student.pth         (학습 완료 후 생성)
├── PROJECT.md               ← 이 파일
├── README.md                구 README (CNN train/test/dump 도구, 미사용)
│
├── .agents/                 플랜·검토·아카이브 모음
│   ├── plan_v1.md           초기 플랜 (A1W1 단독, KD 포함)
│   ├── plan_v2.md           Codex 리뷰 반영 후 (RPReLU, double-skip, grad clip 추가)
│   ├── plan_v4.md           1주일 일정 — Track A (정확도) + Track B (demo)
│   ├── plan_avx_demo.md     AVX CPU demo 설계 + Codex 리뷰
│   ├── notebook_excerpt.py  Codex 리뷰용 노트북 핵심 코드 발췌
│   ├── code_v2_for_review.py    Codex 코드 리뷰용 발췌
│   ├── train.backup.ipynb   Cleanup 전 노트북 백업
│   ├── reviews/
│   │   ├── codex_review.md  Codex 1차 검토 (LSQ 버그, EMA·자명 트릭 등)
│   │   ├── codex_review.json
│   │   └── gemini_review.json   (cancelled)
│   └── avx_demo/            ★ C 추론 파이프라인 (별도 섹션)
│       ├── xnor_kernel.c
│       ├── build.sh
│       ├── xnor_kernel.so   (빌드 산출물)
│       ├── export_weights.py
│       ├── models.py
│       ├── inference_c.py
│       ├── demo_inference.py
│       └── validate_and_bench.py
│
├── conda_env.yaml           구 conda 환경 (PyTorch 1.12)
├── README.md, dump.py, ...  실습실 원본 (미사용)
└── bnn/                     이전 실습 잔여 (미사용)
```

---

## 3. 핵심 파일 설명

### 3.1 `train.ipynb` (13 cells)

| Cell | 역할 |
|---|---|
| 0 | Imports |
| 1 | Kaggle 경로, device 설정 |
| **2** | **Config — 모든 하이퍼파라미터 여기만 수정** |
| 3 | `set_seed` (cudnn deterministic + benchmark=False) |
| 4 | `CIFAR100Dataset` (RAM 캐시, ThreadPool 로드) |
| 5 | `train_val_loader`, `test_loader`, GPU transforms |
| 6 | `mixup`, `cutmix`, `train_epoch`, `evaluate`, **Feature KD 헬퍼** |
| 7 | `build_teacher` (ResNet18 pretrained), `make_optimizer` |
| 8 | Binary primitives (`BinaryActivationSTE`, `RSignActivation`, `ShortcutDownsample`) |
| 9 | `RPReLU`, `A1W1BinaryConv2dV2`, `A1W1BasicBlockV2`, **`A1W1ResNet18v2`**, `init_student_from_teacher` |
| 10 | `EMA`, `DistillationWrapper` |
| **11** | **`main_pipeline()` — Phase A → Phase B 자동 실행** |
| 12 | Submission CSV 생성 |

### 3.2 Config 주요 파라미터 (Cell 2)

```python
# Phase A — Teacher quick FT (80ep, ~1.5h)
teacher_epochs       = 80
teacher_lr           = 1e-3

# Phase B — A1W1 student + KD (300ep, ~3h with fp16 binary)
student_epochs       = 220     # 노트북 현행 값 (2026-05-14 동기화, 이전 plan 300)
student_lr           = 5e-4
student_grad_clip    = 5.0
student_kd_T         = 4.0
student_kd_weight    = 0.7      # logit KL weight
student_feature_kd_w = 10.0     # AT loss weight (낮춤 — 1000에서 폭발했음)
student_use_rprelu   = True
student_use_double_skip = True

# Skip Phase A if best_teacher.pth exists
skip_phase_a_if_ckpt = True
teacher_ckpt_path    = "best_teacher.pth"
```

### 3.3 모델 구조 — `A1W1ResNet18v2`

```
Input (3, 32, 32)
  → Conv1 (FP32, 3→64, 3×3)
  → BN1 → RPReLU(act_in)
  → Layer1 (2 binary blocks, 64→64)
  → Layer2 (2 binary blocks, 64→128, stride 2 첫 블록)
  → Layer3 (2 binary blocks, 128→256, stride 2)
  → Layer4 (2 binary blocks, 256→512, stride 2)
  → BN_out → AvgPool(1×1)
  → FC (FP32, 512→100)
```

각 binary block (double-skip 적용):
```
x ─→ BN → RSign → BinConv → BN → RPReLU ─┬─→ +res_a ─→ ...
                                          │
                                          └→ BN → RSign → BinConv → BN → RPReLU ─→ +res_b ─→ out
```

핵심 수식 (binary conv):
```
y[c_o, i, j] = α_{c_o} × (2 × popcount(¬(A_{ij} ⊕ W_{c_o})) - K_bits)
                where α_{c_o} = E[|W_{c_o}|], detached
```

---

## 4. `.agents/avx_demo/` — C 추론 파이프라인

학습된 모델을 **C 코드만으로** 추론하는 demo. T4 GPU에 INT1 native 가속이 없으므로 **CPU AVX + POPCNT**로 실측 speedup 확보.

```
avx_demo/
├── xnor_kernel.c           ★ C primitives (350 lines)
├── build.sh                컴파일 스크립트
├── xnor_kernel.so          빌드 산출물
├── export_weights.py       .pth → .bnn 변환 (binary는 packed bits)
├── models.py               PyTorch 참조 모델 (검증용)
├── inference_c.py          ★ Python orchestrator + 3가지 모드
├── demo_inference.py       (legacy) PyTorch CPU 단독 demo
└── validate_and_bench.py   GEMM 단독 벤치마크
```

### 4.1 `xnor_kernel.c` — C primitives

| 함수 | 역할 | OpenMP |
|---|---|---|
| `xnor_gemm` | Binary 행렬곱 (XNOR + scalar POPCNT, N-blocked) | ✅ M축 |
| `xnor_gemm_padded` | K_bits가 64의 배수 아닐 때 trailing pad 보정 | — |
| `pack_sign_to_bits` | float → ±1 → packed uint64 | — |
| `fp32_conv2d` | FP32 컨볼루션 (stem용, teacher용) | ✅ C_out축 |
| `batch_norm_2d` | 추론용 BN (running mean/var) | ✅ 채널축 |
| `rsign` | 채널별 학습 threshold sign | ✅ 채널축 |
| `rprelu` | PReLU(x - γ) + β | ✅ 채널축 |
| `binary_conv2d` | 1-bit conv + im2col + padding correction | ✅ 출력위치축 |
| `shortcut_downsample` | AvgPool + 채널 zero-pad | — |
| `adaptive_avgpool_1` | 글로벌 평균 풀링 | — |
| `linear_fp32` | FP32 linear (classifier) | — |
| `add_inplace`, `relu_inplace` | 잡다 헬퍼 | — |
| `cpu_set_num_threads(n)` | Runtime thread count 설정 | — |
| `cpu_get_max_threads()` | 사용 가능 코어 수 조회 | — |

**Padding correction**: zero-padding을 binary {-1,+1}로 표현할 때 발생하는 오차를, 패딩 패턴별 popcount 보정으로 해결. **PyTorch와 bit-exact 일치 (max diff = 0.0000) 검증 완료**.

**OpenMP 통합**: `-fopenmp`로 컴파일. 기본 1 thread (생성자에서 초기화). `cpu_set_num_threads(N)`으로 런타임 조절. 작은 op은 thread spawn overhead 회피 위해 `if(workload > threshold)` 가드 적용.

### 4.2 `export_weights.py`

`.pth` → `.bnn` 커스텀 binary 포맷 변환.

자동 감지:
- `rsign1.threshold` 키 있으면 → **student** (binary conv pack + alpha 저장)
- 없으면 → **teacher** (전부 FP32 저장)

```bash
python3 export_weights.py best_student.pth -o best_student.bnn  # auto-detect
python3 export_weights.py best_teacher.pth -o best_teacher.bnn  # auto-detect
```

### 4.3 `inference_c.py` — 4가지 모드 + 풀 demo 폴리시

| 모드 | 명령 | 역할 |
|---|---|---|
| `--validate` | `--validate --ckpt X.pth --weights X.bnn` | PyTorch vs C 출력 bit-exact 검증 |
| `--demo` | `--demo --weights X.bnn` | 단일 모델 visual demo (auto student/teacher) |
| `--compare` | `--compare --weights stu.bnn --teacher_weights tch.bnn` | 둘 다 C에서 fair 비교 (Student W1A1 vs Teacher FP32) |
| `--compare3` | `--compare3 --weights stu --teacher_weights tch --teacher_ckpt tch.pth` | 3-way: PyTorch BLAS + C Teacher + C Student |

공통 옵션: `--threads N` (기본 1), `--max N` (sample 제한), `--color` (ANSI O/X 색), `--data`, `--show-image` (ASCII thumbnail, demo 모드 전용)

**Demo 폴리시 (모든 모드 적용)**:
- ✅ CIFAR-100 클래스 이름 (49 → "sea", 33 → "forest" 등)
- ✅ Memory footprint banner (Teacher 44 MB vs Student 1.4 MB → 32× 압축)
- ✅ Latency bar 시각화 (`▇▇▇▇▇` per sample)
- ✅ Top-3 predictions + softmax confidence
- ✅ Class-wise accuracy breakdown (worst/best 5)
- ✅ Confusion table (top 5 true→pred 혼동)
- ✅ ASCII 이미지 미리보기 (`--show-image`, 6×18 grayscale)

**Three-way compare 출력 예시**:
```
[ 100/10000] mountain  PyT [O]  11.6ms ▇·   | C-Tch [O] 196.7ms ▇▇▇▇▇▇▇▇▇  | C-Stu [O]  10.9ms ▇·
...
Student faster than C-Teacher :  18.0×  (algorithm only — both naive C)
Student faster than PyTorch   :   1.1×  (vs production FP32 BLAS)
PyTorch faster than C-Teacher :  17.0×  (BLAS+multi-thread vs naive C)
```

→ 핵심 메시지: **C Student W1A1은 PyTorch FP32 BLAS와 동등한 속도**. 같은 C 런타임에선 18× 빠름.

`--demo`는 `.bnn`의 `__model_type` 마커 보고 student/teacher 자동 선택.

---

## 5. 학습 방법 (Kaggle 기준)

### 5.1 환경
- Kaggle T4 ×2 / 12h budget
- Python 3.12, PyTorch 2.x, torchvision v2 transforms

### 5.2 단일 세션 흐름
```
1. train.ipynb 업로드
2. 인터넷 ON (ImageNet pretrained 다운로드 위해)
3. (선택) best_teacher.pth를 input에 첨부하면 Phase A 자동 skip
4. 전체 실행:
   • Phase A: Teacher 80ep FT (~1.5h) — best_teacher.pth 없을 때만
   • Phase B: A1W1 student 300ep + KD (~3h)
   • 제출용 best_student.pth, submission.csv 생성
```

### 5.3 학습 시간 예상 (T4 ×2 단일 GPU 모드)
| Phase | Epoch | 시간/epoch | 총 시간 |
|---|---|---|---|
| Teacher quick FT | 80 | ~11s | ~15분 |
| Student (fp16 binary) | 300 | ~20-25s | ~100-125분 |
| **합** | | | **~2-2.5h** |

---

## 6. C 추론 Demo 방법 (로컬)

학습이 끝나서 `best_student.pth` + `best_teacher.pth` 두 파일이 프로젝트 루트에 있다는 가정.

### 6.1 첫 셋업
```bash
cd .agents/avx_demo/
bash build.sh                                       # C 커널 빌드
python3 export_weights.py ../../best_teacher.pth -o best_teacher.bnn
python3 export_weights.py ../../best_student.pth -o best_student.bnn  # 학습 끝나면
```

### 6.2 검증 (정확도가 PyTorch와 일치하는지)
```bash
python3 inference_c.py --validate \
    --ckpt ../../best_student.pth --weights best_student.bnn
# 기대: max |diff| = 0.0000  ✓
```

### 6.3 Visual demo (단일 모델)
```bash
# Teacher 단독 (학습 끝나기 전엔 이게 가능)
python3 inference_c.py --demo --color --weights best_teacher.bnn

# Student 단독 (학습 끝난 후)
python3 inference_c.py --demo --color --weights best_student.bnn

# 일부만 빠르게
python3 inference_c.py --demo --color --max 100 --weights best_teacher.bnn
```

### 6.4 공정 비교 (둘 다 C, 핵심 발표 자료)
```bash
# Single-thread (알고리즘 차이만, 발표용 기준)
python3 inference_c.py --compare --color --threads 1 \
    --weights best_student.bnn \
    --teacher_weights best_teacher.bnn

# Multi-thread (4 코어, 실용 시나리오)
python3 inference_c.py --compare --color --threads 4 \
    --weights best_student.bnn \
    --teacher_weights best_teacher.bnn

# 100장만 빠르게
python3 inference_c.py --compare --color --max 100 --threads 4 \
    --weights best_student.bnn \
    --teacher_weights best_teacher.bnn
```

기대 출력:
```
[    0/10000] Stu [O]   22.0ms  Tch [O]  556.1ms  speedup 25.27x  acc S=100% T=100%
...
Student acc (W1A1)   : ~68%
Teacher acc (FP32)   : ~82%
speedup (T/S)        : ~25x (1 thread) / ~12x (4 threads)
```

### 6.5 Threading 옵션 (`--threads N`)

| `--threads` | Student | Teacher | Ratio | 의미 |
|---|---|---|---|---|
| **1** (default) | ~22ms | ~550ms | **~25×** | 알고리즘 차이만 (BLAS·SMT 영향 없음) |
| **4** | ~15ms | ~170ms | **~12×** | 실용 환경, 둘 다 OpenMP 가속 |
| 8 (HT 포함) | ~21ms | ~200ms | ~9× | SMT contention 발생, **비추** |

**왜 thread 수에 따라 ratio가 변하나**
- Teacher (FP32): 큰 GEMM 루프 → multi-thread로 3× 가속
- Student (binary): 작은 op이 많아 (16 binary convs) thread spawn overhead로 1.5×만 가속
- 따라서 4 thread 시 teacher가 더 큰 이득 → ratio 25× → 12×로 감소
- **두 숫자 모두 정직한 측정값**. 발표에서 둘 다 보여주면 강력

**Default는 single-thread (1)** — 호환성 + Codex 리뷰의 "fair comparison" 가이드 준수. 다른 thread 수는 명시적으로 `--threads N` 지정.

**CPU 코어 확인**:
```bash
# 실행 시 첫 줄에 표시됨
python3 inference_c.py --compare --threads 4 ...
# → "C threads: 4  (available cores: 8)"
```

---

## 7. 검증 결과 (현재)

| 항목 | 결과 |
|---|---|
| **PyTorch ↔ C bit-exact** | ✅ max diff = 0.0000 (dummy 가중치 검증) |
| **Teacher acc** | 82.44% (학습 완료) |
| **Student acc (1차, AT 1000)** | 68.72% → KD 활성 시 폭발 → 재학습 중 (AT 10) |
| **C inference (1 thread)** | Stu 22ms / Tch 547ms / **25× speedup** |
| **C inference (4 threads)** | Stu 15ms / Tch 173ms / **12× speedup** |
| **CPU 환경** | Cascade Lake AVX-512F + POPCNT (VPOPCNTDQ ✗), 4 cores + HT (8 logical) |

---

## 8. 발표 narrative 권장 흐름

1. **Motivation**: 모바일/엣지에 큰 NN은 못 올린다. 1-bit BNN으로 32× 압축 + 추론 가속.
2. **Method**: ReActNet-lite + KD + Attention Transfer
   - 99% binary conv (16 layers), 1% FP32 (stem + classifier)
   - RPReLU + double-skip + AMP-safe fp16 forward
3. **Training Pipeline** (single Kaggle session):
   - Phase A: ImageNet pretrained → CIFAR FT (teacher)
   - Phase B: Teacher init → A1W1 + logit KD + AT distillation
4. **Quantitative Results**:
   - Teacher 82.4% / Student ~68-70% (literature SOTA 영역)
   - Memory: 44 MB → 1.4 MB (**32× 압축**)
5. **HW Demo** (핵심 차별화):
   - 같은 C runtime에서 W1A1 vs FP32 추론 비교
   - **Single-thread: ~25× speedup** (알고리즘 자체의 우위)
   - **4-thread: ~12× speedup** (실용 환경, 둘 다 OpenMP 가속)
   - Visual per-sample `[O/X]` 추론 — 추론이 눈으로 보임
6. **Caveats / Honest discussion**:
   - W1A1 SOTA (~71%)와의 gap, BNN field 천장
   - C 비교는 naive FP32 vs naive binary (둘 다 BLAS·VPOPCNTDQ 미사용)
   - GPU에서 INT1 native 없는 한계 → CPU demo로 우회
   - Thread 수에 따라 ratio 25× → 12× — teacher가 multi-thread 이득 더 큼

---

## 9. 결정 사항 / 비결정 사항 (FAQ)

**Q. 왜 ResNet18 conv1이 여전히 7×7?**
A. torchvision pretrained 가중치 보존 위해. stride만 1로, maxpool만 제거. 32×32에 7×7 receptive field는 살짝 비효율적이지만, pretrained 활용이 더 큰 이득. Teacher 82.4% 달성.

**Q. Stem/FC INT8 안 함?**
A. 가치 작음. 200 KB만 절감하고 -0.3% acc. BNN 표준은 stem/FC를 FP32로 유지.

**Q. EMA 왜 student에서 끔?**
A. Binary 가중치 평균은 sign()을 부수므로 random 모델이 됨. ReActNet/Bi-Real/ReCU 모두 student EMA 안 씀.

**Q. AT loss weight 1000이 폭발한 이유?**
A. `F.normalize` 적용한 AT map은 0.01-0.5 range. 1000을 곱하면 CE+KL 합(~3) 대비 100배 큼. 현재 10으로 낮춤 + 10-epoch ramp-up.

**Q. T4 GPU에서 binary 가속 실측 가능?**
A. ❌ T4는 INT4까지만 native (Turing). INT1은 A100/H100+. T4에서 binary 모델 돌려도 fp16 conv 비용 그대로. → **CPU AVX demo로 대체**.

**Q. Custom CUDA 만들 가치?**
A. T4에선 ROI 매우 나쁨 (+30-50%만 더). 1-2일 들임 = 비추.

**Q. Multi-thread 켜면 비율이 떨어지는데 (25× → 12×) 그럼 손해 아닌가?**
A. 손해 아님 — 둘 다 빨라짐. Teacher가 GEMM 크기 덕에 multi-thread로 3× 가속, student는 작은 op 많아 1.5×만 가속 → 비율 감소. 절대값은 둘 다 좋아짐 (15ms / 173ms). 발표에선 single-thread (알고리즘 차이) + multi-thread (실용) 둘 다 제시 권장.

**Q. 왜 default가 1 thread?**
A. (1) Codex 리뷰 권고 — fair comparison 가이드, (2) 호환성 — multi-thread는 환경별 (코어 수, SMT, 캐시 크기)로 결과가 달라짐, (3) Single-thread가 알고리즘 차이를 가장 명확히 드러냄. 멀티는 `--threads N` 명시.

---

## 10. 다음 단계 (Plan v4 시나리오 B)

- [x] Plan v1 → v4 작성, Codex 검토
- [x] 노트북 v2 구현 (RPReLU, double-skip, fp16 binary, grad clip)
- [x] Teacher 학습 완료 (82.44%)
- [x] Student 1차 학습 (68.72%)
- [x] Feature KD (AT) 추가 + buggy weight 수정 (1000 → 10 + ramp)
- [x] AVX C 추론 파이프라인 (Level 1 + Level 2: full forward in C)
- [x] PyTorch vs C bit-exact 검증
- [x] Student vs Teacher fair C 비교 모드
- [ ] **Student A1+A2 재학습 (진행 중)** — 70% 노림
- [ ] 학습 완료 후 best_student.pth 받아 C 추론 실측
- [ ] 발표 슬라이드 작성 + 결과 그래프

---

## 11. 참고 사항

**핵심 논문**:
- XNOR-Net (Rastegari+ 2016)
- Bi-Real Net (Liu+ 2018) — double-skip
- **ReActNet (Liu+ 2020)** — RSign + RPReLU + 2-stage training
- ReCU (Xu+ 2021)
- Attention Transfer (Zagoruyko & Komodakis 2017) — AT distillation

**플랜 변천사** (`.agents/plan_v*.md`):
- v1: 2-stage W32A1→W1A1 + Plan B fallback
- v2: Codex 리뷰 반영 (RPReLU, double-skip, alpha detach, fp16 autocast 안전화)
- v4: Single-session 단순화, KD weight 조정, Feature KD 도입
- 다음 v5: 최종 학습 결과 + AVX demo 통합 후

**작업 백업**:
- `.agents/train.backup.ipynb` — cleanup 전 원본 (cnn/vgg yaml 시대 코드 포함)
- `.agents/notebook_excerpt.py`, `.agents/code_v2_for_review.py` — Codex 리뷰용 발췌
