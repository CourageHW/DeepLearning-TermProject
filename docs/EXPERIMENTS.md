# 추가 실험 체크리스트 — 조원 병렬 분담용

**작성일**: 2026-05-14
**목적**: 발표·보고서 보강용 ablation/분석 실험을 조원이 **동시 병렬 실행** 하여 시간 단축
**기반 모델**: A1W1ResNet18v2, baseline = Student 68.72 % (`best_student.pth`, 220 epoch logit KD only)
**컨텍스트**: `gap_checklist_v2.md` §E (Codex + Gemini cross-check 통과한 실험만)

---

## 0. 공통 전제

### 0.1 환경
- Kaggle T4 (≥ 1×) — Phase B 단독 학습 약 80분, Phase A skip (best_teacher.pth 공유)
- 단일 GPU 모드 (DataParallel 미사용 — student 부분)
- Python 3.12, PyTorch 2.x

### 0.2 공유 자산
- `best_teacher.pth` (Teacher 82.44 %) — 모든 실험에서 **공유** (Phase A 재학습 X)
- `train.ipynb` Cell 2 의 hyperparam 만 수정 후 Cell 11 의 `main_pipeline()` 재실행
- seed = 42 (모든 실험 동일)

### 0.3 결과물 보고 형식 (각 실험 공통)
각 담당자는 학습 완료 후 다음 **5개** 를 보고:
```
- best_student_<exp_id>.pth      (체크포인트, 1.4 MB)
- log_<exp_id>.txt               (per-epoch stdout 복사, ~50 KB)
- history_stu_<exp_id>.csv       ★ 학습 곡선 데이터 (epoch×10 columns)
- history_stu_<exp_id>.pkl       ★ 동일 데이터 pickle 형식 (matplotlib 직접 로드용)
- summary_<exp_id>.md            (아래 템플릿)
```

> **★ history 파일은 노트북이 자동 생성합니다** — `_run_phase` 함수가 매 epoch 마다 train/val loss·top1·top5·lr·alpha 를 누적하여 학습 종료 시 CSV + pickle 로 저장 (cell 11). 별도 코드 작성 불필요.

**`summary_<exp_id>.md` 템플릿**:
```markdown
## Experiment <ID>: <이름>
- 담당자: <이름>
- 변경 hyperparam: <key=val 형식>
- best epoch: <N>
- best val Top-1: <XX.XX %>
- best val Top-5: <XX.XX %>
- 학습 시간: <분>
- 비고 (이상 증상, loss 발산 등): <자유 기술>
```

### 0.4 노트북 수정 표준 패턴
모든 실험은 **Cell 2 (Config) 한 곳만 수정**하면 됨. 그 외 cell 은 건드리지 않는다.

```python
# Cell 2 수정 예시 (E1 — KD off)
student_epochs       = 220
student_kd_weight    = 0.0                             # ← 0.7 → 0.0 (KD off)
submission_ckpt_path = "best_student_e1_kd0.pth"       # ← 결과 파일명
exp_id               = "e1_kd0"                        # ★ history 파일도 이 id 로 저장
```

`exp_id` 만 바꿔주면 `history_stu_<exp_id>.csv` / `.pkl` 가 자동 생성됨. 다른 조원 실험과 충돌 안 함.

### 0.5 Baseline history 수집
현재 `best_student.pth` 는 history 없는 상태로 학습됨. **plot 비교를 위해 한 명이 baseline 도 1회 재학습 권장**.
- `exp_id = "baseline"`, `submission_ckpt_path = "best_student_baseline.pth"` (기존 `best_student.pth` 와 분리)
- 모든 다른 hyperparam 은 현재 cell 2 default 그대로 (변경 0)
- 80 분 학습 → `history_stu_baseline.csv` 확보 → 모든 비교 plot 의 기준선

> baseline 재학습이 부담스러우면: 새 실험 결과 그래프에 "baseline 68.72 %" 가로선만 그어도 비교 가능 (plot_curves.py 에 옵션 추가 가능).

---

## 1. 실험 우선순위 + 분담표 (제안)

| 우선순위 | ID | 실험 | 학습 필요? | 예상 시간 | 추천 담당자 수 |
|---|---|---|---|---|---|
| ⭐⭐⭐ **1순위 (필수)** | **E1** | KD on/off ablation | ✅ Phase B 1회 | 80 분 | 1명 |
| ⭐⭐⭐ **1순위 (필수)** | **E4** | Class-wise accuracy + confusion table | ❌ inference 만 | 10 분 | 1명 |
| ⭐⭐⭐ **1순위 (필수)** | **E5** | BitOPs 분석 (paper calc) | ❌ 종이 계산 | 30 분 | 1명 |
| ⭐⭐ **2순위 (권장)** | **E2** | Double-skip on/off | ✅ Phase B 1회 | 80 분 | 1명 |
| ⭐⭐ **2순위 (권장)** | **E3** | RPReLU vs PReLU | ✅ Phase B 1회 | 80 분 | 1명 |
| ⭐ **3순위 (선택)** | **E8** | KD temperature sweep (T=2,4,8) | ✅ Phase B × 2회 | 160 분 | 1명 |
| ⭐ **3순위 (선택)** | **E7** | Mixup α sweep (0.2, 0.4, 0.8) | ✅ Phase B × 2회 | 160 분 | 1명 |

### 1순위만으로 충분한가
**충분.** 1순위 (E1, E4, E5) 만 끝내면:
- E1 → "KD 효과 +N %p" 정량화 — 발표 narrative 핵심
- E4 → 정성 분석 (어떤 class 를 틀리나)
- E5 → 32× 메모리 + N× 연산량 → 발표 정량성 완비

2-3순위는 시간 여유 있을 때만. **3-4명 조라면 1순위 3개 + 2순위 1-2개** 권장.

### 4명 조 추천 분담 (예시)
| 조원 | 담당 |
|---|---|
| A | **E1** (KD on/off, Kaggle) |
| B | **E2** (double-skip, Kaggle 다른 계정) |
| C | **E4 + E5** (학습 불필요, 로컬에서 처리) |
| D | **E3** (RPReLU off, Kaggle 또 다른 계정) |

→ 80 분 만에 4개 실험 완료 + 정성/정량 분석.

---

## 2. 실험별 상세 실행 가이드

### ✅ E1 — KD on/off ablation (1순위 ⭐⭐⭐)

**목적**: Logit KD 의 정량적 기여도 측정. "KD 가 없으면 student 가 몇 % 인가" 답.

**Cell 2 수정 (3 줄만)**:
```python
student_kd_weight    = 0.0                              # 0.7 → 0.0 (KD off)
student_warmup_epochs = 15                              # 그대로
submission_ckpt_path = "best_student_e1_kd0.pth"        # 파일명 분리
```

**예상 결과**: 65 – 67 % (BNN literature 의 no-KD baseline 영역). 만약 60 % 미만이면 학습 이상.

**보고서 반영 위치**: report.md §5.2 또는 §6.2 에 "**KD 효과: +N.N %p** (logit KD off = XX.XX % vs with KD = 68.72 %)" 추가.

**Risk**: KD off 시 mixup 강도가 상대적으로 더 영향 클 수 있으므로 mixup α 는 baseline 과 동일 (0.4) 유지.

---

### ✅ E2 — Double-skip on/off ablation (2순위 ⭐⭐)

**목적**: Bi-Real 의 double-skip 설계가 single-skip 대비 얼마나 기여하는지 정량화.

**Cell 2 수정 (2 줄)**:
```python
student_use_double_skip = False                          # True → False
submission_ckpt_path    = "best_student_e2_singleskip.pth"
```

**예상 결과**: 66 – 67 % (Bi-Real 논문에서 double-skip 기여 +1-2 %p 보고). 만약 차이 미미하면 noise.

**보고서 반영 위치**: report.md §4.2 "Bi-Real 의 double-skip 을 RPReLU 와 결합한 변형" 끝에 "(double-skip 제거 시 -N %p, §6 ablation 참조)" 추가. §6 에 ablation 표.

---

### ✅ E3 — RPReLU vs PReLU ablation (2순위 ⭐⭐)

**목적**: ReActNet 의 RPReLU 가 일반 PReLU 대비 얼마나 기여하는지.

**Cell 2 수정 (2 줄)**:
```python
student_use_rprelu      = False                          # True → False (PReLU fallback)
submission_ckpt_path    = "best_student_e3_noprelu.pth"
```

**참고**: `use_rprelu = False` 시 cell 9 의 `A1W1BasicBlockV2` 가 `RPReLU` 대신 `nn.Identity` 로 fallback (현재 코드 line 524). 즉 act1/act2 가 identity. PReLU 와 직접 비교는 cell 9 수정 필요.
- **단순화**: `nn.Identity` 로 두어도 "RPReLU 의 기여 (vs no-activation)" 측정 가능.

**예상 결과**: 65 – 67 %. 만약 더 떨어지면 RPReLU 가 핵심.

**보고서 반영 위치**: report.md §4.2 마지막에 "RPReLU 제거 시 -N %p" 추가.

---

### ✅ E4 — Class-wise accuracy + Confusion table (1순위 ⭐⭐⭐)

**목적**: 정성 분석. 어떤 class 를 잘 / 잘못 맞추는지.

**학습 불필요**. 로컬에서 `inference_c.py --compare` 실행만 하면 됨.

**실행 명령** (로컬, best_student.pth + best_teacher.pth 필요):
```bash
cd .agents/avx_demo
bash build.sh                          # 최초 1회
python3 export_weights.py ../../best_student.pth -o best_student.bnn
python3 export_weights.py ../../best_teacher.pth -o best_teacher.bnn

python3 inference_c.py --compare --threads 4 --color \
    --weights best_student.bnn \
    --teacher_weights best_teacher.bnn \
    > log_e4_compare.txt 2>&1
```

**산출 데이터**:
- Per-sample `[O/X]` predictions
- Top-3 predictions + softmax confidence (자동 출력)
- **Class-wise accuracy breakdown (worst/best 5)** ← 캡처
- **Confusion table (top 5 true→pred 혼동)** ← 캡처

**보고서 반영 위치**:
- report.md §6 끝에 §6.6 "Qualitative analysis" 신규 절 — worst 5 class + top confusion pairs 표
- slides.md backup 슬라이드 추가 ("Failure analysis")

---

### ✅ E5 — BitOPs 분석 (1순위 ⭐⭐⭐)

**목적**: 32× 메모리 압축에 대응하는 **연산량 (FLOPs vs BitOPs) 정량 비교**.

**학습 불필요**. 종이 계산 + 노트북에 간단 코드 1 셀 추가.

**계산 절차**:
1. ResNet18 (CIFAR head, stem 3×3 stride 1, maxpool→Identity) 의 FP32 FLOPs 측정
   - Tools: `torchinfo`, `thop`, `fvcore` 중 택1
   - 예상값: ≈ **1.82 GFLOPs** (CIFAR head)
2. A1W1ResNet18v2 의 **분해 계산**:
   - FP32 stem (Conv1 3×3, 3→64): ≈ 5 MFLOPs (full FP32)
   - FP32 FC (512→100): ≈ 0.05 MFLOPs
   - Binary blocks (8 blocks, 16 binary conv): FP32 conv 비용 약 1.81 GFLOPs **× 1/64 ratio (uint64 packed popcount)** ≈ **28 MFLOPs equivalent**
   - BN, RSign, RPReLU: < 5 MFLOPs (negligible)
3. 비율 계산: **FP32 1.82 GFLOPs / Binary ≈ 33 MFLOPs ≈ 55× 이론 연산량 감소**
4. 실제 wall-clock 25× (single-thread) 와의 격차 → 메모리 access, control overhead 로 설명

**노트북에 추가할 cell (선택)**:
```python
# Cell 13 (BitOPs 분석)
from torchinfo import summary
teacher = build_teacher().to(device).eval()
student = A1W1ResNet18v2(num_classes=100).to(device).eval()
print("=== Teacher (FP32) ===")
summary(teacher, input_size=(1, 3, 32, 32), col_names=["output_size", "num_params", "mult_adds"])
print("\n=== Student (W1A1) ===")
summary(student, input_size=(1, 3, 32, 32), col_names=["output_size", "num_params", "mult_adds"])
# Multiply student binary conv mult_adds by 1/64 manually for BitOPs estimate
```

**보고서 반영 위치**: report.md §6.4 Compression analysis 에 다음 표 추가:

| 항목 | Teacher (FP32) | Student (W1A1) | 비율 |
|---|---|---|---|
| 파라미터 수 | 11.22 M | 11.22 M | 1× |
| 저장 비트 수 | 358.9 Mbits | 12.8 Mbits | 28× |
| 디스크 크기 | 44.0 MB | 1.4 MB | 31.4× |
| **이론 연산량** | ≈ 1.82 GFLOPs | ≈ 33 MFLOPs (FP32 stem/FC + binary popcount × 1/64) | **≈ 55×** |
| **실측 latency (1-thread)** | 547 ms | 22 ms | 25× |

**Slides 반영**: slide 7 의 "📦 Memory" 줄 옆에 "🧮 Compute ~55× theoretical" 추가 가능.

---

### ⚪ E7 — Mixup α sweep (3순위 ⭐, 선택)

**목적**: Phase B 의 mixup α 가 최적값 0.4 인지 검증.

**Cell 2 수정** (3개 run):
```python
# Run 7a: α=0.2
student_mixup_alpha = 0.2
submission_ckpt_path = "best_student_e7a_mixup02.pth"

# Run 7b: α=0.8
student_mixup_alpha = 0.8
submission_ckpt_path = "best_student_e7b_mixup08.pth"
```

**예상 결과**: 67 – 69 % (small variation). 발표 임팩트 낮음.

---

### ⚪ E8 — KD temperature sweep (3순위 ⭐, 선택)

**목적**: KD T = 4 가 최적인지 sweep.

**Cell 2 수정** (2개 run):
```python
# Run 8a: T=2
student_kd_T = 2.0
submission_ckpt_path = "best_student_e8a_T2.pth"

# Run 8b: T=8
student_kd_T = 8.0
submission_ckpt_path = "best_student_e8b_T8.pth"
```

**예상 결과**: 67 – 69 %. T 영향은 보통 작음.

---

## 3. 결과 통합 절차

각 담당자가 결과 보고 후 통합:

### 3.1 메인 담당자 (예: 발표자) 가 할 일

#### Step 1: 파일 수집
조원들의 `history_stu_*.csv` (+ `.pkl`, `summary_*.md`, `best_student_*.pth`) 를 프로젝트 루트로 모음.

#### Step 2: 비교 plot 생성 (학습 곡선)
```bash
cd /home/yonggi/KAU/3-1/딥러닝/Termp

# 4개 실험 비교 (baseline 포함)
python3 plot_curves.py \
  --inputs history_stu_baseline.csv \
           history_stu_e1_kd0.csv \
           history_stu_e2_singleskip.csv \
           history_stu_e3_no_rprelu.csv \
  --labels "Baseline" "KD off (E1)" "Single-skip (E2)" "RPReLU→Identity (E3)" \
  --output curves_phase_b_ablation.png \
  --title "Phase B (Student) — Design Ablation Curves"

# 단일 metric (val top-1 만) — 슬라이드용 슬림 버전
python3 plot_curves.py \
  --inputs history_stu_baseline.csv history_stu_e1_kd0.csv \
  --labels "Baseline (with KD)" "Without KD" \
  --output curves_kd_effect.png \
  --metric top1 \
  --title "KD Effect on Student Top-1"
```

자동 산출 콘솔 출력:
```
Saved curves_phase_b_ablation.png
=== Final epoch summary ===
  Baseline                       | last_ep=220 final_val=68.45% best_val=68.72% (ep 213)
  KD off (E1)                    | last_ep=220 final_val=65.10% best_val=65.34% (ep 218)
  ...
```

#### Step 3: 보고서·슬라이드 반영
1. `report.md` 의 각 위치에 정량 데이터 삽입:
   - E1 → §5.2 "KD 효과: **+3.4 %p** (logit KD off = 65.3 % vs 68.7 %)"
   - E2 → §4.2 "double-skip 효과: +N %p"
   - E3 → §4.2 "RPReLU 효과: +N %p"
   - E4 → §6.6 새 절 (Qualitative analysis)
   - E5 → §6.4 표 (이론 연산량)
2. `report.md` §6.6 ablation 표 + **학습 곡선 그림** 임베드:
   ```markdown
   ![Phase B 학습 곡선 비교](curves_phase_b_ablation.png)
   *Figure 6: 4가지 student 설계 변형의 학습 곡선. dashed = train, solid = val.*
   ```
3. `slides.md` 변경:
   - **신규 본문 슬라이드 추가** (slide 7 결과 다음 / slide 8 전): "Student Design Ablation" — `curves_kd_effect.png` 또는 `curves_phase_b_ablation.png` 임베드 + 5행 ablation 표
   - 백업 F (새로 추가): 전체 비교 plot
4. PDF 재컴파일 (`pandoc` / `marp` 명령은 PROGRESS.md §4 참조)

### 3.2 보고서 새 ablation 표 예시 (메인 담당자 작업)
```markdown
### 6.6 Student Design Ablation (4명 조원 1회씩 학습)

| 변형 | Top-1 | Δ |
|---|---|---|
| Baseline (logit KD T=4 w=0.7, double-skip, RPReLU) | 68.72 % | — |
| KD off (E1) | XX.XX % | -N.N %p |
| Single-skip (E2) | XX.XX % | -N.N %p |
| RPReLU → Identity (E3) | XX.XX % | -N.N %p |
```

---

## 4. 보고 / 의사소통 채널

조원 간 공유 권장 폴더 구조 (Google Drive / GitHub 등):
```
team16-experiments/
├── E0_baseline_rerun/             ← (선택) plot 비교 기준선 생성용
│   ├── best_student_baseline.pth
│   ├── history_stu_baseline.csv   ★
│   ├── history_stu_baseline.pkl   ★
│   ├── log_baseline.txt
│   └── summary_baseline.md
├── E1_kd_off/
│   ├── best_student_e1_kd0.pth
│   ├── history_stu_e1_kd0.csv     ★
│   ├── history_stu_e1_kd0.pkl     ★
│   ├── log_e1.txt
│   └── summary_e1.md
├── E2_singleskip/
│   ├── ... (동일 구조, exp_id="e2_singleskip")
├── E3_no_rprelu/
│   ├── ... (동일 구조, exp_id="e3_no_rprelu")
├── E4_qualitative/
│   ├── log_e4_compare.txt
│   └── summary_e4.md              ← worst/best 5 class + confusion table 캡처
├── E5_bitops/
│   └── summary_e5.md              ← BitOPs 표
├── plots/                         ← 메인 담당자가 통합 생성
│   ├── curves_phase_b_ablation.png
│   ├── curves_kd_effect.png
│   └── curves_kd_effect_slim.png  ← 슬라이드용
└── README.md                      ← 진행률 / 담당 / 마감
```

**★ 표시된 파일** (`history_stu_*.csv/.pkl`) 은 노트북이 자동 생성 — 조원은 신경 안 써도 됨.

---

## 5. 체크리스트 (조원 진행 트래킹용)

| ID | 담당자 | 상태 | 시작 | 완료 | best Top-1 | history 제출 | 비고 |
|---|---|---|---|---|---|---|---|
| E0 (baseline rerun) | | ⏸ optional | | | | ☐ | plot 기준선 — 필수 아님 |
| E1 (KD off) | | ⏳ pending | | | | ☐ | ⭐⭐⭐ |
| E2 (single-skip) | | ⏳ pending | | | | ☐ | ⭐⭐ |
| E3 (no RPReLU) | | ⏳ pending | | | | ☐ | ⭐⭐ |
| E4 (qualitative) | | ⏳ pending | | | | n/a | ⭐⭐⭐ 학습 불필요 |
| E5 (BitOPs) | | ⏳ pending | | | | n/a | ⭐⭐⭐ 학습 불필요 |
| E7 (mixup sweep) | | ⏸ optional | | | | ☐ | |
| E8 (KD T sweep) | | ⏸ optional | | | | ☐ | |

**상태 표기**: `⏳ pending` → `🟡 running` → `✅ done` / `❌ failed`
**history 제출**: ☐ → ☑ (history_stu_<exp_id>.csv + .pkl 둘 다 전달 시 체크)

---

## 6. 시간표 (4명 조 + 1순위 + 2순위 모두 진행하는 경우)

| 시간 | A (E1) | B (E2) | C (E4+E5) | D (E3) |
|---|---|---|---|---|
| 0:00 | Kaggle 세션 시작 | Kaggle 세션 시작 | 로컬 빌드 (avx_demo) | Kaggle 세션 시작 |
| 0:10 | Phase B 학습 중 | Phase B 학습 중 | E4 inference 실행 | Phase B 학습 중 |
| 0:20 | Phase B 학습 중 | Phase B 학습 중 | E4 결과 캡처 | Phase B 학습 중 |
| 0:30 | Phase B 학습 중 | Phase B 학습 중 | E5 BitOPs 계산 | Phase B 학습 중 |
| 0:40 | Phase B 학습 중 | Phase B 학습 중 | summary_e4 / e5.md | Phase B 학습 중 |
| 1:20 | ✅ 완료 | ✅ 완료 | ✅ 완료 | ✅ 완료 |
| 1:30 | 메인 담당자 결과 통합 + 보고서·슬라이드 반영 |

**총 소요: ~ 1 시간 30 분** (병렬 학습 80 분 + 통합 10 분)

---

## 7. 발표 narrative 강화 효과 (예상)

E1-E5 + 학습 곡선 plot 모두 끝나면 발표에서:
- **(NEW) 학습 곡선 슬라이드** — 4개 실험을 한 그래프에 겹쳐 표시, "KD off (E1) 가 KD 보다 일관되게 낮음" 시각 증명
- "**KD 가 +N %p 기여** (E1 ablation)" — 단순 수치 → 인과 증명
- "double-skip / RPReLU 각각 +N %p (E2/E3)" — 설계 정당화
- "32× 메모리 + **~55× 이론 연산량** + 25× 실측 speedup (E5)" — 정량 trio
- "Worst class: insect/lizard 등 visual similar pair (E4)" — 정성 깊이

→ **학부 텀프로젝트 평가단에게 "체계적 ablation 수행 + 학습 곡선 시각화" 인상** 추가.

### 슬라이드 임베드 예시 (수정 후 slides.md)
```markdown
## Student Design Ablation (NEW)

<style scoped> section { font-size: 22px; } </style>

![w:900](curves_phase_b_ablation.png)

| 변형 | best Top-1 | Δ |
|---|---|---|
| Baseline | 68.72 % | — |
| KD off (E1) | 65.34 % | -3.38 %p |
| Single-skip (E2) | 67.10 % | -1.62 %p |
| RPReLU → Identity (E3) | 66.95 % | -1.77 %p |
```

---

## 8. Memory / Risk note

- 외부 모델 호출 ledger: 없음 (본 체크리스트는 이미 cross-check 된 v2 기준)
- 기존 `best_student.pth` 는 **건드리지 않음** — 새 실험은 모두 별도 파일명 사용
- Kaggle 세션이 12 h budget 이므로 한 세션에 80 분 학습 1회 충분히 여유
