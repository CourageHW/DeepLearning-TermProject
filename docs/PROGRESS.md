# PROGRESS.md — 보고서·발표 자료 작성 진행 상황

**최종 업데이트**: 2026-05-15 (Task 1–14: 보고서 마감일 분석 보강)
**작업 범위**: 학습된 모델 기반 보고서·PPT(10 분 발표) 작성 → PDF 변환 → 비약·보충 진단 → 조원 병렬 실험 인프라 → ablation 결과 보고서·슬라이드 통합 → **BitOPs / latency 시각화 / class-wise 분석 / padding correction narrative**

---

## 0. 현재 상태 (한 줄 요약)

✅ **보고서 제출 가능 상태 (오늘 마감) — 분석 4 종 + PDF 재컴파일 완료**.
- `report.md` → `report.pdf` (**14 페이지**, §6.4 BitOPs + §6.6 ablation + §6.7 class-wise + §7.1 padding correction)
- `slides.md` (Marp) → `slides.pdf` (**18 페이지**, Backup F 추가)
- `student_history/` — 4 실험 (baseline + E1/E2/E3) × 220 epoch × 10 metric CSV/PKL
- Visualization: `curves_phase_b_ablation.png` + `ablation_bars.png` + `latency_bars.png` + `curves_kd_effect.png`
- 분석 스크립트: `compute_bitops.py` (thop + manual cross-check) + `eval_class_wise.py` (PyTorch CPU inference)
- 핵심 발견:
  - **연산 압축 53.8× 이론** (메모리 32× 와 별개) — 실측 25× 는 이론치 46 %
  - **RPReLU 단일 최대 기여 (−9.22 %p)** — ablation
  - **Worst classes 가 semantic cluster 형성** — 포유류·사람·나무 fine-grained 구별이 약함

남은 작업: 발표 리허설 (예상 10 분), HW pipeline diagram, slide 본문 footnote 일부.

---

## 1. 핵심 사실 (확정값)

| 항목 | 값 |
|---|---|
| 데이터셋 | CIFAR-100 (Kaggle competition `26-deep-learning-course`) |
| Teacher | ResNet18 (ImageNet pretrained → CIFAR FT, 80 ep) = **82.44 %** (EMA shadow eval) |
| Student | A1W1ResNet18v2 (ReActNet-lite, 1-bit W + 1-bit A) = **68.72 %** (raw, 1차 결과) |
| Literature 비교 | ReActNet 69.4 % / ReCU 66.4 % / RBNN ~ 67.1 % |
| 메모리 압축 | 44 MB → 1.4 MB (이론 32×, .pth 31.4×, 비트 예산 28×) |
| C 추론 speedup | single-thread **25×**, 4-thread **12×** (Intel Cascade Lake) |
| 학생 정확도 채택 | 68.72 % (1차) — Teacher 재학습 X |
| 발표 메시지 비중 | **BNN 정확도 7 : HW demo 3** |
| 학습 곡선 | 초기엔 미수록 결정 → Task 12 에서 인프라 추가, 조원 실험 시 자동 수집 |
| Ablation (KD/double-skip/RPReLU) | **완료 (2026-05-15)** — E1: −2.30 / E2: −2.46 / E3: −9.22 %p (RPReLU 가 단일 최대 기여) |

---

## 2. 완료된 작업 트레일 (Task 1–12)

### Task 1. 프로젝트 현황 파악
- `PROJECT.md`, `train.ipynb` (13 cells), CSV 7-step ablation, `.agents/avx_demo/` 정독.
- CSV: ResNet18 from-scratch 150 ep, 75.74 % → **83.12 %** 최종.

### Task 2. 보고서·PPT 아키텍처 v1 → v2 (Codex + Gemini cross-check)
- 결과물: `.agents/report_slides_architecture_v1.md`, `_v2.md`.
- 슬라이드 12 → **11**장 (HW Design + Speedup merge), AT 폭발 정직 포함, Related Work 0.7 p 압축.

### Task 3. 초안 작성 (report.md + slides.md)
- 보고서 9 sections + abstract/refs. Marp 11 본문 + 5 백업.

### Task 4. 최종본 — Codex + Gemini 초안 cross-check 후 10 건 수정
- CRITICAL: KD KL divergence 방향 `KL(p_s‖p_t)` → **`KL(p_t‖p_s)`**. EMA 설명 softening.
- HIGH: 메모리 비율 정의, Teacher 82.44 vs CSV 83.12 주석, α detach 정확화.
- LOW: SOTA 범위화, Threading 원인, FP32 1% 구체화, Δ 컬럼, ReActNet 격차 한 줄.

### Task 5. 재훈련 vs 수정 결정 — Codex + Gemini 만장일치 "내용 수정"
- 근거: ML noise + paradigm 차이 + 메인 성과 무관.
- 반영: §3.9 "ablation = from-scratch 150 ep, Phase A = pretrained FT 80 ep" 명시.

### Task 6. PDF 자료(트랙 B)로 데이터 증강 풍부화
- `v2&bit_flip 1.pdf` p18-25 활용 → 보고서 §5.6 신규 절 + 슬라이드 백업 C 추가.

### Task 7. PDF 변환 문제 진단 및 수정 (Codex + Gemini cross-check)
- 진단: `.agents/pdf_diagnosis_v1.md`.
- **slides.pdf overflow 8장** 해결: 폰트 26 → **24 px**, padding 조정, 콘텐츠 다이어트.
- **report.pdf glyph 7건** 해결: `‖F‖₂` / `≤` / ASCII art / Mixup 표 cell / longtable / T₀ / 한글 코드 주석.

### Task 8. 보고서·발표자료 논리적 비약·보충 검토 (v1)
- 진단: `.agents/gap_checklist_v1.md` (39 항목 × 5 카테고리).
- A (CRITICAL) 2 / B (HIGH) 6 / C (MEDIUM) 8 / D (LOW) 7 / E (추가 실험) 9.

### Task 9. 체크리스트 v1 cross-check (Codex + Gemini) → v2
- 진단: `.agents/gap_checklist_v2.md`.
- 두 모델 합의: A1/A2/B4/B5/B6 동의 / B3 강등 / E6·E9·D7 삭제 (학부 over-engineering).
- 신규: **A3** (PROJECT.md `student_epochs=300` outdated) / **B7** (AT ramp timing) / **C9** (Padding Correction narrative).
- E 우선순위: **🥇 E1 (KD on/off) · 🥈 E5 (BitOPs) · 🥉 E4 (class-wise)**.

### Task 10. v2 체크리스트 CRITICAL/HIGH 9 건 적용
- **A1** RandomErasing scale → `default (0.02, 1/3)`.
- **A2** §6.1 에 "Normalize = ImageNet stats (pretrained 호환)" 명시.
- **A3** PROJECT.md `student_epochs = 220` 동기화.
- **B1** §5.2 "KD epoch 15 부터 활성화".
- **B2** §4.1 wording 명확화 — STE backward identity vs latent clamp(-1,1) 구분.
- **B4** §5.2 student init — shape 일치만 복사, stem 은 random init.
- **B5** §6.2 — Teacher 82.44 % = EMA shadow, Student 68.72 % = raw.
- **B6** §3 — CSV "SWA enable" = custom EMA class clarification.
- **B7** §5.3 — AT ramp: epoch 15 부터 10 epoch 동안 0 → final_weight.

### Task 11. 조원 병렬 분담용 실험 체크리스트
- 산출물: `EXPERIMENTS.md` (8 sections, 7 실험 + baseline 옵션).
- 1순위: **E1 (KD off) · E4 (class-wise) · E5 (BitOPs)**.
- 2순위: **E2 (single-skip) · E3 (no RPReLU)**.
- 4명 조 1시간 30분 → 4실험 + 정성/정량 분석 동시 완료 가능.

### Task 12. 학습 곡선 수집 + 비교 plot 인프라
- **노트북 in-place patch** (백업: `.agents/train.ipynb.bak.before_history`):
  - Cell 0: `import pickle`
  - Cell 2: `exp_id = "baseline"` (실험별 변경)
  - Cell 11 `_run_phase`: 매 epoch 10 metric (train/val loss·top1·top5, EMA, lr, alpha) 자동 누적
  - Cell 11 종료 시: `history_{phase_name.lower()}_{exp_id}.csv` + `.pkl` 자동 저장
  - Phase A skip 경로에도 teacher 1-row history dump
- **비교 plot 도구**: `plot_curves.py` (argparse CLI)
  - 여러 `history_stu_*.csv/.pkl` 입력 (자동 detect)
  - 2-panel PNG (loss + top1), train=dashed / val=solid / EMA=dotted
  - Final epoch summary 콘솔 출력
  - Sanity check 통과 (fake 2 run → PNG 134KB 생성)

---

## 3. 산출물 위치

```
/home/yonggi/KAU/3-1/딥러닝/Termp/
├── PROJECT.md                       — 프로젝트 사양 (Task 10에서 student_epochs=220 동기화)
├── PROGRESS.md                      — 이 파일 (작업 트레일 / 결정 사항 / 다음 단계)
├── EXPERIMENTS.md                   ★ NEW — 조원 분담 가이드 (Task 11+12)
│
├── report.md                        ★ 보고서 source (Markdown)
├── slides.md                        ★ 슬라이드 source (Marp)
├── report.pdf                       ★ 보고서 PDF (11 페이지)
├── slides.pdf                       ★ 슬라이드 PDF (17 페이지)
│
├── train.ipynb                      ★ 학습 노트북 (Task 12 history patch 적용)
├── plot_curves.py                   ★ NEW — 학습 곡선 비교 plot CLI
│
├── DeepLearning Term Project Team16(Sheet1).csv  — Teacher ablation 데이터
├── v2&bit_flip 1.pdf                — 데이터 증강 자료 출처 (참고용)
│
└── .agents/
    ├── report_slides_architecture_v1.md  — 아키텍처 초안
    ├── report_slides_architecture_v2.md  — 아키텍처 확정안
    ├── pdf_diagnosis_v1.md               — PDF 변환 진단
    ├── gap_checklist_v1.md               — 비약·보충 1차 진단
    ├── gap_checklist_v2.md               — cross-check 반영 (확정)
    ├── train.ipynb.bak.before_history    — history patch 직전 백업
    ├── best_teacher.pth                  — Teacher checkpoint (82.44 %)
    ├── best_student.pth                  — Student checkpoint (68.72 %)
    └── avx_demo/                          — C 추론 파이프라인
```

---

## 4. PDF 재생성 명령 (확정안)

```bash
cd /home/yonggi/KAU/3-1/딥러닝/Termp

# Report (Markdown → PDF, pandoc + xelatex)
pandoc report.md -o report.pdf --pdf-engine=xelatex \
  -V mainfont="Noto Sans CJK KR" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=20mm \
  -V longtable=true

# Slides (Marp → PDF)
npx @marp-team/marp-cli slides.md --pdf --allow-local-files

# Slides → PPTX (PowerPoint 호환)
npx @marp-team/marp-cli slides.md --pptx --allow-local-files
```

---

## 5. 핵심 결정 사항 (트레일)

| # | 결정 | 근거 |
|---|---|---|
| D1 | Student 정확도 = 68.72 % (1차 결과) | Codex + Gemini 만장일치 "효과 < 비용" |
| D2 | 발표 비중 = BNN 7 : HW 3 | 학부 평가단이 정확도 비교 중심 |
| D3 | 결과물 포맷 = Markdown + Marp | 깃 관리·재현 용이 |
| D4 | 슬라이드 = 11 본문 + 5 백업 | 10분 발표에 50초/장 + Q&A 대응 |
| D5 | C Inference Design + Speedup 한 슬라이드 (#9) | Gemini 권고 — HW 메시지 집중화 |
| D6 | Limitations 단독 슬라이드 (#10) | 학부 평가에서 "솔직한 한계 인식" 가점 |
| D7 | Student 정확도 reveal = 슬라이드 #7 | 도입 노출 X — narrative 긴장감 유지 |
| D8 | Student/KD ablation 초기 미실시 (D20 로 갱신) | 시간 제약, §8 Limitations 명시 |
| D9 | Attention Transfer disabled | weight 1000 → 10 ramp 시도 실패, BN 불안정 |
| D10 | Student EMA disabled | latent weight oscillation 평탄화 (BNN literature 관행) |
| D11 | KD KL 방향 = `KL(p_t‖p_s)` | 표준 Hinton + PyTorch 구현 일치 |
| D12 | Teacher 재훈련 X | 학습 패러다임 차이로 ablation best 전이 불보장 |
| D13 | 학습 곡선 초기 미수록 → **D20 로 갱신** | 사용자 결정 변경 (Task 12) |
| D14 | RandomErasing scale = default `(0.02, 1/3)` | 노트북 코드가 default 사용 — 보고서 수정 |
| D15 | Normalize = ImageNet stats | pretrained ResNet18 호환성 우선 |
| D16 | Teacher 82.44 % = EMA shadow / Student 68.72 % = raw | cell 11 best ckpt 저장 로직 |
| D17 | Binary weight 안정화 = "STE identity + 매 step latent clamp(-1,1)" | ReActNet 표준 — 보고서 wording 명확화 |
| D18 | KD start epoch = 15 (LR warmup 종료 시점) | 노트북 `distill_start_ep = student_warmup_epochs` |
| D19 | AT loss ramp = epoch 15 부터 10 epoch 동안 0 → final | 실제 코드 cell 6 line 348 |
| **D20** | **조원 병렬 ablation 실험 추진 (E1/E2/E3/E4/E5)** | Codex + Gemini Top 3 + 학부 평가 narrative 강화 |
| **D21** | **학습 곡선 자동 수집 인프라 구축** | 발표에 비교 plot 보이기 위함 (사용자 요청 변경) |
| **D22** | 학부 over-engineering 항목 제거: E6 / E9 / D7 | Codex + Gemini 합의 |

---

## 6. 다음 단계 — 조원 분담 후 실행

### 즉시 가능 (조원 분담만 결정되면)

#### Phase 1: 학습 / 분석 (병렬, 80 분 학습)
- [x] **E0 (선택) — Baseline rerun**: `exp_id="baseline"` 로 1회 재학습 → `history_stu_baseline.csv` 생성 (plot 기준선)
- [x] **E1 — KD on/off**: `student_kd_weight=0.0`, `exp_id="e1_kd0"` (⭐⭐⭐ 1순위)
- [x] **E2 — Single-skip**: `student_use_double_skip=False`, `exp_id="e2_singleskip"` (⭐⭐ 2순위)
- [x] **E3 — No RPReLU**: `student_use_rprelu=False`, `exp_id="e3_no_rprelu"` (⭐⭐ 2순위)
- [ ] **E4 — Class-wise/confusion**: `.agents/avx_demo/inference_c.py --compare` 실행, 출력 캡처 (⭐⭐⭐ 학습 불필요)
- [ ] **E5 — BitOPs 분석**: 종이 계산 + `torchinfo` 등으로 검증 (⭐⭐⭐ 학습 불필요)

#### Phase 2: 결과 통합 (메인 담당자, 30 분) — **완료 2026-05-15**
- [x] 조원 history 파일 수집 → `student_history/` (4 실험 × CSV+PKL = 8 파일)
- [x] `python3 plot_curves.py --inputs student_history/history_stu_*.csv --output curves_phase_b_ablation.png`
- [x] `curves_kd_effect.png` (KD on/off 단일 metric) 추가 생성
- [x] 보고서 §6.6 "Student Design Ablation" 신규 절 — 4-row 표 + plot + 3 관찰 + 합산 추정
- [x] 슬라이드 Backup F — Student Design Ablation 추가 (본문 11장 budget 유지)
- [x] PDF 재컴파일 (§4 명령)

### 발표 직전 추가 보강 (선택, 30 분)
- [ ] gap_checklist_v2 §C MEDIUM 일부 (C1 BitOPs, C5 bit-width 컬럼 등)
- [ ] Marp speaker note 텍스트 추출 → 발표자 cue note
- [ ] 발표 리허설 1회 (예상 9 분 40 초)

### 후속 (발표 후, 선택)
- [ ] gap_checklist_v2 §D LOW 일괄 적용
- [ ] **E7 / E8** sweep (Mixup α / KD T)
- [ ] AVX-512 VPOPCNTDQ + BLAS-style GEMM 으로 C 커널 확장
- [ ] SAM optimizer / AT loss 안정화

---

## 7. 컨텍스트 보존용 메모

### Q&A 예상 질문 + 답안 (2026-05-15 Codex review 기반)

1. **Q. 왜 GPU가 아니라 CPU로 추론 측정했나요?**
   A. T4 (Turing) 는 INT4 까지만 native 가속을 지원하고 INT1 native 는 A100/H100 이상. T4 에서 binary 모델을 돌려도 fp16 conv 비용 그대로라 GPU 가속의 의미가 없음. **알고리즘 차이를 정직하게 보여주려면 CPU 가 정직**.

2. **Q. "No RPReLU" 는 PReLU 로 대체한 건가요, 제거한 건가요?**
   A. block 안의 RPReLU 는 `nn.Identity()` 로 **완전 제거**. Stem (`act_in`) 만 `use_rprelu=False` 일 때 `nn.PReLU(64)` 로 대체 (RPReLU 가 BatchNorm-like running stats 없이 학습 가능한 형태). 즉 E3 = "block activation 제거" 가 정확한 표현.

3. **Q. 25× speedup 은 PyTorch FP32 대비인가요, C teacher FP32 대비인가요?**
   A. **동일 C 런타임 안의 student vs teacher** 비교. PyTorch FP32 와 비교하면 BLAS/MKL gap 때문에 25× 가 안 나옴. "동일 런타임 알고리즘 차이" 라는 표현이 정확. (§7.4 와 §8 Limitations 에 명시됨)

4. **Q. KD weight 0.7 은 어떻게 정했나요?**
   A. ReActNet 논문이 KD weight 1.0 (pure distillation) 을 쓰는데, 본 환경에서 base CE loss 가 너무 약해져 학습 후반 plateau 가 빨라짐. 0.7 은 0.5-1.0 sweep 없이 선택한 합리적 default 임. **KD weight sweep 은 후속 작업 (Future Work)** 으로 명시.

5. **Q. Teacher 가 EMA shadow 인데 student EMA 는 왜 disable 했나요?**
   A. Student weight 는 binary (sign 함수 출력) 라 평균이 무의미 — EMA shadow 가 random 모델로 수렴. ReActNet, Bi-Real, ReCU 모두 동일하게 student EMA 비활성. **§5.4 에 명시**.

6. **Q. RPReLU 가 9.22 %p 기여인데, 천장 71 % 도 RPReLU 덕인가요?**
   A. 정확히는 RPReLU + α detach + double-skip + KD 의 조합이 천장에 도달하는 데 필요. RPReLU 가 단독 최대 기여이지만, 다른 컴포넌트 없이는 RPReLU 만으로도 천장에 못 도달함. **단일 ablation 만으로는 cross-component interaction 미측정** (보고서 §6.6 명시).

7. **Q. 메모리 비율 28× vs 31.4× vs 32× 어느 게 맞나요?**
   A. 셋 다 정직한 measurement (§6.4):
   - **28×**: bit 예산 정의 (FP32 32-bit vs 1-bit + overhead)
   - **31.4×**: 디스크 .pth 파일 크기
   - **32×**: 이론 상한 (overhead 무시 시 FP32 → 1-bit)

---

### Task 15 산출물 (2026-05-15 BLAS-backed FP32 teacher conv 추가)

**목적**: §8 Limitations 의 "BLAS 미사용 한계" 를 실제 구현으로 보강 + fair speedup 비교 데이터 확보.

| 파일 | 역할 |
|---|---|
| `.agents/avx_demo/fp32_blas_kernel.c` | im2col + `cblas_sgemm` 기반 FP32 conv. 헤더 없이 cblas ABI 직접 declare → 모든 BLAS 구현과 link 호환 |
| `.agents/avx_demo/build.sh` (확장) | `fp32_blas_kernel.so` 빌드 target 추가 (`-l:libblas.so.3`) |
| `.agents/avx_demo/inference_c.py` (확장) | `FP32ResNet18Engine(use_blas=...)` 옵션 + `--blas` CLI flag |
| `.agents/avx_demo/bench_blas_vs_naive.py` | conv-layer-only bench, PyTorch vs naive vs BLAS 정확도 + latency |

**결과 (CIFAR-100 50 samples, OpenBLAS via LD_PRELOAD)**:

| Config | Student (W1A1) | Teacher (naive) | Teacher (BLAS) | T/S (naive) | T/S (BLAS) |
|---|---|---|---|---|---|
| 1-thread | 23.3 ms | 530.5 ms | **74.8 ms** | 22.7× | **3.15×** |
| 4-thread | 10.7 ms | 160.7 ms | **65.4 ms** | 15.0× | **6.12×** |

- 정확도: 둘 다 student 70% / teacher 78% (50 sample, 완전 동일 prediction)
- 정확도 gap vs PyTorch: max 5e-6 (fp32 summation order noise)
- LD_PRELOAD 으로 OpenBLAS 사용 시 reference libblas 대비 7× 추가 빠름

**실행 예시**:
```
# Default (reference BLAS via -l:libblas.so.3 from build)
python3 inference_c.py --compare --threads 1 --blas

# Production OpenBLAS (LD_PRELOAD)
LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/opencv_python.libs \
LD_PRELOAD=~/.local/lib/python3.10/site-packages/opencv_python.libs/libopenblas-r0-f650aae0.3.3.so \
python3 inference_c.py --compare --threads 4 --blas
```

**보고서 narrative 영향 (사용자 선택)**:
- 현재 §7.3 / §8 의 "naive C 25×" 옆에 "BLAS 3.15× ~ 6.12×" 를 추가하면 §8 Limitations 의 "BLAS 미사용" 항목이 자동 해결 + 더 정직한 비교 narrative.
- 또는 헤드라인 number 유지하고 §7/§8 에 한 절만 추가.

---

### Task 14 산출물 (2026-05-15 보고서 마감일 분석 보강)

| # | 분석 | 보고서 위치 | 산출 파일 |
|---|---|---|---|
| 1 | BitOPs 분석 (FP32 555.5M MACs vs Student 10.4M FP32-equiv) → 53.8× 이론 압축 | §6.4 (확장) | `compute_bitops.py` |
| 2 | Latency 시각화 (bar chart + 53.8× 이론 reference line) | §7.3 figure | `latency_bars.png` |
| 3 | Class-wise breakdown — worst 10 / best 10 / top confusions, semantic cluster 분석 | §6.7 (신규) | `eval_class_wise.py` |
| 4 | Padding correction narrative (왜 필요한지 + 4-step encoding/correction) | §7.1 (확장) | (text only) |

### 외부 모델 호출 이력 (Codex + Gemini)
- **2026-05-14 세션 총 5회** (모두 사용자 명시 cross-check 요청에 따라):
  1. 아키텍처 v1 → v2 (Task 2)
  2. 초안 검토 (Task 4)
  3. 재훈련 vs 수정 결정 (Task 5)
  4. PDF 변환 진단 (Task 7)
  5. gap checklist v1 → v2 (Task 9)

### 사용자 컨벤션 (Memory 참조)
- `user_role.md` — KAU 3-1, PyTorch 친숙, 한국어 응답 선호
- `feedback_review_workflow.md` — 큰 작업 시 Codex + Gemini cross-check
- `feedback_bnn_gotchas.md` — student EMA 금지, fp16 sign 안전, T4 INT1 미지원

### 발표 narrative 흐름 (slides 기준)
1. (0:00–1:00) Motivation: 모바일/엣지에 32× 압축의 필요성
2. (1:00–4:00) Method: ReActNet-lite + double-skip + 2-phase KD
3. (4:00–7:00) Quantitative + Design discussion (AT 폭발 정직 포함)
4. (7:00–8:30) HW Demo: C 추론 25× / 12× speedup
5. (8:30–9:30) Limitations + Conclusion
6. Q&A

### 노트북 history patch 동작 (Task 12)
조원이 Cell 2 의 `exp_id` 만 바꾸면 학습 종료 시 다음이 자동 생성:
- `history_tch_<exp_id>.csv` / `.pkl` (Phase A 실제 학습 시 또는 skip 시 1-row dump)
- `history_stu_<exp_id>.csv` / `.pkl` (Phase B per-epoch 220 rows)

각 history 파일 = `epoch, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5, val_top1_ema, lr, alpha` 10 columns.

---

## 8. 이 파일을 사용하는 방법 (다음 세션에서)

다음 conversation 에서 작업을 이어가려면:

1. **"이전 작업 PROGRESS.md 읽어줘"** — 이 파일 전체 로드
2. 특정 작업 이어할 때:
   - **"§6 Phase 1 결과 통합 진행해줘"** — 조원 history 수집 후
   - **"§5 결정 사항 D14 근거 다시 설명해줘"** — 디테일 질문
   - **"EXPERIMENTS.md 진행 상황 update 해줘"** — 실험 트래킹 표 갱신
3. Memory 시스템이 자동으로 `project_cifar100_termp.md` 를 불러와 컨텍스트 보완

특히 **§5 결정 사항 트레일 (D1–D22)** 은 발표 직후 "왜 이렇게 결정했지?" 질문에 즉답 가능.
