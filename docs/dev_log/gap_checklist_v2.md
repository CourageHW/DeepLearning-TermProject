# 보고서·발표 자료 보충/수정 체크리스트 v2 (Codex + Gemini cross-check 반영)

**작성일**: 2026-05-14
**v1 → v2 변경**:
- 신규 critical 추가: **A3** (PROJECT.md outdated epoch) — Gemini 발견
- 명확화: **B2** (STE backward vs latent clamp 구분) — 두 모델 충돌 직접 검증 결과
- 신규 high 추가: **B7** (AT loss ramp timing) — Codex 발견
- 신규 medium 추가: **C9** (Padding Correction 슬라이드 narrative) — Gemini 발견
- **B3 강등** (HIGH → MEDIUM) — Gemini "학부 over-engineering" 의견 반영
- **D7 / E6 / E9 삭제** — 두 모델 합의 over-engineering

---

## 🔴 A. CRITICAL — 사실관계 불일치 (3건)

### A1 (확정 — Codex + Gemini 동의)
- **위치**: report.md §5.6 / slides.md Backup C, RandomErasing 행
- **문제**: 보고서·슬라이드에 `scale = (0.02, 0.2)` 로 명시되어 있으나 노트북 cell 5 line 194 는 `scale` 미지정 → torchvision v2 default `(0.02, 1/3)` 사용
- **권장**: 보고서·슬라이드 표를 `scale=default (0.02, 0.333)` 로 수정 (재학습 비용 회피)

### A2 (확정 — Codex + Gemini 동의)
- **위치**: report.md §6.1 (또는 §5.5)
- **문제**: 노트북 cell 2 의 `CIFAR100_MEAN/STD` 가 ImageNet 표준값 (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) 으로 misleading. CIFAR-100 native stats 와 다름. 의도 (pretrained 호환) 미설명
- **권장**: §6.1 한 줄 — "Normalize stats = ImageNet (pretrained backbone 호환). CIFAR-100 native stats 미사용"

### A3 ⭐ 신규 (Gemini 발견)
- **위치**: PROJECT.md §3.2 (와 일부 .agents/plan_v*.md 등)
- **문제**: PROJECT.md 에 `student_epochs = 300` 명시되어 있으나 실제 노트북 cell 2 line 60 은 `student_epochs = 220`. 보고서·슬라이드는 220 으로 일관. PROJECT.md 가 outdated
- **권장**: PROGRESS.md 에 한 줄 추가 OR PROJECT.md §3.2 의 `student_epochs` 수치를 220 으로 업데이트 (참조 문서 정합성)

---

## 🟠 B. HIGH — 핵심 디테일 누락 (7건)

### B1 (확정 — Codex 동의, Gemini "삭제" 의견은 §5.2 오해석)
- **위치**: report.md §5.2 / slides.md slide 6
- **검증**: §5.2 의 "warmup 15 ep + cosine 205 ep" 는 **LR scheduler** 의 warm-up. **KD 시작 epoch** 은 별도 — 노트북 cell 11 line 788: `distill_start_ep=student_warmup_epochs=15`
- **권장**: §5.2 한 줄 추가 — "KD 는 epoch 15 부터 활성화 (`distill_start_ep=15`, LR warm-up 종료 시점 일치). Epoch 0–14 는 CE-only — student stem/binary blocks 의 초기 학습 단계."

### B2 ⭐ 명확화 (두 모델 충돌 직접 검증 후)
- **위치**: report.md §4.1
- **두 모델 충돌**:
  - Codex: "매 step 후 latent weight 를 [-1,1] 로 clamp" — `cell 6 line 272-276` + `cell 11 line 789 clip_binary=True`
  - Gemini: "BinaryWeightSTE 는 STE only, clamp 없음" — `cell 8`
- **사실관계**: **둘 다 부분 정답**.
  - STE backward = identity (clamp 없음) ← Gemini 맞음
  - **하지만** 매 optimizer step 후 `m.weight.data.clamp_(-1.0, 1.0)` 별도 적용 (cell 6 `clip_binary_weights`) ← Codex 맞음
- **현재 보고서 §4.1**: "weight clip 은 하지 않으며" 표현 → **(b) 해석 ("latent weight 자체에 clip 없음") 으로는 사실 오류**
- **권장**: §4.1 wording 두 가지 clamp 의미 구분
  - "STE backward 는 identity (weight 의 gradient 자체에는 clip 없음)"
  - "다만 학습 안정성을 위해 매 optimizer step 후 latent weight 를 [-1, +1] 로 clamp" (ReActNet/Bi-Real 표준)

### B3 ⚠️ 강등 (HIGH → MEDIUM, Gemini "over-engineering" 의견 반영)
- 학부 발표 수준에서 AdamW group split 디테일은 본문 분량 차지 — footnote 정도면 충분.
- → **C 카테고리로 이동**

### B4 (확정)
- **위치**: report.md §5.2
- **문제**: `init_student_from_teacher` 가 **shape 일치 텐서만** 복사 → student stem (3×3) 은 teacher stem (7×7) 과 shape 불일치로 random init
- **권장**: §5.2 update — "teacher state-dict 중 shape 일치 텐서만 복사 (binary block conv/BN). Student stem conv (3×3) 는 teacher stem (7×7) 과 shape 불일치로 random init."

### B5 (확정)
- **위치**: report.md §3.9 / §6.2
- **문제**: Teacher 82.44 % 가 **EMA shadow evaluation 결과** 인지 raw 결과인지 미명시
- **검증**: cell 11 line 712-713: `target = shadow if (use_ema and epoch >= warmup_eps) else base` → teacher 는 use_ema=True → EMA shadow 저장
- **권장**: §6.2 한 줄 — "Teacher 82.44 % 는 EMA shadow (decay 0.999) 의 best epoch evaluation. live (raw) model 은 약 1 %p 낮은 수준 (typical EMA gap)."

### B6 (확정)
- **위치**: report.md §3 / CSV
- **문제**: CSV "SWA enable=TRUE" vs 노트북 의 custom EMA class (`torch.optim.swa_utils` 미사용) 명명 불일치
- **권장**: §3 도입부 한 줄 — "본 ablation 의 'SWA enable' 컬럼은 EMA shadow (decay = 0.999) 사용을 의미. `torch.optim.swa_utils` 가 아닌 custom EMA class (cell 10) 로 구현."

### B7 ⭐ 신규 (Codex 발견)
- **위치**: report.md §5.3
- **문제**: AT loss ramp-up 의 timing 이 모호. 실제 코드 (cell 6 line 348): `ramp = min(1.0, max(0, epoch - distill_start_ep + 1) / 10.0)` → **epoch 15–25 동안 0 → 1 linear ramp**. 보고서는 "epoch 0 ~ 10 동안 0 → 10" 으로 잘못 기술 가능
- **권장**: §5.3 wording 정확화 — "AT weight 의 점진 ramp-up: epoch 15 (KD 활성화 시작) 부터 10 epoch 동안 0 → final_weight 로 linear 증가 (cell 6 line 348). 그러나 KD switch 시점 BN running stats 불안정으로 최종 disable."

---

## 🟡 C. MEDIUM — 발표 임팩트 강화 (9건, v1 의 8건 + 신규 1건 + B3 강등)

### C1 (BitOPs 분석) — 그대로 유지 — 두 모델 모두 Top 2 로 추천
### C2 (단일 seed 명시) — 그대로
### C3 (Kaggle leaderboard) — 그대로 (있으면 추가)
### C4 (ImageNet vs CIFAR-100 footnote) — 그대로
### C5 (Param count 같음 + bit-width footnote) — 그대로
### C6 (Backbone choice rationale) — 그대로
### C7 (Mixup/CutMix 50% 확률) — 그대로 (footnote 수준)
### C8 (AT 1000 → 10 출처) — B7 과 통합
### C9 ⭐ 신규 (Gemini 발견)
- **위치**: slides.md slide 8 (C Inference Design + Speedup)
- **문제**: "PyTorch ↔ C bit-exact (max diff = 0.0000)" 의 핵심 기술 — Padding Correction 메커니즘 — 이 슬라이드에 없음. 보고서 §7.1 에는 한 줄 있으나 slide 가 빈약
- **권장**: slide 8 에 한 줄 추가 (또는 backup D 강화) — "Padding Correction: zero-pad → binary {-1,+1} 변환 시 발생하는 popcount 오차를 패딩 패턴별 lookup 으로 보정 → bit-exact 달성의 핵심 trick"

### C10 ⭐ 강등 추가 (v1 B3)
- **위치**: report.md §5.5 footnote
- **문제**: AdamW group split — 1-D parameter no-decay
- **권장**: §5.5 표 footnote (학부 발표에선 본문 X)

---

## 🟢 D. LOW — 선택적 보강 (v1 의 D1-D6 유지, D7 삭제)

D1-D6: 그대로 유지
**D7 (TensorCore 활용률) 삭제** — 두 모델 합의 over-engineering

---

## 🔵 E. 추가 실험 후보 (학부 over-engineering 항목 삭제)

### 두 모델 합의 Top 3
| 순위 | ID | 비용 | 권장 시나리오 |
|---|---|---|---|
| 🥇 | **E1** (KD on/off ablation) | 80 min 학습 + 5 min 보고서 추가 | 시간 있으면 강력 권고 — KD 효과 정량화 |
| 🥈 | **E5** (BitOPs 분석) | 30 min 종이 계산 | **즉시 가능** — §6.4 보강 |
| 🥉 | **E4** (Class-wise / confusion) | 10 min `inference_c.py --compare` 실행 | **즉시 가능** — 정성 분석 |

### Future Work / Defer 처리 (삭제 또는 minor 언급)
| ID | 처리 |
|---|---|
| **E2** (double-skip on/off) | Future work — 시간 남으면 E1 이후 |
| **E3** (RPReLU vs PReLU) | Future work |
| **E6** (multi-seed variance) | ❌ **삭제** — 두 모델 합의 학부 over-engineering |
| **E7** (Mixup α sweep) | Future work |
| **E8** (KD T sweep) | Future work |
| **E9** (Bit-width Pareto) | ❌ **삭제** — 두 모델 합의 scope 이탈 (별도 보고서급) |

---

## 📋 적용 우선순위 — 확정안

### Phase 1 (즉시 적용, 15-20분)
1. **A1, A2, A3** — 사실관계 수정 (CRITICAL)
2. **B1, B2, B4, B5, B6, B7** — HIGH 6건 텍스트 보강
3. **C9** — slide 8 Padding Correction 한 줄
4. **C1 (BitOPs)** — 즉시 가능

### Phase 2 (시간 있으면, +10–15 분)
5. **C2–C8, C10, D1–D6** — MEDIUM/LOW 보강

### Phase 3 (학습 자원 + 80 분 있으면)
6. **E1 (KD on/off)** — student 1회 재학습 → §6 강화

---

## Risk markers ledger
- **Risk markers touched**: schema (보고서 wording 정정), numeric (binary clamp 의미 구분)
- **External call used**: yes — Codex + Gemini 병렬 cross-check (v1 → v2)
- **두 모델 충돌 해결**: B1 (Gemini 오해석) / B2 (둘 다 부분 정답 → 명확화)
