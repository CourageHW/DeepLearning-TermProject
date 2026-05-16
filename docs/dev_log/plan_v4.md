# Plan v4 — Post-First-Run (W1A1 = 68.72%)

## 0. 현재 위치
- **Student W1A1 best val = 68.72%** (literature SOTA 영역: ReActNet 69.4% / ReCU 69% / RBNN ~68%)
- Teacher FP32 = 82.44% (이미 충분히 강함)
- 70% target까지 1.28%p
- 1-bit BNN 분야 자체 천장이 ~71% 부근 → 더 짜낼 여지는 좁지만 있음

## 1. 트랙 분리 — 정확도 vs 발표 임팩트

남은 시간을 두 트랙에 나눠 투자해야 함. 각 트랙은 독립이라 병렬 진행 가능.

### Track A — 정확도 push (1.28%p 노림)
68.72% → 70%+ 도달 시도. 작은 트릭 여러 개 누적.

### Track B — 발표 infrastructure
정확도와 무관하게 발표 임팩트 키우는 자산. 68.72%로도 충분히 강한 발표 가능.

---

## 2. Track A — 정확도 push (옵션 ranked by ROI)

| # | 옵션 | 예상 이득 | 코드 시간 | 학습 시간 | 누적 risk |
|---|---|---|---|---|---|
| **A1** | **Longer training (220 → 300ep)** | +0.3-0.8% | 1줄 | +25분 | 거의 0 |
| **A2** | **Feature KD (FitNets-style hint)** | +0.5-1.5% | ~30줄 | 동일 | 낮 |
| **A3** | **2-stage training (W32A1 → W1A1)** | +0.5-1.5% | ~100줄 | 동일/+30분 | 중 |
| A4 | SAM optimizer | +0.5-2% | ~30줄 | ×1.6 학습 시간 ↑ | 중 |
| A5 | Stronger teacher KD (T sweep, weight sweep) | +0.2-0.5% | 10줄 | 동일 | 낮 |

### 권장 조합: A1 + A2

- **A1 (epoch 220→300)**: 그냥 epoch 늘리면 cosine schedule이 더 많이 cool down → 추가 +0.3-0.8%. 거의 무료.
- **A2 (Feature KD)**: 이미 teacher가 in-memory. 학생 layer4 출력과 teacher layer4 출력을 1×1 projection MLP로 align → L2 loss 추가. 표준 BNN 트릭, 위험 낮음.

두 개 합치면 **+1.0~2.0% 기대** → 70% 도달 가능성 ~70%.

### 구현 우선순위
1. **A1만 먼저 시도** (30분 추가 학습): 베이스라인 어디까지 가는지 확인
2. A1으로 70% 안 넘으면 **A2 추가**: 1-2시간 코드 + 학습 재시작
3. 그래도 모자라면 **A4 (SAM)** 시도: BNN×SAM은 발표 narrative도 좋음

## 3. Track B — 발표 infrastructure

| # | 자산 | 효과 | 시간 |
|---|---|---|---|
| **B1** | **AVX CPU 실측 demo** (plan_avx_demo.md) | "FP32 대비 1.5-5× speedup 실측" 그래프 | 1일 |
| **B2** | **BitOPs / 메모리 / params 분석표** | "32× 압축, 이론 32× 가속" 표 | 2-3h |
| **B3** | **Bit-width Pareto** (W8A8 LSQ, W4A4 LSQ 추가 학습) | 정확도-비트 trade-off 곡선 | 1일 (학습 시간 포함) |
| **B4** | **KD ablation (on/off 1회)** | "KD가 얼마나 기여했는지" 1줄 정량화 | +3h 학습 1회 |
| **B5** | 학습 곡선 plot + 발표 슬라이드 | 결과 정리 | 3-4h |

### 권장 조합: B2 + B5 필수 / B1 추천 / B3, B4는 시간 남으면

- **B2 (분석표)**: 코드 없이 계산만으로 됨. 발표 임팩트 큼 ("실측 못 보여줘도 이론은 정확").
- **B5 (정리)**: 발표 1주일 전부터 해야 함. 마지막에 몰아서 X.
- **B1 (AVX demo)**: Codex 검토 후 scope 축소 (Level 1만). 1일 투자 → 실측 그래프 1장으로 차별화 강력.
- **B3 (Pareto)**: 학습 1-2회 추가 필요. 시간 있으면 가치 큼. LSQ 코드는 기존에 작성했다가 cleanup으로 빠짐 → 복원 필요.
- **B4 (KD ablation)**: KD off로 1회 재학습. 결과 비교 한 줄 표.

## 4. 시간 예산 시나리오 (남은 시간별)

### 시나리오 A — **3일 남음**
- Day 1: A1+A2 학습 (70% 도전) + B5 슬라이드 골격
- Day 2: B2 분석표 + AVX Level 1 (B1)
- Day 3: 발표 자료 마무리

### 시나리오 B — **1주 남음**
- Day 1: A1 학습 (+30분)
- Day 2-3: A2 (Feature KD) 구현 + 학습
- Day 4: B1 (AVX demo)
- Day 5: B2 (분석표) + B3 (Pareto 1-2개)
- Day 6-7: B5 (발표 자료)

### 시나리오 C — **2주+ 남음**
- 전부 다 가능 (A1+A2+A4+B1+B2+B3+B4+B5)
- 단, A4 (SAM)는 ROI 낮은 편이라 시간 남으면 시도

## 5. Plan B (지금 시점에 와도 늦지 않음)
- A1+A2가 70% 못 넘기면: **68.72%로 정직하게 보고** + "literature SOTA 영역, BNN field ceiling 근처" narrative
- Track B만으로도 충분히 강한 발표 가능
- 발표 임팩트 = (정확도 1.5%) + (HW demo 차별화 8.5%) — 후자가 훨씬 큼

## 6. 즉시 결정 필요 사항
1. **남은 시간** — 시나리오 A/B/C 중 어디?
2. **70% 달성에 얼마나 집착?** — 거의 도달 or 무조건 달성?
3. **A2 (Feature KD) 구현 의향?** — 일반적인 트릭이지만 코드 작성 필요
4. **AVX demo 진행?** — B1 / Codex가 Level 1만 권장

## 7. 추천 우선순위 (시나리오 B 기준)

```
Week start:
  1. A1: epoch 300 재학습 (백그라운드, 30분 추가) → 결과 확인
  2. (if < 70%) A2: Feature KD 코드 작성 → 재학습 1회 (1일)
  3. B2: 이론 분석표 작성 (3h)
  4. B1: AVX demo Level 1 (1일)
  5. B5: 발표 자료 작성 (1-2일)
  6. (시간 남으면) B3 Pareto, B4 KD ablation
```
