# CIFAR-100 텀프로젝트 Plan v2 — Single-Session A1W1 + KD (Codex 반영)

## 0. 변경 요약 (v1 → v2)
1. **Kaggle single-session 제약**: teacher/student 한 노트북에서 순차 실행. teacher는 ImageNet pretrained → CIFAR-100 fine-tune으로 빠르게.
2. **KD 유지** (T=4, w=0.7). Codex 권고대로 KD ablation은 본 학습 후 sanity 1회만.
3. **Target 조정**: 메인 A1W1 Top-1 **68–70%** (정직), 70% 달성하면 plus. Plan B = W2A2 LSQ로 스위치하면 70–73% 확보.
4. **모델 보강 (Codex 지적)**: RPReLU, double-skip, gradient clipping, autocast-내 sign ops 안전화.
5. **부수실험 축소**: Pareto + HW 중심, KD ablation은 1회 (KD on/off).

## 1. 단일 세션 파이프라인 (총 ~8h on Kaggle T4×2)

```
[Phase A] Teacher Quick FT      ImageNet pretrain ResNet18 → CIFAR-100  (80 ep, ~1.5h)
                ↓ (teacher_fp32.pth in memory, no external file)
[Phase B] Student A1W1 KD       Init from teacher → A1W1 train + KD     (220 ep, ~5.5h)
                ↓ (best_a1w1_resnet18.pth)
[Phase C] Evaluate + Submit     val Top-1/5 + test predictions          (~30 min)
                ↓
[Phase D] Pareto/HW (별도 session)  W4A4/W2A2 sweep + HW 표        (다음 세션들)
```

## 2. Phase A — Teacher (FP32 ResNet18, 80 ep, ~1.5h)
- 모델: `resnet18(weights=ResNet18_Weights.DEFAULT)` + CIFAR stem (conv1 stride 1, maxpool Identity) + fc(100).
- Optimizer: AdamW lr 1e-3, wd 1e-4, warmup 8 ep (linear 0.1→1.0) + Cosine to 1e-6.
- Aug: Pad+RandomCrop + HFlip + RandAugment(N=2, M=9) + RandomErasing(p=0.2) + Mixup/CutMix α=1.0 (절반 확률, step decay), label smoothing 0.05.
- AMP on, batch 1024.
- EMA on (decay 0.999), SWA off (시간 절약).
- 종료 후: EMA 가중치를 in-memory `teacher_model` 변수에 보관 (디스크 저장 옵션도 추가).
- **목표: val Top-1 74–76%.** (300ep full 76–78% 대비 약간 손해지만 KD source로 충분.)

## 3. Phase B — Student A1W1 (Single-stage, 220 ep, ~5.5h)
v1의 2-stage(W32A1 → W1A1)는 Kaggle 시간 압박상 single-stage로 단순화. 대신 코드 보강으로 정확도 확보.

### 3.1 모델: A1W1ResNet18 보강 (`A1W1ResNet18v2`)
- 기존 RSign + α-only Conv 위에 다음 추가:
  - **RPReLU**: `PReLU(x − γ_c) + β_c`. RSign 직전의 분포 reshape용.
  - **Double-skip**: BasicBlock 출력에 conv1·conv2 residual 외에 RPReLU 후 한번 더 identity add.
  - 블록 구조:
    ```
    x → BN → RSign → BinConv(W1) → BN → RPReLU →  ┐
                                                  ⊕ shortcut1
                                                  ↓
        BN → RSign → BinConv(W1) → BN → RPReLU →  ┐
                                                  ⊕ shortcut2 (double-skip)
                                                  → next block
    ```
  - Stem conv1, fc는 FP32 유지.
- `alpha = weight.abs().mean(...).detach()` — Codex 지적 반영 (gradient flow는 sign STE를 통해서만).

### 3.2 학습 레시피
- Init: `init_xnor_from_teacher(student, teacher_model)`로 동일 shape param 복사.
- Optimizer: AdamW lr 5e-4, wd 0 (binary는 wd 0이 표준). RPReLU/RSign threshold는 별도 param group, lr same, wd 0.
- Scheduler: Cosine 220 ep, warmup 15 ep (linear 0.1→1.0), eta_min 1e-6.
- Loss: CE(label smoothing 0) + KD(KL-div, T=4, w=0.7), KD는 warmup 이후 시작.
- AMP on, but **binary sign/RPReLU/STE 연산은 autocast disabled** (수치 안정성 — Codex 지적 반영). 즉:
  ```python
  with torch.amp.autocast('cuda'):
      x = self.conv1(x)  # fp16
      x = self.bn1(x)
  with torch.amp.autocast('cuda', enabled=False):
      x = self.rsign(x.float())
      x = binary_conv(x)
      x = rprelu(x.float())
  ```
- `clip_grad_norm_(params, max_norm=5.0)` (Codex 지적 반영).
- `clip_xnor_weights` 매 step 후 (기존).
- Mixup α 0.4 → 0 step decay, CutMix 같이 (mixup-only / cutmix-only ablation은 부수실험).
- EMA on student (decay 0.999), `eval`은 EMA 위주, best val acc → save.
- Early stop patience 60.

### 3.3 목표: val Top-1 68–70% (KD 포함, EMA 평가)

## 4. Phase C — 평가/제출
- Best EMA checkpoint 로드 → test set 추론 → `submission.csv`.
- val Top-1/5, test Top-1 기록.

## 5. Phase D — 부수실험 (별도 Kaggle 세션 1–2회)

### 5.1 Bit-width Pareto (단일 노트북, 4 bit 설정 sweep, 각 150ep)
- FP32 teacher 고정 (Phase A로부터 in-memory).
- Student bit-width: W8A8 / W4A4 / W2A2(LSQ) / W1A1.
- 모두 같은 init/recipe, KD on. 시간 절약 위해 150ep로 통일.
- 출력: 모델별 Top-1, Top-5, params×bit memory, BitOPs.

### 5.2 KD ablation (sanity, 메인 모델로 KD on/off 2회, 100ep)
- 같은 A1W1 보강 모델, KD off 1회 + KD on 1회 비교 (시드 동일).

### 5.3 HW inference 추정
- BitOPs = FLOPs × W_bit × A_bit + FP stem/classifier/BN의 FP ops.
- Weight memory = Σ params × bit / 8 (FP32 stem/fc는 FP32로 계산).
- Activation memory peak (per layer) 계산.
- PyTorch unaccelerated latency 측정 (FP32 vs W1A1 모듈 forward time, CPU/GPU). XNOR 가속 부재 명시.
- Optional ONNX export: FP32 vs W8A8 INT8 ORT-CPU 비교 (1-bit은 ORT 미지원).
- 표: 모델별 Acc / params bytes / activation bytes / BitOPs / unaccel latency.

## 6. Plan B (메인이 70% 미달일 때)
- A1W1 < 68%: 같은 노트북에서 `model_name = "LSQW2A2ResNet18"` 으로 스위치, teacher init 그대로 사용. 150ep 추가 학습으로 70–72% 확보.
- 보고서: "1-bit 한계 식별 → 2-bit fallback의 sweet spot 확인" 스토리.

## 7. 산출물
1. 학습 노트북 (Phase A+B+C 한 번에).
2. 부수실험 노트북 (Pareto + HW).
3. `submission.csv`.
4. `pareto.csv`, `hw_estimate.csv`, `kd_ablation.csv`.
5. 발표 슬라이드.

## 8. 시간 예산 (Kaggle T4×2 기준)
| Phase | Epochs | 예상 시간 |
|---|---|---|
| A. Teacher FT | 80 | ~1.5h |
| B. A1W1 KD | 220 | ~5.5h |
| C. Eval+submit | — | ~0.5h |
| **Session 1 total** | | **~7.5h** (12h 한도, 4.5h 여유) |
| D1. Pareto sweep (4 runs × 150ep) | 600 | ~10h → 별도 세션 2회 분할 |
| D2. KD ablation (2 × 100ep) | 200 | ~3h |
| D3. HW 측정 | — | ~1h |

## 9. Risk / Mitigation
| Risk | Mitigation |
|---|---|
| Kaggle 세션 끊김 | epoch마다 checkpoint, resume 지원 |
| A1W1 < 68% | Plan B 자동 발동 (W2A2 LSQ swap) |
| RPReLU 수치 불안정 | autocast 비활성, grad clip 5.0 |
| Teacher quick FT 정확도 부족 | warmup만 충실히, EMA 사용. 70% 안 나오면 PReLU 안 쓰고 GELU 옵션 |

## 10. Codex 검토 반영 체크리스트
- [x] 70% → 68–70% honest target
- [x] RPReLU 코드 추가 예정
- [x] double-skip wiring 코드 추가 예정
- [x] grad clip 추가 예정
- [x] alpha detach 수정 예정
- [x] autocast 안 sign ops disable 예정
- [x] EMA on student 추가
- [x] KD ablation 4 → 1로 축소 (Codex 권고)
- [x] HW: BitOPs + memory + unaccel latency 분리 보고
- [x] W2A2 method = LSQ (DSQ cut)
- [ ] RSign + mixup risk: 코드에 mixup α 낮춤(0.4) + ablation 1회로 대응
