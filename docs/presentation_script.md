# 발표 대본 — CIFAR-100 1-bit BNN (Team 16)

**대상 슬라이드**: `slides_native.pdf` (28 페이지)
**총 길이**: 약 11분 (10–11분 목표, 1인 화자 기준 / 팀 분담 시 마커 참고)
**Status**: 최종본 v2 — Codex + Gemini-flash cross-check 2회 반영

---

## 🎯 핵심 메시지 (발표 전체를 관통하는 한 줄)

> **"정확도 13%p 손실로 메모리 32배 압축과 CPU 21배 추론 가속을 동시에 달성한, 학습–ablation–CPU 실측까지 한 파이프라인으로 닫은 1-bit BNN 프로젝트"**

---

## p1. 표지 (≈10초)

> 안녕하세요. 딥러닝 텀프로젝트 16조 발표를 시작하겠습니다.
> 저희 주제는 **CIFAR-100을 1-bit Binary Neural Network로 학습하고, 실제 CPU에서 추론 가속까지 측정한 프로젝트**입니다.

---

## p2. Motivation & Goal (≈55초)

> 최근 클라우드에 의존하지 않는 **on-device AI** 수요가 빠르게 커지고 있지만, 모바일·노트북·엣지 디바이스의 메모리와 전력 예산은 모델 크기를 따라가지 못합니다.
> 예를 들어 ResNet18에 CIFAR-100 분류 헤드를 붙이면 1100만 파라미터, FP32로 약 **44MB**가 필요합니다.
> 저희 해결책은 가장 극단적인 양자화인 **Binary Neural Network**입니다. 가중치와 활성을 ±1, 즉 1-bit로 양자화하면 이론상 32배 메모리 압축이 가능하고, 곱셈 누산을 **XNOR + POPCNT** 비트 연산으로 대체할 수 있습니다.
> 저희 프로젝트의 차별점은 SOTA 정확도 추격이 아니라, **학습 → 체계적 ablation → CPU 실측까지 하나의 파이프라인으로 닫았다**는 것입니다. 구체적으로 세 가지를 진행했습니다 — 첫째 CIFAR-100 W1A1 BNN 학습, 둘째 teacher를 7단계 ablation으로 강화한 뒤 KD로 student 정확도를 끌어올리기, 셋째 **POPCNT 기반 C 추론 커널을 직접 작성**해서 노트북 CPU에서 실제 가속을 실측하는 것입니다.

**[강조]** 마지막 문장 — "파이프라인 closure"가 우리만의 novelty.

---

## p3. Method Overview (≈35초)

> 모델은 **ReActNet-lite**를 채택했습니다. 1-bit 양자화는 표현력을 잃을 수밖에 없는데, 이를 보완하는 세 가지 장치를 표준 ResNet18에 이식했습니다.
> 학습 가능한 sign threshold인 **RSign**, 학습 가능한 활성 함수 **RPReLU**, 그리고 Bi-Real의 **double-skip** 잔차 연결입니다. 모두 1-bit으로 잃은 정보를 복구하는 보정 장치라고 보시면 됩니다.
> 구성은 Teacher–Student 두 모델입니다. Teacher는 FP32 ResNet18을 ImageNet pretrained에서 CIFAR로 fine-tune한 것이고, Student는 이 teacher로부터 KD받은 1-bit 모델입니다. Student는 stem과 분류기만 FP32로 남기고 본체는 모두 binary입니다.

**[화자 교대 가능 지점]**

---

## p4. Teacher: Ablation Study 개요 (≈35초)

> Teacher는 단순 fine-tune이 아니라 **7단계 ablation**으로 정확도를 한 단계씩 끌어올렸습니다.
> 출발점은 baseline ResNet18 from-scratch 학습 75.74%, 단계별로 stem 구조, optimizer, batch/LR, scheduler, activation, mixup decay, RandAugment를 최적화했습니다.
> 슬라이드에 두 가지 숫자가 등장합니다 — ablation 환경에서의 최고치 **83.12%**, 그리고 최종 제출에 사용한 **Phase A teacher 82.44%**. **실제 KD에 사용한 teacher는 82.44% 한 개**이고, 83.12%는 ablation setting에서의 참고 수치입니다. 이 차이는 RandAugment 슬라이드에서 다시 짚겠습니다.

**[중요]** 한 줄로 고정: "ablation 최고치 83.12 / 최종 사용 teacher 82.44 — 환경이 다름"

---

## p5. Step 1 — Stem adaptation (≈30초)

> Step 1은 stem 구조입니다. ImageNet 원본 stem은 7×7 stride 2 conv에 max pooling까지 붙어있어서 32×32 입력에는 다운샘플링이 과합니다. 네 가지 변형을 비교했고, **최종 채택은 D안 — Conv 7×7 stride 1에 max pool 제거**입니다.
> 핵심은 두 가지로, 공간 해상도를 32×32 그대로 layer1까지 유지하고, ImageNet pretrained 7×7 weight를 재사용한 것입니다. 학습 시간은 1.8배 늘었지만 정확도는 **+10.92%p**, ablation 7단계 중 단일 최대 기여입니다.

**[강조]** "+10.92%p 단일 최대 기여" — 학부 청중에게 가장 직관적인 lever.

---

## p6. Step 2 — Optimizer (≈12초)

> Step 2부터는 teacher recipe tuning입니다. Optimizer 8개 비교에서 **AdamW weight decay 1e-4**가 82.38%로 최고였습니다.

---

## p7. Step 3 — Batch / LR (≈12초)

> Batch와 LR sweep에서 **1024 / 1e-3** 채택. 512와 거의 차이 없지만 학습 시간 효율이 더 좋습니다.

---

## p8. Step 4 — Scheduler (≈12초)

> Scheduler는 **CosineAnnealingLR**. WarmRestarts는 reset마다 흔들려서 1%p 손해입니다.

---

## p9. Step 5 — Activation (≈20초)

> Activation에서 GELU가 82.46%로 가장 높았지만, **ReLU 82.38%를 채택**했습니다. ImageNet pretrained ResNet18이 ReLU 기반이라 호환성을 우선했고, 0.08%p 차이는 noise 범위입니다.

---

## p10. Step 6 — Mixup Decay (≈12초)

> Mixup α를 학습 후반부로 갈수록 줄이는 **step decay**가 82.54%로 최고입니다.

---

## p11. Step 7 — RandAugment (≈25초)

> 마지막은 RandAugment 강도입니다. magnitude 10이 83.12%로 ablation 최고치였지만, magnitude 9 82.58% 대비 over-regularization 위험이 있다고 보고 **보수적으로 mag 9를 채택**했습니다.
> 그리고 앞서 짚은 대로 — 최종 Phase A teacher는 ablation과 환경이 다릅니다. **pretrained 가중치 + 80 epoch fine-tune + RandAug mag 9** 조건에서 **82.44%**가 KD에 들어간 진짜 teacher입니다.

**[종결]** 82.44 ↔ 83.12 차이 종결.

---

## p12. Student Architecture — KD/Double-skip/RPReLU Ablation (≈55초) ⭐

> 이제 Student 쪽으로 넘어갑니다. 저희가 가장 신경 쓴 부분이 바로 이 ablation 표입니다.
> Baseline은 **KD + double-skip + RPReLU** 세 컴포넌트를 모두 켠 구성으로 best validation top-1 **68.74%**입니다. 그리고 한 번에 하나씩 제거해서 영향을 측정했습니다.
> KD를 끄면 -2.30%p, double-skip을 single-skip으로 바꾸면 -2.46%p, 그리고 **RPReLU를 제거하면 -9.22%p**로 정확도가 무너집니다.
> 즉 1-bit BNN에서 단일 최대 기여 컴포넌트는 **RPReLU**입니다. 1-bit activation으로 잃은 정보를 channel-wise learnable shift로 복구하는 역할이 결정적이라는 뜻입니다.

**[강조]** "RPReLU 단일 -9.22%p" — 발표 전체에서 가장 강한 ablation 메시지.

---

## p13. Student Architecture Overview (≈30초)

> 구조는 표준 ResNet18을 1-bit화한 형태입니다. Stem과 layer1–4, BN+GAP, FC까지 ResNet18의 stage 구성을 그대로 따랐고, **layer 내부의 binary residual block 16개만 1-bit**로 바꿨습니다.
> 메모리 예산을 보면 — **binary body가 1.39MB, FP32 stem과 분류기가 0.48MB, 합쳐서 약 1.87MB**입니다. FP32 teacher 44MB 대비 약 **23배 압축**, binary body만 기준으로 보면 **32배 압축**입니다. 슬라이드에 표시된 1.4MB는 binary body 부분에 가까운 표기로 봐주시면 됩니다.

**[정직]** "32× 메모리 압축"은 binary body 기준, 전체는 23× — 둘 다 솔직히 제시.

---

## p14. BasicBlock — double-skip variant (≈35초)

> Binary residual block 한 개를 자세히 보면 — **BN → RSign → BinConv → BN → RPReLU** 구조를 두 번 반복하고, 그 사이마다 두 개의 residual 연결이 있습니다.
> 핵심은 **각 BinConv 뒤에 별도의 shortcut**이 붙는다는 점입니다. 첫 번째 residual_a는 stride나 채널 변경 시 average pool + zero-pad shortcut이고, 두 번째 residual_b는 identity shortcut입니다. 1-bit conv가 정보를 잃으면 그때마다 FP32 shortcut으로 한 번씩 보충해주는 구조라고 보시면 됩니다.
> 이 double-skip이 Bi-Real Net의 핵심 트릭이고, 저희 ablation에서 -2.46%p 기여를 확인했습니다.

---

## p15. Student Parameters — Phase A / B (≈25초)

> 학습은 두 단계입니다. **Phase A teacher fine-tuning** 80 epoch, AdamW wd=1e-4, batch 1024.
> **Phase B student** 220 epoch, AdamW wd=0, KD는 temperature 4, weight 0.7, warmup 종료 시점인 **epoch 15부터 활성화**됩니다.
> Student 학습에서는 **EMA가 비활성**입니다. Binary weight 평균은 sign을 부숴서 random 모델이 되기 때문입니다. Mixup α는 0.4로 teacher보다 약하게, grad clip은 5.0으로 STE jump를 안정화합니다.

---

## p16. Training Pipeline — Augmentation 요약 (≈20초)

> Augmentation은 image-level 4개 (RandomCrop, HorizontalFlip, RandAugment, RandomErasing) + sample-level 2개 (Mixup, CutMix)입니다. Normalize는 ImageNet 통계를 그대로 써서 pretrained 호환성을 유지합니다.

---

## p17–18. Mixup/CutMix · RandomErasing/HorizontalFlip 시각 예시 (합쳐 ≈25초)

> Mixup은 두 이미지를 픽셀 단위로 blending, CutMix는 패치를 교체합니다. 서로 다른 invariance를 학습시키고, RandomErasing과 HorizontalFlip은 occlusion과 좌우 대칭 robustness를 보강합니다. **1-bit 모델의 제한된 capacity를 augmentation으로 보완하는 게 핵심**입니다.

---

## p19. Single Kaggle Session (≈25초)

> 학습 환경은 단일 Kaggle T4 12시간 budget입니다. **Phase A 약 15분 + Phase B 약 80분, 총 1시간 35분**에 완료됩니다.
> Phase B 핵심 hyperparameter 하나만 강조하면 — **매 step latent weight를 [-1, +1] 구간으로 clamp**해서 sign STE의 dynamic range를 보호하는 부분입니다. 이게 없으면 학습 후반에 weight가 발산합니다.

---

## p20. Training Curve — Teacher (≈15초)

> Phase A teacher의 80 epoch 학습 곡선입니다. EMA shadow가 validation을 약 1%p 더 높게 안정화시키는 게 보입니다.

---

## p21. Training Curve — Student (≈20초)

> Phase B student의 220 epoch 학습 곡선입니다. **Epoch 15에서 KD가 켜지면서 validation이 한 번 꺾이는 게** 특징이고, train보다 val이 위쪽인 mixup/cutmix의 영향도 그대로 보입니다.

---

## p22. Quantitative Results (≈70초) ⭐⭐

> 결과를 정리하면 — **Teacher는 82.44%에 44MB, Student는 69.06%에 1.4MB**입니다.
> **정확도는 약 13%p 손실**, 그러나 **메모리는 약 30배 압축**. 이게 본 프로젝트의 핵심 tradeoff입니다.
> Literature와 비교해보면 — XNOR-Net이 ImageNet에서 51.2%, Bi-Real이 56.4%, ReActNet이 69.4%, ReCU가 66.4%이고, CIFAR-100에서 가장 가까운 RBNN이 67.1%입니다. 저희는 CIFAR-100에서 **69.06%**로 RBNN 대비 약 +2%p 우위, ImageNet ReActNet 수준의 영역에 도달했습니다.
> 한 가지 주의 — **ImageNet과 CIFAR-100은 직접 비교가 아니라 W1A1이라는 같은 난이도 카테고리 안에서의 reference point**로 봐주시면 됩니다.

**[강조]** "13%p 정확도 손실로 30배 메모리 압축" — 전체 메시지의 첫 번째 절반.

---

## p23. Design Choices & Ablation 정리 (≈35초)

> 몇 가지 디자인 결정을 강조합니다.
> 첫째, **FP32 stem과 분류기는 전체 파라미터의 1%만 차지하면서 정확도를 5–10%p 보존**합니다. 200KB 메모리 부담이라면 충분히 가치 있는 선택입니다.
> 둘째, AMP fp16과 1-bit STE가 함께 안정적으로 작동해서 학습 속도가 1.5–2배 빨라집니다.
> 셋째, 앞서 본 ablation에서 **RPReLU가 -9.22%p로 단일 최대 기여**입니다.
> **이제, 정확도와 메모리에서 얻은 이 tradeoff가 실제 추론 속도로 이어지는지 보여드리겠습니다.**

**[브릿지]** 마지막 문장 — p24로 자연스럽게 연결.

---

## p24. C Inference — Wall-clock Latency (≈60초) ⭐⭐

> 하드웨어 추론 결과입니다. **Intel Core i5-1135G7 노트북 CPU**에서 PyTorch와 bit-exact 일치하는 C 커널로 실측했습니다.
> Single thread에서 student는 24.65ms, teacher FP32는 522.93ms — **21.21배 speedup**입니다. 4 thread에서는 11.87ms 대 150.49ms로 **12.68배**입니다.
> 왜 thread를 늘리면 speedup ratio가 줄어들까요. **Teacher는 큰 GEMM 루프 하나라 multi-thread로 약 3배 가속이 되는 반면, student는 작은 binary conv 16개로 쪼개져 있어서 thread spawn overhead 때문에 1.5배 정도만 가속**됩니다. 결과적으로 4 thread에서는 teacher가 더 큰 이득을 보면서 ratio가 21→12배로 줄어듭니다.
> 두 수치 모두 정직한 측정값입니다. **비교 조건은 single-thread, PyTorch와 bit-exact, AVX-512 VPOPCNTDQ나 SIMD GEMM 같은 production 최적화는 미적용**입니다.

**[강조]** "13%p 손실 vs 32× 메모리 + 21× 속도" — 발표 전체 메시지의 두 번째 절반 완성.

---

## p25. C Inference Demo (≈15초)

> 왼쪽이 FP32 teacher, 오른쪽이 1-bit student의 C 추론 실시간 데모 화면입니다. 같은 입력에 대해 bit-exact로 일치하는 것을 확인할 수 있습니다.

---

## p26. Limitations & Honest Discussion (≈30초)

> 정직한 한계입니다.
> 첫째, Kaggle T4 GPU는 native INT1 연산을 지원하지 않습니다. Turing 아키텍처는 INT4까지만 지원하기 때문에 GPU에서 binary 가속은 불가능하고, 그래서 **CPU C 데모로 우회**했습니다.
> 둘째, 저희 C 커널은 scalar POPCNT만 사용하고 AVX-512 VPOPCNTDQ나 SIMD GEMM 같은 production-grade 최적화는 미적용입니다. 따라서 21배 수치는 알고리즘 차이이지 production-grade 비교는 아닙니다.
> 셋째, ablation은 단일 컴포넌트만 진행했고 cross-component interaction이나 KD weight sweep은 미실시입니다.

---

## p27. Conclusion + Future Work (≈45초)

> 핵심 성과 세 가지로 정리합니다.
> 첫째, **CIFAR-100 W1A1 BNN에서 69.06% top-1**, literature SOTA 영역에 도달했습니다.
> 둘째, **메모리 32배 압축** (44MB → 1.4MB)과 함께 teacher empirical study로 baseline **75.7%에서 83.1%까지 끌어올린 체계적인 ablation**을 검증했습니다.
> 셋째, **POPCNT 기반 C 추론 데모**가 PyTorch와 bit-exact로 일치하면서 single-thread에서 21배 speedup을 실측했습니다.
> 이번 프로젝트의 가장 큰 의의는 **학습 → 체계적 ablation → CPU 실측까지 한 파이프라인 안에서 닫았다는 점**이라고 생각합니다.
> Future work로는 cross-component ablation, BNN × SAM optimizer, AVX-512 VPOPCNTDQ로 production-grade 커널 확장 등을 계획하고 있습니다.

**[강조]** "파이프라인 closure" novelty 재명시.

---

## p28. Thank You (≈10초)

> 발표 들어주셔서 감사합니다. 질문 받겠습니다.

---

## ⏱️ 시간 배분 합산

| 구간 | 페이지 | 누적 시간 |
|---|---|---|
| 인트로 | p1–p3 | 1:40 |
| Teacher ablation | p4–p11 | + 2:35 → **4:15** |
| Student 핵심 | p12–p15 | + 2:25 → **6:40** |
| Training detail | p16–p21 | + 2:00 → **8:40** |
| Results + Speedup ⭐ | p22–p24 | + 2:45 → **11:25** |
| Wrap-up | p25–p28 | + 1:40 → **13:05** |

**전체 약 11–13분** — 발표 압박이 심하면 p6–p10을 표 가리키며 더 빠르게 (각 8초), p17–p18 단축, p25는 한 문장으로 처리하면 10:30까지 줄일 수 있습니다.

---

## 🎤 Q&A 대비 멘트 (별도 준비)

| 예상 질문 | 답변 힌트 |
|---|---|
| "69.06%는 너무 낮지 않나?" | FP32 대비 13%p 손실이지만 **32× 메모리 압축 + 21× CPU speedup**의 tradeoff. CIFAR-100 W1A1 literature 영역(RBNN 67.1%) 안에 안전하게 들어옴 |
| "왜 stem과 분류기는 FP32?" | 전체 파라미터 1%만 차지, 정확도는 +5–10%p 보존 — BNN 표준 관행 (XNOR-Net, ReActNet, Bi-Real 동일) |
| "RPReLU 왜 그렇게 중요?" | 1-bit activation으로 잃은 정보를 channel-wise learnable shift β로 복구. ablation에서 -9.22%p로 단일 최대 기여 |
| "21배 speedup 공정한 비교인가?" | Single-thread / bit-exact / naive C 조건. 알고리즘 차이가 메인. AVX-512나 BLAS 적용 시 ratio는 더 줄어듭니다 (3×대) — 정직한 제시 |
| "Ablation에 83.12% 있는데 왜 82.44% 사용?" | Ablation은 from-scratch 150 epoch / Phase A는 pretrained FT 80 epoch — **환경이 다릅니다**. 보수적으로 RandAug mag 9 채택해서 82.44% 도착 |
| "메모리 1.4MB vs 1.87MB 어느 게 맞나?" | Binary body 1.39MB + FP32 1% 0.48MB = **1.87MB가 전체**. 1.4MB는 binary body 부분 표기. 32× 압축은 binary body 기준, 전체 기준은 23× |
| "EMA를 student에서 안 쓰는 이유?" | Binary weight 평균은 sign을 부숴서 random 모델이 됨 — ReActNet/Bi-Real도 동일 |
| "T4 GPU에서 왜 안 돌렸나?" | Turing 아키텍처는 INT4까지만 native 지원, INT1은 미지원. 그래서 CPU C 데모로 우회 |
| "KD weight 0.7은 어떻게 정했나?" | Hinton 원논문 기준 + 학부 텀프 범위라 sweep은 미실시 — limitation에 명시 |
| "EMA shadow 표기와 SWA enable=TRUE 충돌?" | CSV의 SWA enable=TRUE는 EMA shadow(decay=0.999)를 의미하는 custom 구현. torch.optim.swa_utils API 아닙니다 |

---

## ✏️ 발표 직전 체크리스트

- [ ] **숫자 4개 확실히 외우기**: 69.06% / 82.44% / 21.21× / 32×
- [ ] **차이 표현 한 줄**: "83.12는 ablation, 82.44는 최종 KD teacher"
- [ ] **tradeoff 한 줄**: "정확도 13%p 손실로 메모리 30배·속도 21배"
- [ ] **novelty 한 줄**: "학습–ablation–CPU 실측을 한 파이프라인으로 닫음"
- [ ] **시간 timer**: p22와 p24에 가장 많이 할애 (각 60–70초)
- [ ] Q&A 대비: 위 표 상단 5개 (69%/FP32 stem/RPReLU/21× 공정성/82.44 vs 83.12) 우선 숙지
