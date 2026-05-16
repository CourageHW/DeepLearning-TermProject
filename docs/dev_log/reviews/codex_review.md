# Codex Adversarial Review of plan_v1.md (CIFAR-100 W1A1 ReActNet-lite term project)

## Q1. Realism of W1A1 ResNet18 ≥70%
- **Change:** Treat ≥70% as a stretch goal, not the plan's success criterion. ReActNet-18 CIFAR-100 ≈ 69.35-69.37%; ReCU ResNet-18 ≈ 68.7% at 300 ep / 69.1% at 600 ep, max ~69.02%; Bi-Real-style CIFAR-100 ≈ low/mid-60s; RBNN CIFAR-100 high-60s to low-70s (source-dependent). 2-week custom impl unlikely to beat mature recipes.
- **Change:** Make "68-70% A1W1, switch/report if below 70%" the honest target.

## Q2. Gaps in W32A1→W1A1 Recipe
- **Change:** Not enough optimizer detail for binary stability: AdamW vs SGD matters; BN parameter treatment matters; no separate lr for thresholds/RPReLU/BN.
- **Change:** Ordering under-specified. Plan says BN→RSign→Conv→BN→RPReLU but code currently has BN→RSign→Conv→BN→RSign→Conv→BN→add (no RPReLU).
- **Change:** Mixup/CutMix compatibility hand-waved. Don't assume Mixup α 0.4→0 is benign. Run no-mix / mixup-only / cutmix-only ablation.

## Q3. Missing Tricks
- **Change:** Double-skip in plan but missing in code. `A1W1BasicBlock.forward` only `out + residual`.
- **Change:** RPReLU planned but absent in code. ReActNet depends on activation distribution reshaping.
- **Change:** Gradient clipping planned but missing. There is `clip_binary_weights=True`, but no `clip_grad_norm_`.
- **Change:** EMA-on-binary missing. `DistillationWrapper(teacher)` = fixed teacher, not EMA of binary student.
- **Keep:** Label smoothing `0.0` consistent (ablate it).

## Q4. W2A2: LSQ vs PACT vs DSQ
- **Keep:** LSQ is safest 2-week choice.
- **Cut:** DSQ. Annealing introduces failure modes under deadline.
- **Change:** PACT viable for activations, but adds clipping-alpha tuning; less clean for signed BN-centered acts. Use LSQ or simple symmetric fake-quant fallback.

## Q5. HW Inference Estimate
- **Change:** BitOPs alone not enough. Exaggerates speed when memory/BN/fp stem/fp classifier/packing/missing XNOR kernels ignored.
- **Keep:** BitOPs + weight memory as main table.
- **Change:** Add simple roofline: binary BOPs + FP ops for stem/fc/BN + parameter bytes + activation bytes. PyTorch latency = "unaccelerated baseline." ONNX → W8A8 only. Larq/bitorch optional.

## Q6. Presentation Story
- **Keep:** Pareto frontier (accuracy vs bit-width; rescues project if A1W1 misses 70%).
- **Keep:** HW estimate (without it, binary looks like accuracy downgrade).
- **Cut:** Full KD story (4 runs expensive, distracts). Keep one KD on/off sanity if time remains.

## Q7. Likely Code Bugs / Risks
- **Keep:** `BinaryActivationSTE` not visibly broken (saves x, masks `abs(x)<=1`).
- **Change:** Alpha semantics risky. `alpha = self.weight.abs().mean(...)` not detached; gradients flow through both sign STE and scale. If static alpha intended, this is wrong / undocumented.
- **Change:** RSign under mixup risky: mixed pixels created while thresholds learned per-channel; no regularization or mixup-specific handling.
- **Keep:** Weight clipping frequency matches plan.
- **Change:** FP16 autocast around binary forward can destabilize sign/threshold boundaries; `ge(0)` and STE masks run in autocast dtype. Disable autocast inside binary sign/threshold ops if noisy.
