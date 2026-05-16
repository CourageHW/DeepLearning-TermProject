#!/usr/bin/env python3
"""
Generate fully-editable native PowerPoint (.pptx) from slides.md content.

Each slide becomes a real pptx slide with native text shapes (clickable in
PowerPoint/Keynote) — unlike Marp's image-only output. Images (plots) and
tables are inserted as native shapes too.

Output: slides_native.pptx (~150-300 KB, 18 slides, 16:9 widescreen).
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

DIR = Path(__file__).parent
GREEN = RGBColor(0x21, 0x96, 0x4B)
RED   = RGBColor(0xC0, 0x39, 0x2B)
GRAY  = RGBColor(0x7F, 0x8C, 0x8D)
NAVY  = RGBColor(0x2C, 0x3E, 0x50)
LIGHT = RGBColor(0xEC, 0xEC, 0xEC)

# 16:9 widescreen
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

SW, SH = prs.slide_width, prs.slide_height
LEFT_MARGIN = Inches(0.5)
TOP_CONTENT = Inches(1.5)
WIDTH_FULL  = Inches(12.333)

BLANK_LAYOUT = prs.slide_layouts[6]  # blank


def add_title(slide, text, color=NAVY, size=32):
    tb = slide.shapes.add_textbox(LEFT_MARGIN, Inches(0.3), WIDTH_FULL, Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    return tb


def add_bullets(slide, items, top=TOP_CONTENT, height=Inches(5.5), font_size=18,
                left=LEFT_MARGIN, width=WIDTH_FULL):
    """items = list of (text, level, optional_color). Level 0=top bullet, 1=sub."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text = item[0]; level = item[1] if len(item) > 1 else 0
            color = item[2] if len(item) > 2 else None
        else:
            text, level, color = item, 0, None
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet = "  " * level + ("• " if level == 0 else "– ")
        p.text = bullet + text
        p.font.size = Pt(font_size if level == 0 else font_size - 2)
        p.font.color.rgb = color or NAVY
        p.space_after = Pt(6)
    return tb


def add_subtitle(slide, text, top, size=20, color=GRAY, bold=False):
    tb = slide.shapes.add_textbox(LEFT_MARGIN, top, WIDTH_FULL, Inches(0.6))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    return tb


def add_image(slide, path, left, top, width=None, height=None):
    if not Path(path).exists():
        return None
    if width and height:
        return slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    if width:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    return slide.shapes.add_picture(str(path), left, top)


def add_table(slide, headers, rows, top, height=None, font_size=14,
              col_widths=None, header_color=NAVY, header_text_color=RGBColor(255,255,255)):
    n_cols = len(headers)
    n_rows = 1 + len(rows)
    width = WIDTH_FULL
    height = height or Inches(0.4 * n_rows)
    table_shape = slide.shapes.add_table(n_rows, n_cols, LEFT_MARGIN, top, width, height)
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    # Header
    for ci, h in enumerate(headers):
        cell = table.cell(0, ci)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = header_color
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(font_size); r.font.bold = True
                r.font.color.rgb = header_text_color
    # Rows
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)
    return table_shape


def add_speaker_note(slide, text):
    notes = slide.notes_slide.notes_text_frame
    notes.text = text


def add_footer(slide, text, color=GRAY):
    tb = slide.shapes.add_textbox(LEFT_MARGIN, Inches(7.05), WIDTH_FULL, Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.RIGHT


# ============================================================
# Slide 0 — Title
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
# Big title centered
tb = s.shapes.add_textbox(Inches(1), Inches(2.0), Inches(11.333), Inches(1.5))
p = tb.text_frame.paragraphs[0]
p.text = "CIFAR-100 1-bit Binary Neural Network"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = NAVY
p2 = tb.text_frame.add_paragraph()
p2.text = "with Knowledge Distillation"
p2.alignment = PP_ALIGN.CENTER
p2.font.size = Pt(32); p2.font.bold = True; p2.font.color.rgb = NAVY

tb2 = s.shapes.add_textbox(Inches(1), Inches(4.2), Inches(11.333), Inches(0.6))
p = tb2.text_frame.paragraphs[0]
p.text = "ReActNet-lite ResNet18 · CIFAR-100 · Knowledge Distillation · POPCNT-based C inference"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(18); p.font.color.rgb = GRAY

tb3 = s.shapes.add_textbox(Inches(1), Inches(5.0), Inches(11.333), Inches(1.0))
p = tb3.text_frame.paragraphs[0]
p.text = "Team 16 · KAU Deep Learning Term Project"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(22); p.font.bold = True
p2 = tb3.text_frame.add_paragraph()
p2.text = "2026 Spring"
p2.alignment = PP_ALIGN.CENTER
p2.font.size = Pt(16); p2.font.color.rgb = GRAY

add_speaker_note(s, "발표 시작. 10분 발표.")

# ============================================================
# Slide 1 — Motivation & Goal
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "1. Motivation & Goal")
add_bullets(s, [
    ("문제: 클라우드 의존 없는 on-device AI inference 수요가 커지지만, 모바일·노트북·엣지의 메모리·전력 예산은 모델 크기를 못 따라감", 0),
    ("ResNet18 (CIFAR-100 head) ≈ 11.2 M params, 44 MB FP32", 1),
    ("해결 아이디어: Binary Neural Network (BNN)", 0),
    ("Weight·Activation 을 ±1 (1-bit) 로 양자화 → 이론 32× 메모리 압축", 1),
    ("MAC 연산을 XNOR + POPCNT 로 대체 → 정수·HW 가속 친화적", 1),
    ("본 프로젝트 목표", 0),
    ("CIFAR-100 W1A1 BNN 학습 → literature SOTA 영역 (~69%) 도달", 1),
    ("Teacher 를 empirical study 로 강화 후 KD 로 student 정확도 push", 1),
    ("POPCNT 기반 C 추론 커널로 CPU (Intel Core i5-1135G7) 에서 speedup 실측", 1),
], font_size=18)
add_footer(s, "Slide 1 / 18  ·  ~50초")
add_speaker_note(s, "BNN은 32배 메모리 압축과 XNOR-POPCNT 기반 정수 연산이 핵심. 본 프로젝트는 정확도와 HW 추론 두 가지 목표를 동시에 진행했습니다. 70% literature SOTA 영역 도달을 목표로 잡았습니다.")

# ============================================================
# Slide 2 — Method Overview
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "2. Method Overview")
add_subtitle(s, "Why ReActNet-lite?", Inches(1.4), size=18, bold=True, color=NAVY)
add_bullets(s, [
    ("ResNet18 backbone — 강의 standard, ImageNet pretrained 활용 가능", 0),
    ("ReActNet 의 학습 가능한 sign threshold (RSign) + PReLU 활성 (RPReLU)", 0),
    ("Bi-Real 의 double-skip 으로 1-bit 정보 손실 완화", 0),
], top=Inches(1.8), font_size=16)

add_subtitle(s, "모델 구성", Inches(3.5), size=18, bold=True, color=NAVY)
add_bullets(s, [
    ("Teacher: FP32 ResNet18 (CIFAR-adapted, ImageNet pretrained → FT)", 0),
    ("Student: A1W1ResNet18 (W1A1, FP32 stem + classifier, binary body)", 0),
], top=Inches(3.9), font_size=16)

add_subtitle(s, "Distillation: Logit KD (Hinton)", Inches(5.4), size=18, bold=True, color=NAVY)
tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(5.8), WIDTH_FULL, Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = "L = (1 − w) · CE(p_s, y) + w · T² · KL(p_t ‖ p_s),  T = 4, w = 0.7"
p.font.size = Pt(16); p.font.name = "Consolas"
p2 = tb.text_frame.add_paragraph()
p2.text = "Phase B: KD ON from epoch 15 (warmup 종료 후), Attention Transfer 는 final 학습에서 미사용"
p2.font.size = Pt(14); p2.font.color.rgb = GRAY

add_footer(s, "Slide 2 / 18  ·  ~45초")
add_speaker_note(s, "ReActNet의 RSign/RPReLU + Bi-Real의 double-skip을 결합한 lite 디자인. Teacher는 ImageNet pretrained ResNet18을 FT, Student는 W1A1로 KD 학습합니다.")

# ============================================================
# Slide 3 — Student Architecture
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "3. Student Architecture: A1W1ResNet18")

tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(1.3), WIDTH_FULL, Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = ("ReActNet 의 핵심 트릭 (RSign + RPReLU + double-skip + detached α) 차용 + 두 가지 단순화:")
p.font.size = Pt(15); p.font.color.rgb = NAVY
p2 = tb.text_frame.add_paragraph()
p2.text = "  • backbone: ReActNet-A → 표준 ResNet18    • 학습: 2-stage W32A1→W1A1 KD → 단일 stage W1A1 KD"
p2.font.size = Pt(14); p2.font.color.rgb = GRAY

add_subtitle(s, "Binary Residual Block (double-skip + RPReLU)", Inches(2.4), size=16, bold=True, color=NAVY)
tb2 = s.shapes.add_textbox(LEFT_MARGIN, Inches(2.9), WIDTH_FULL, Inches(1.7))
tb2.fill.solid(); tb2.fill.fore_color.rgb = LIGHT
p = tb2.text_frame.paragraphs[0]
p.text = ("x ──┬── BN → RSign → BinConv → BN → RPReLU ──┬── + res_a ──┐\n"
         "    └────────────────────────────────────────┘             │\n"
         "        ─┬── BN → RSign → BinConv → BN → RPReLU ──┬── + res_b ──→ y\n"
         "         └────────────────────────────────────────┘")
p.font.size = Pt(11); p.font.name = "Consolas"; p.font.color.rgb = NAVY

add_subtitle(s, "Memory budget (11.22 M params)", Inches(4.8), size=16, bold=True, color=NAVY)
add_table(s, ["부분", "params", "bits/param", "memory"],
          [["Binary conv body (16 layers)", "11.1 M", "1 bit", "1.39 MB"],
           ["FP32 stem + classifier (1%)", "0.12 M", "32 bit", "0.48 MB"],
           ["Total student", "11.22 M", "—", "≈ 1.4 MB"],
           ["Teacher (FP32)", "11.22 M", "32 bit", "44.0 MB"]],
          top=Inches(5.3), font_size=12)

add_footer(s, "Slide 3 / 18  ·  ~60초")
add_speaker_note(s, "Block 안에 2개의 binary conv + 2개의 residual skip. 메모리는 99%가 binary, 1%만 FP32 유지로 44MB → 1.4MB.")

# ============================================================
# Slide 4 — Teacher Empirical Study
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "4. Teacher: Empirical Study (CSV 7-step ablation)")
add_bullets(s, [
    ("ResNet18 baseline (150 ep from-scratch): 75.74 %", 0),
    ("Step 1 stem (7×7 stride 2 → 3×3 stride 1, maxpool 제거): 79.62 %", 0),
    ("Step 2 optimizer (SGD → AdamW, wd 1e-4): 80.84 %", 0),
    ("Step 3 batch / LR sweep: best 81.43 %", 0),
    ("Step 4 scheduler (CosineAnneal + warmup): 81.66 %", 0),
    ("Step 5 activation: ReLU 82.38 % (GELU 82.46 % — pretrained 호환성 우선)", 0),
    ("Step 6 mixup decay: 82.50 %", 0),
    ("Step 7 RandAugment mag=9: 82.58 % (mag=10 83.12 % — over-reg 보수 선택)", 0),
    ("Final Phase A: ImageNet pretrained + 80 epoch FT → 82.44 %", 0, GREEN),
], font_size=15)
add_footer(s, "Slide 4 / 18  ·  ~55초")
add_speaker_note(s, "체계적 ablation으로 75.7%에서 83.1%까지 끌어올렸고, pretrained-FT 환경의 안정성을 위해 ReLU + mag=9를 최종 채택해 82.44%를 확보했습니다.")

# ============================================================
# Slide 5 — Training Pipeline
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "5. Training Pipeline")
add_subtitle(s, "Single Kaggle session (T4, 12h budget)", Inches(1.3), size=16, bold=True, color=NAVY)
add_table(s, ["Phase", "Model", "Epoch", "Time"],
          [["A", "Teacher FT (ImageNet → CIFAR)", "80", "≈ 15 min"],
           ["B", "Student (A1W1 + Logit KD T=4, w=0.7)", "220", "≈ 80 min"],
           ["Total", "", "", "≈ 1h 35 min"]],
          top=Inches(1.8), font_size=13)
add_subtitle(s, "Phase B 핵심 hyperparam", Inches(3.7), size=16, bold=True, color=NAVY)
add_bullets(s, [
    ("Optimizer: AdamW (wd = 0), LR 5e-4 → 1e-6 cosine", 0),
    ("Warmup 15 ep, KD 활성 시점 = warmup 종료 (epoch 15)", 0),
    ("Latent weight clamp [−1, +1] (매 step) — sign STE dynamic range 보호", 0),
    ("Student EMA disabled (binary weight 평균 무의미)", 0),
    ("Augmentation: RandAugment / Mixup α=0.4 step decay / CutMix / RandomErasing", 0),
], top=Inches(4.2), font_size=14)
add_footer(s, "Slide 5 / 18  ·  ~50초")
add_speaker_note(s, "단일 Kaggle 세션 안에 두 phase 모두. Student는 220 epoch + KD T=4 w=0.7로 학습. Student EMA는 binary 특성상 비활성화.")

# ============================================================
# Slide 6 — Quantitative Results
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "6. Quantitative Results")
add_subtitle(s, "CIFAR-100 Top-1", Inches(1.3), size=16, bold=True, color=NAVY)
add_table(s, ["Model", "Params", "Memory", "Top-1"],
          [["Teacher (FP32 ResNet18)", "11.22 M", "44.0 MB", "82.44 %"],
           ["Student (A1W1 ReActNet-lite)", "11.22 M", "1.4 MB", "68.72 %"]],
          top=Inches(1.8), font_size=14)

add_subtitle(s, "Literature comparison (W1A1)", Inches(3.3), size=16, bold=True, color=NAVY)
add_table(s, ["Method", "Dataset", "Top-1"],
          [["XNOR-Net (ECCV'16)", "ImageNet", "51.2 %"],
           ["Bi-Real (ECCV'18)", "ImageNet", "56.4 %"],
           ["ReActNet (ECCV'20)", "ImageNet", "69.4 %"],
           ["ReCU* (ICCV'21)", "ImageNet", "66.4 %"],
           ["RBNN", "CIFAR-100", "~ 67.1 %"],
           ["Ours", "CIFAR-100", "68.72 %"]],
          top=Inches(3.8), font_size=12)

tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(6.7), WIDTH_FULL, Inches(0.4))
p = tb.text_frame.paragraphs[0]
p.text = "📦 44 MB → 1.4 MB · 32× 압축   ⚠️ ReActNet 격차 −0.7 %p (1-bit 천장 ~ 71 %)"
p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = GREEN
add_footer(s, "Slide 6 / 18  ·  ~70초")
add_speaker_note(s, "Student 68.72%는 literature SOTA 영역. CIFAR-100 RBNN 67.1% 대비 +1.6%p이고, ReActNet ImageNet 69.4%와 0.7%p 차이. 메모리는 44MB에서 1.4MB로 32배 압축, 이론 상한에 매우 근접.")

# ============================================================
# Slide 7 — Design Choices & Discussion
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "7. Design Choices & Ablation")
add_bullets(s, [
    ("Student EMA 비활성: binary weight 평균은 sign() 부수므로 random 모델 — ReActNet/Bi-Real/ReCU 모두 동일", 0),
    ("FP32 stem + classifier (1%): 200 KB 만 차지하면서 +5–10 %p 정확도 보존 (BNN 표준)", 0),
    ("α 를 graph 에서 detach: 안 하면 weight 가 sign 미분 0 영역으로 끌려 발산", 0),
    ("AMP fp16 binary STE: ±1 sign 은 fp16 동적 범위 내 안정, Tensor Core 활용 → 학습 1.5–2 ×", 0),
    ("Ablation 검증 — 컴포넌트 기여도 (Backup F 상세):", 0, NAVY),
    ("RPReLU 가 단일 최대 기여 — 제거 시 −9.22 %p", 1, RED),
    ("Double-skip 제거 시 −2.46 %p, KD off 시 −2.30 %p (비슷한 크기)", 1),
], top=Inches(1.4), font_size=15)
add_footer(s, "Slide 7 / 18  ·  ~55초")
add_speaker_note(s, "디자인 결정을 강조합니다. Student EMA는 sign을 부수므로 비활성, stem과 분류기 1%만 FP32 유지, 알파 스케일은 detach. Ablation으로 RPReLU가 9.22%p 단일 최대 기여 컴포넌트임을 검증.")

# ============================================================
# Slide 8 — C Inference Design + Speedup
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "8. C Inference: Design + Speedup")

add_subtitle(s, "C kernel (xnor_kernel.c, ~350 LOC, POPCNT + OpenMP)", Inches(1.3), size=15, bold=True, color=NAVY)
add_bullets(s, [
    ("xnor_gemm: packed uint64 + _mm_popcnt_u64 + cache-blocked, M-axis OpenMP", 0),
    ("Padding correction: zero-pad ↔ binary {−1,+1} 오차를 패턴별 popcount 보정", 0),
    ("기타: fp32_conv2d (stem), batch_norm_2d, rsign, rprelu, linear_fp32", 0),
], top=Inches(1.8), font_size=13)

tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(3.4), WIDTH_FULL, Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = "✅ PyTorch ↔ C max diff = 0.0000 (test set 10,000 sample 모두)"
p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = GREEN

add_subtitle(s, "Wall-clock latency (CPU: Intel Core i5-1135G7, 4C / 8T)", Inches(4.0), size=15, bold=True, color=NAVY)
add_table(s, ["모드", "Student (W1A1)", "Teacher (FP32)", "Speedup", "의미"],
          [["Single-thread", "22 ms", "547 ms", "25 ×", "알고리즘 차이"],
           ["4-thread (OpenMP)", "15 ms", "173 ms", "12 ×", "실용 환경"]],
          top=Inches(4.5), font_size=12)

tb2 = s.shapes.add_textbox(LEFT_MARGIN, Inches(6.4), WIDTH_FULL, Inches(0.5))
p = tb2.text_frame.paragraphs[0]
p.text = ("Ratio 감소 이유: Teacher 큰 GEMM 으로 multi-thread 이득 3× / "
          "Student 작은 binary conv 16 개로 1.5×. 두 수치 모두 정직한 측정값.")
p.font.size = Pt(11); p.font.color.rgb = GRAY

add_footer(s, "Slide 8 / 18  ·  ~75초")
add_speaker_note(s, "C 커널은 약 350줄, XNOR-POPCNT, padding correction, BN, RSign, RPReLU, FP32 stem+분류기 포함. PyTorch와 bit-exact 일치를 검증. Intel Core i5-1135G7 노트북 CPU에서 single-thread 25배, 4-thread 12배 speedup. 두 수치 모두 정직한 측정값.")

# ============================================================
# Slide 9 — Limitations & Honest Discussion
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "9. Limitations & Honest Discussion")
add_bullets(s, [
    ("W1A1 SOTA 격차 ~0.7 %p: ReActNet 69.4 % vs Ours 68.7 %. 1-bit 천장 ~71 %", 0),
    ("T4 GPU INT1 native 미지원 (Turing 은 INT4 까지) → GPU 가속 불가, CPU 추론 demo 로 우회", 0),
    ("Naive C 비교 한계 + BLAS 보강 측정:", 0),
    ("메인 C 는 VPOPCNTDQ / SIMD-GEMM 미사용 → 25× / 12× = 알고리즘 차이", 1),
    ("BLAS-backed teacher (OpenBLAS) 와 비교 시 1-thread 3.15×, 4-thread 6.12× = production 차이", 1),
    ("Ablation 범위 한계 (§6.5 / Backup F):", 0),
    ("단일 컴포넌트 ablation 만 수행 — RPReLU 가 최대 기여 (−9.22 %p)", 1),
    ("cross-component interaction, KD weight sweep, AT 안정화는 미수행", 1),
    ("AT loss 실패: layer-wise LR, BN sync, weight scheduling 등 추가 필요", 0),
], top=Inches(1.4), font_size=14)
add_footer(s, "Slide 9 / 18  ·  ~50초")
add_speaker_note(s, "솔직한 한계 정리: SOTA 격차 0.7%p, T4 GPU INT1 미지원으로 CPU demo로 우회, naive C 한계 — BLAS와 비교 시 3.15-6.12배 (production 차이). AT loss는 안정화에 실패.")

# ============================================================
# Slide 10 — Conclusion + Future Work
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "10. Conclusion + Future Work")
add_subtitle(s, "핵심 성과 (3 줄 요약)", Inches(1.4), size=18, bold=True, color=NAVY)
add_bullets(s, [
    ("CIFAR-100 W1A1 BNN 68.72 % top-1 — literature SOTA 영역 (ReActNet 69.4 % 대비 −0.7 %p)", 0, GREEN),
    ("메모리 32× 압축 (44 MB → 1.4 MB) + 체계적 teacher empirical study (75.7 % → 83.1 %)", 0, GREEN),
    ("POPCNT-based C 추론 demo — PyTorch 와 bit-exact, single-thread 25× speedup 실측", 0, GREEN),
], top=Inches(1.9), font_size=15)

add_subtitle(s, "Future Work", Inches(4.0), size=18, bold=True, color=NAVY)
add_bullets(s, [
    ("Cross-component ablation (E1 × E2 × E3), KD T/w sweep", 0),
    ("SAM optimizer (BNN × SAM 효과 검증)", 0),
    ("Bit-width Pareto (W8A8 / W4A4 LSQ) 비교", 0),
    ("AT loss 안정화 (layer-wise LR, BN sync)", 0),
    ("AVX-512 VPOPCNTDQ + BLAS-style GEMM → production-grade C 커널 확장", 0),
], top=Inches(4.5), font_size=14)
add_footer(s, "Slide 10 / 18  ·  ~40초")
add_speaker_note(s, "3줄 요약: CIFAR-100 W1A1 68.72%, 메모리 32배 압축, C 추론 25배 speedup. 시간 관계상 짧게 마무리.")

# ============================================================
# Slide 11 — Q & A
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
tb = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11.333), Inches(2.5))
p = tb.text_frame.paragraphs[0]
p.text = "Q & A"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(72); p.font.bold = True; p.font.color.rgb = NAVY

tb2 = s.shapes.add_textbox(Inches(1), Inches(4.8), Inches(11.333), Inches(1.0))
p = tb2.text_frame.paragraphs[0]
p.text = "감사합니다."
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(28); p.font.color.rgb = GRAY

add_speaker_note(s, "Q&A. 예상 질문: 왜 CPU? No RPReLU=PReLU/제거? 25× 는 PyTorch 대비?")

# ============================================================
# Backup A — Phase A Teacher Hyperparam
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup A — Phase A Teacher Hyperparameters")
add_table(s, ["항목", "값"],
          [["Backbone", "ResNet18 (torchvision, ImageNet pretrained)"],
           ["Stem", "Conv1 7×7 stride 1, maxpool → Identity"],
           ["Optimizer", "AdamW (wd = 1e-4)"],
           ["Scheduler", "LinearWarmup (8 ep, 0.1→1.0) + CosineAnnealingLR (72 ep, η_min = 1e-6)"],
           ["LR / Batch / Epoch", "1e-3 / 1024 / 80"],
           ["Activation", "ReLU"],
           ["Augmentation", "RandAugment (num_ops=2, mag=9), RandomErasing p=0.2"],
           ["Regularization", "Label smoothing 0.05, Mixup α=1.0 step decay → 0.0625, CutMix"],
           ["Final top-1", "82.44 %"]],
          top=Inches(1.4), font_size=12,
          col_widths=[Inches(3), Inches(9.333)])

add_footer(s, "Backup A  ·  Slide 12 / 18")

# ============================================================
# Backup B — Phase B Student Hyperparam
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup B — Phase B Student Hyperparameters")
add_table(s, ["분류", "항목", "Phase A", "Phase B"],
          [["Optimizer", "AdamW", "wd = 1e-4", "wd = 0"],
           ["LR", "base / min", "1e-3 / 1e-6", "5e-4 / 1e-6"],
           ["Scheduler", "Warmup + Cosine", "warmup 8 / cos 72", "warmup 15 / cos 205"],
           ["Epoch / Batch", "—", "80 / 1024", "220 / 1024"],
           ["KD", "Logit (T, w)", "—", "T=4, w=0.7 (epoch ≥ 15)"],
           ["KD", "Attention Transfer", "—", "disabled"],
           ["EMA", "shadow", "decay = 0.999", "disabled (binary 특성)"],
           ["Mixup", "α 초기 / min / decay", "1.0 / 0.0625 / step", "0.4 / 0 / step"],
           ["Reg", "label smoothing", "0.05", "0"],
           ["Reg", "grad clip", "—", "5.0"]],
          top=Inches(1.4), font_size=11,
          col_widths=[Inches(1.5), Inches(2.5), Inches(3.5), Inches(4.833)])

add_footer(s, "Backup B  ·  Slide 13 / 18")

# ============================================================
# Backup C — Data Augmentation
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup C — Data Augmentation Strategy")
add_table(s, ["Level", "기법", "Param", "역할"],
          [["Image", "RandomCrop", "size 32, pad 4 (reflect)", "위치 다양성"],
           ["Image", "RandomHorizontalFlip", "p = 0.5", "좌우 대칭"],
           ["Image", "RandAugment", "num_ops = 2, mag = 9", "색·기하 augment"],
           ["Image", "RandomErasing", "p = 0.2, scale (0.02, 1/3)", "occlusion robustness"],
           ["Sample", "Mixup", "α = 1.0 (teacher) / 0.4 (student), step decay", "interpolation regularization"],
           ["Sample", "CutMix", "α = 1.0, prob 0.5", "spatial replacement"],
           ["Norm", "Normalize", "ImageNet stats", "pretrained 호환성"]],
          top=Inches(1.4), font_size=11,
          col_widths=[Inches(1.5), Inches(3), Inches(3.5), Inches(4.333)])

add_footer(s, "Backup C  ·  Slide 14 / 18")

# ============================================================
# Backup D — C Kernel Detail
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup D — C Kernel Detail")
add_subtitle(s, "xnor_gemm (core primitive)", Inches(1.3), size=15, bold=True, color=NAVY)
tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(1.8), WIDTH_FULL, Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = "y = α · (2 · POPCNT(¬(A ⊕ W)) − K_bits)"
p.font.size = Pt(14); p.font.name = "Consolas"; p.font.color.rgb = NAVY

add_bullets(s, [
    ("Packed uint64 (64 bits / word), _mm_popcnt_u64 scalar POPCNT (VPOPCNTDQ 미사용)", 0),
    ("Cache blocking on N axis, OpenMP parallel on M axis (output channels)", 0),
], top=Inches(2.5), font_size=13)

add_subtitle(s, "함수 목록 (xnor_kernel.c, ~350 LOC)", Inches(3.5), size=15, bold=True, color=NAVY)
add_table(s, ["함수", "역할", "OpenMP"],
          [["xnor_gemm", "Binary matmul, packed uint64 + POPCNT", "✅ M-axis"],
           ["fp32_conv2d", "FP32 stem & teacher", "✅ C_out"],
           ["binary_conv2d", "1-bit conv + im2col + padding correction", "✅ 출력위치"],
           ["batch_norm_2d / rsign / rprelu", "Inference BN / RSign / RPReLU", "✅ 채널축"],
           ["shortcut_downsample / linear_fp32", "AvgPool + zero-pad / Classifier", "—"]],
          top=Inches(4.0), font_size=11,
          col_widths=[Inches(3.5), Inches(6.5), Inches(2.333)])

add_footer(s, "Backup D  ·  Slide 15 / 18")

# ============================================================
# Backup E — Threading 25× → 12× 의 이유
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup E — Threading: 25× → 12× 의 이유")
add_table(s, ["Threads", "Student (W1A1)", "Teacher (FP32)", "Ratio"],
          [["1 (default)", "22 ms", "547 ms", "25 ×"],
           ["4", "15 ms", "173 ms", "12 ×"],
           ["8 (HT 포함)", "21 ms", "200 ms", "9 ×"]],
          top=Inches(1.4), font_size=13)

add_subtitle(s, "왜 thread 수에 따라 ratio 가 변하는가", Inches(3.5), size=16, bold=True, color=NAVY)
add_bullets(s, [
    ("Teacher (FP32): 큰 GEMM 루프 → multi-thread 로 ~3 × 가속", 0),
    ("Student (binary): 작은 op 16 개 (binary conv) → thread spawn overhead 로 ~1.5 × 만 가속", 0),
    ("→ 4-thread 시 teacher 가 더 큰 이득 → ratio 25 → 12 × 감소", 0),
    ("두 숫자 모두 정직한 측정값. 발표에서는 single-thread (알고리즘) + multi-thread (실용) 모두 제시", 0),
    ("Default = single-thread (Codex review 의 fair comparison 가이드 준수)", 0, GRAY),
], top=Inches(4.0), font_size=14)

add_footer(s, "Backup E  ·  Slide 16 / 18")

# ============================================================
# Backup F — Student Design Ablation (with image)
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Backup F — Student Design Ablation")

tb = s.shapes.add_textbox(LEFT_MARGIN, Inches(1.2), WIDTH_FULL, Inches(0.4))
p = tb.text_frame.paragraphs[0]
p.text = "220 epoch 동일 schedule, baseline 에서 단일 컴포넌트만 제거"
p.font.size = Pt(14); p.font.color.rgb = GRAY

add_table(s, ["Variant", "Best Val", "Δ"],
          [["Baseline (KD + double-skip + RPReLU)", "68.74 %", "—"],
           ["E1: KD off", "66.44 %", "−2.30"],
           ["E2: Single-skip", "66.28 %", "−2.46"],
           ["E3: No RPReLU", "59.52 %", "−9.22"]],
          top=Inches(1.7), font_size=13,
          col_widths=[Inches(6), Inches(3), Inches(3.333)])

# Embed ablation bar plot
bars_img = DIR / "ablation_bars.png"
if bars_img.exists():
    add_image(s, bars_img, Inches(0.7), Inches(4.5), width=Inches(7.5))

add_bullets(s, [
    ("RPReLU 가 단일 최대 기여 (−9.22 %p)", 0, RED),
    ("activation reshape > weight precision (BNN 표현력 회복)", 1),
    ("KD ≈ double-skip (~2.3 %p)", 0),
    ("둘 다 gradient flow 보조 역할", 1),
    ("Baseline 만 후반 fine refinement (ep 215 vs 196–203)", 0),
], top=Inches(4.6), left=Inches(8.5), width=Inches(4.5), font_size=12)

add_footer(s, "Backup F  ·  Slide 17 / 18")
add_speaker_note(s, "RPReLU는 channel-wise learnable shift β로 1-bit activation 정보 손실을 보상. 단독 9%p 차이. KD와 double-skip은 각각 2.3%p로 거의 같은 크기 — 둘 다 gradient flow 보조.")

# ============================================================
# Slide 18 (final cover for End)
# ============================================================
s = prs.slides.add_slide(BLANK_LAYOUT)
tb = s.shapes.add_textbox(Inches(1), Inches(3.0), Inches(11.333), Inches(1.5))
p = tb.text_frame.paragraphs[0]
p.text = "Thank You"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(60); p.font.bold = True; p.font.color.rgb = NAVY
tb2 = s.shapes.add_textbox(Inches(1), Inches(5.0), Inches(11.333), Inches(0.5))
p = tb2.text_frame.paragraphs[0]
p.text = "Team 16 · KAU Deep Learning Term Project"
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(18); p.font.color.rgb = GRAY

# ============================================================
# Save
# ============================================================
OUT = DIR / "slides_native.pptx"
prs.save(OUT)
print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes, {len(prs.slides)} slides)")
