"""Step 1 stem-adaptation figure — 4 variants side by side."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- Korean font + crisp rendering --------------------------------------------
mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"]  = 42        # TrueType in PDF (no Type-3)
mpl.rcParams["svg.fonttype"]  = "none"    # keep text as text in SVG (PPT can edit)

C_FP    = "#dbe9ff"; C_FP_E    = "#3b6fb4"
C_ACT   = "#f5d6e8"; C_ACT_E   = "#a23b78"
C_POOL  = "#fff6cf"; C_POOL_E  = "#a07a14"
C_OUT   = "#e6f5d6"; C_OUT_E   = "#62995a"
C_ID    = "#eeeeee"; C_ID_E    = "#888888"
C_WARN  = "#ffeae1"; C_WARN_E  = "#c14a10"
C_CHOSEN= "#fff8e1"; C_CHOSEN_E= "#c97a17"
C_RES   = "#ffffff"; C_RES_E   = "#888888"
C_RES_C = "#fff3cf"; C_RES_CE  = "#a07a14"

def box(ax, x, y, w, h, text, fc, ec, fontsize=10, weight="normal",
        ec_w=1.4, dashed=False, fc_text=None):
    ls = "--" if dashed else "-"
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=ec_w, facecolor=fc, edgecolor=ec,
                       linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight,
            color=(fc_text or "black"))

def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=11,
                        color=color, lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

# title
ax.text(8, 9.55, "Step 1 — Stem adaptation  (ImageNet ResNet18 → CIFAR-100 32×32)",
        ha="center", va="center", fontsize=16, weight="bold")
ax.text(8, 9.22,
        "larger spatial map entering layer1 ⇒ more usable features for 32×32 inputs    "
        "(FP32 conv1 throughout — teacher phase)",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

# panel geometry
col_xs   = [0.4, 4.3, 8.2, 12.1]
col_w    = 3.5
panel_y0 = 0.95
panel_h  = 7.95
bw       = 2.7
cx_offset = (col_w - bw) / 2

# variant definitions
variants = [
    {
        "head": "A.  Original ImageNet stem",
        "sub":  "7×7 s=2  +  MaxPool 3×3 s=2",
        "card_fc": "#f7f7f7", "card_ec": "#bbbbbb", "card_lw": 1.4,
        "rows": [
            ("Input\n3 × 32 × 32",                 C_FP,   C_FP_E,   "bold"),
            ("Conv 7×7\nstride 2,  pad 3\n3 → 64", C_FP,   C_FP_E,   "normal"),
            ("ReLU + BN\n→ 64 × 16 × 16",          C_ACT,  C_ACT_E,  "normal"),
            ("MaxPool 3×3\nstride 2,  pad 1",      C_POOL, C_POOL_E, "bold"),
            ("→ 64 × 8 × 8\nenters layer1",        C_OUT,  C_OUT_E,  "bold"),
        ],
        "res_text": "Top-1   70.90 %",
        "res_fc": C_RES, "res_ec": C_RES_E,
        "note": "32×32 입력에 7×7 s=2 + maxpool 적용 시\n"
                "layer1 진입 단계에서 이미 8×8 로 축소.\n"
                "receptive field 가 과해 정보 손실 큼.",
        "chosen": False,
    },
    {
        "head": "B.  MaxPool → Identity",
        "sub":  "7×7 s=2  +  no MaxPool",
        "card_fc": "#f7f7f7", "card_ec": "#bbbbbb", "card_lw": 1.4,
        "rows": [
            ("Input\n3 × 32 × 32",                 C_FP,   C_FP_E,   "bold"),
            ("Conv 7×7\nstride 2,  pad 3\n3 → 64", C_FP,   C_FP_E,   "normal"),
            ("ReLU + BN\n→ 64 × 16 × 16",          C_ACT,  C_ACT_E,  "normal"),
            ("Identity   (no pooling)",            C_ID,   C_ID_E,   "normal"),
            ("→ 64 × 16 × 16\nenters layer1",      C_OUT,  C_OUT_E,  "bold"),
        ],
        "res_text": "Top-1   74.50 %     (+3.60 %p)",
        "res_fc": C_RES, "res_ec": C_RES_E,
        "note": "maxpool 만 제거하여도 layer1 진입 시\n"
                "16×16 가 유지 — 정보 보존 증가.",
        "chosen": False,
        "row3_dashed": True,
    },
    {
        "head": "C.  Conv 3×3 s=1  +  Identity",
        "sub":  "conv1 재초기화  (pretrained 7×7 weight 불호환)",
        "card_fc": "#f7f7f7", "card_ec": "#bbbbbb", "card_lw": 1.4,
        "rows": [
            ("Input\n3 × 32 × 32",                                C_FP,   C_FP_E,  "bold"),
            ("Conv 3×3   stride 1,  pad 1   3 → 64\n"
             "⚠ random init  (shape ≠ pretrained)",               C_WARN, C_WARN_E,"bold"),
            ("ReLU + BN\n→ 64 × 32 × 32",                         C_ACT,  C_ACT_E, "normal"),
            ("Identity   (no pooling)",                           C_ID,   C_ID_E,  "normal"),
            ("→ 64 × 32 × 32\nenters layer1",                     C_OUT,  C_OUT_E, "bold"),
        ],
        "res_text": "Top-1   80.60 %     (+9.70 %p)",
        "res_fc": C_RES, "res_ec": C_RES_E,
        "note": "해상도 보존 효과 큼  (16² → 32²).\n"
                "단, conv1 가 pretrained 7×7 와 shape 불일치 →\n"
                "random init 으로 시작 → ImageNet 정보 손실.",
        "chosen": False,
        "row3_dashed": True,
    },
    {
        "head": "★  D.  Conv 7×7 s=1  +  Identity",
        "sub":  "pretrained 7×7 weight 재사용  +  stride 만 1 로 변경",
        "card_fc": C_CHOSEN, "card_ec": C_CHOSEN_E, "card_lw": 3.0,
        "rows": [
            ("Input\n3 × 32 × 32",                                C_FP,   C_FP_E,  "bold"),
            ("Conv 7×7   stride 1,  pad 3   3 → 64\n"
             "✓ pretrained weight 유지",                          C_FP,   C_FP_E,  "bold"),
            ("ReLU + BN\n→ 64 × 32 × 32",                         C_ACT,  C_ACT_E, "normal"),
            ("Identity   (no pooling)",                           C_ID,   C_ID_E,  "normal"),
            ("→ 64 × 32 × 32\nenters layer1",                     C_OUT,  C_OUT_E, "bold"),
        ],
        "res_text": "Top-1   81.82 %     (+10.92 %p)",
        "res_fc": C_RES_C, "res_ec": C_RES_CE,
        "note": "해상도 보존  +  ImageNet 7×7 weight 재사용.\n"
                "학습시간 5.27 → 9.52 s/epoch (≈ 1.8× ↑) 이지만\n"
                "정확도 +11 %p 로 trade-off 정당화.",
        "chosen": True,
        "row3_dashed": True,
    },
]

# row layout (within panel)
heights = [0.55, 0.85, 0.55, 0.55, 0.55]
gap = 0.18
total_rows = sum(heights) + gap * (len(heights) - 1)
rows_top = panel_y0 + panel_h - 0.95          # leave room for head + subhead at top

for i, v in enumerate(variants):
    x0 = col_xs[i]
    # card background
    box(ax, x0, panel_y0, col_w, panel_h, "",
        v["card_fc"], v["card_ec"], ec_w=v["card_lw"])
    # head + subhead
    ax.text(x0 + col_w/2, panel_y0 + panel_h - 0.30,
            v["head"], ha="center", va="center",
            fontsize=12, weight="bold")
    ax.text(x0 + col_w/2, panel_y0 + panel_h - 0.62,
            v["sub"], ha="center", va="center",
            fontsize=9.5, style="italic", color="#666")

    # rows (stacked downward)
    cx = x0 + col_w/2
    y_top = rows_top
    row_centers_y = []
    for r_idx, (txt, fc, ec, wt) in enumerate(v["rows"]):
        h = heights[r_idx]
        dashed = (r_idx == 3 and v.get("row3_dashed", False))
        box(ax, cx - bw/2, y_top - h, bw, h, txt, fc, ec,
            fontsize=9.5, weight=wt, dashed=dashed)
        row_centers_y.append((y_top, y_top - h))
        y_top -= (h + gap)

    # arrows between row pairs
    arrow_color = C_CHOSEN_E if v["chosen"] else "#333"
    arrow_lw = 1.8 if v["chosen"] else 1.2
    for k in range(len(row_centers_y) - 1):
        y_a = row_centers_y[k][1]      # bottom of upper box
        y_b = row_centers_y[k+1][0]    # top of lower box
        arrow(ax, cx, y_a, cx, y_b, color=arrow_color, lw=arrow_lw)

    # result chip
    res_y_top = y_top - 0.05
    res_h = 0.55
    box(ax, x0 + 0.15, res_y_top - res_h, col_w - 0.30, res_h,
        v["res_text"], v["res_fc"], v["res_ec"],
        fontsize=12, weight="bold")

    # note (below result)
    ax.text(cx, res_y_top - res_h - 0.35, v["note"],
            ha="center", va="top", fontsize=9, style="italic", color="#555")

# bottom takeaway bar
box(ax, 0.4, 0.20, 15.2, 0.55,
    "Takeaway   maxpool 제거 (+3.6 %p)  <  stride 2 → 1 (+6.1 %p)   →   spatial-resolution 보존이 dominant factor.   "
    "pretrained 7×7 weight 재사용 시 +1.22 %p 추가 이득.",
    "#fafafa", "#bbbbbb", fontsize=11)

OUT = "/home/yonggi/KAU/3-1/딥러닝/Termp/arch_stem_step1"
fig.savefig(OUT + ".png", bbox_inches="tight", dpi=300)
fig.savefig(OUT + ".svg", bbox_inches="tight")          # vector — paste into PPT
fig.savefig(OUT + ".pdf", bbox_inches="tight")          # also vector
plt.close(fig)
print("wrote arch_stem_step1.{png,svg,pdf}")
