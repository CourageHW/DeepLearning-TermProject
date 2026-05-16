"""Generate ReActNet-lite v2 (A1W1ResNet18v2) architecture figures.

Outputs (each diagram in PNG + SVG + PDF):
  arch_reactnet_overview.{png,svg,pdf}  — full network pipeline
  arch_reactnet_block.{png,svg,pdf}     — BasicBlockV2 detail
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"]  = 42
mpl.rcParams["svg.fonttype"]  = "none"

# -------- color scheme --------
C_FP    = "#dbe9ff"
C_FP_E  = "#3b6fb4"
C_BIN   = "#ffe2b0"
C_BIN_E = "#c97a17"
C_NORM  = "#e6f5d6"
C_NORM_E= "#62995a"
C_ACT   = "#f5d6e8"
C_ACT_E = "#a23b78"
C_HEAD  = "#e2d6f7"
C_HEAD_E= "#6b3aa6"
C_SHORT = "#fff6cf"
C_SHORT_E="#a07a14"
C_SKIP  = "#666"

def box(ax, x, y, w, h, text, fc, ec, fontsize=9, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.4, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)

def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.2, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=12,
                        color=color, lw=lw, linestyle=style,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# =====================================================================
# Figure 1 — overall pipeline (kept as-is)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 4.2), dpi=300)
ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis("off")

y0 = 2.0; w = 1.55; h = 1.1
stages = [
    ("Input\n3×32×32",                              C_FP,   C_FP_E,  False),
    ("Stem\nConv3×3 s=1\nBN + RPReLU\n→ 64×32×32",  C_FP,   C_FP_E,  False),
    ("layer1 ×2\n64 → 64, s=1\nBasicBlockV2",       C_BIN,  C_BIN_E, True),
    ("layer2 ×2\n64 → 128, s=2\nBasicBlockV2",      C_BIN,  C_BIN_E, True),
    ("layer3 ×2\n128 → 256, s=2\nBasicBlockV2",     C_BIN,  C_BIN_E, True),
    ("layer4 ×2\n256 → 512, s=2\nBasicBlockV2",     C_BIN,  C_BIN_E, True),
    ("BN + GAP\n→ 512",                             C_NORM, C_NORM_E,False),
    ("FC\n512 → 100",                               C_HEAD, C_HEAD_E,False),
]
x = 0.2
for i, (txt, fc, ec, _bin) in enumerate(stages):
    box(ax, x, y0, w, h, txt, fc, ec, fontsize=9,
        weight=("bold" if _bin else "normal"))
    if i > 0:
        arrow(ax, x - 0.18, y0 + h/2, x + 0.02, y0 + h/2)
    x += w + 0.18

ax.text(7.0, 3.85, "A1W1 ReActNet-lite v2  (ResNet18 backbone, 1-bit weight / 1-bit activation)",
        ha="center", va="center", fontsize=13, weight="bold")
ax.text(7.0, 3.45, "stem & classifier: full-precision    |    layer1–4: binary BasicBlockV2",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

leg_y = 0.55
items = [("full-precision", C_FP, C_FP_E),
         ("binary stage",   C_BIN, C_BIN_E),
         ("BN / pool",      C_NORM, C_NORM_E),
         ("classifier head",C_HEAD, C_HEAD_E)]
lx = 1.0
for lbl, fc, ec in items:
    box(ax, lx, leg_y, 0.4, 0.32, "", fc, ec)
    ax.text(lx + 0.5, leg_y + 0.16, lbl, va="center", fontsize=10)
    lx += 2.8

_O1 = "/home/yonggi/KAU/3-1/딥러닝/Termp/arch_reactnet_overview"
fig.savefig(_O1 + ".png", bbox_inches="tight", dpi=300)
fig.savefig(_O1 + ".svg", bbox_inches="tight")
fig.savefig(_O1 + ".pdf", bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Figure 2 — BasicBlockV2 detail (horizontal layout, PPT 16:9 friendly)
# =====================================================================
fig, ax = plt.subplots(figsize=(16.0, 5.4), dpi=300)
ax.set_xlim(0, 16.0); ax.set_ylim(0, 5.4); ax.axis("off")

# title strip (single-line — keeps headroom for the residual_a arc)
ax.text(8.0, 5.10, "BasicBlock  —  double-skip variant",
        ha="center", va="center", fontsize=15, weight="bold")

# horizontal spine — boxes from left to right
spine_y = 2.65          # vertical center of all boxes on the main spine
bh      = 0.85          # main box height
ch      = 0.55          # ⊕ circle/box size

# (label, width, fillcolor, edgecolor, weight, fontsize)
items_h = [
    ("x",                                 0.70, C_FP,   C_FP_E,   "bold",   11),
    ("BN1",                               0.95, C_NORM, C_NORM_E, "normal", 10),
    ("RSign\nsign(x−θ)",                  1.20, C_ACT,  C_ACT_E,  "normal",  9),
    ("BinConv 3×3\ns=stride",             1.30, C_BIN,  C_BIN_E,  "bold",    9),
    ("BN2",                               0.95, C_NORM, C_NORM_E, "normal", 10),
    ("RPReLU\nPReLU(x−γ)+β",              1.45, C_ACT,  C_ACT_E,  "normal",  9),
    ("⊕",                                  ch,  C_FP,   C_FP_E,   "bold",   15),
    ("BN3",                               0.95, C_NORM, C_NORM_E, "normal", 10),
    ("RSign\nsign(x−θ)",                  1.20, C_ACT,  C_ACT_E,  "normal",  9),
    ("BinConv 3×3\ns=1",                  1.30, C_BIN,  C_BIN_E,  "bold",    9),
    ("BN4",                               0.95, C_NORM, C_NORM_E, "normal", 10),
    ("RPReLU",                            1.00, C_ACT,  C_ACT_E,  "normal", 10),
    ("⊕",                                  ch,  C_FP,   C_FP_E,   "bold",   15),
    ("out",                               0.70, C_FP,   C_FP_E,   "bold",   11),
]

# compute x-positions so total fills xlim with uniform gaps
total_w = sum(it[1] for it in items_h)
n_gaps  = len(items_h) - 1
margin  = 0.25
avail   = 16.0 - 2 * margin
gap_x   = (avail - total_w) / n_gaps

xs = []                       # list of (xL, xR, ycenter, label_text)
xL = margin
for label, w, fc, ec, wt, fs in items_h:
    h = ch if label == "⊕" else bh
    box(ax, xL, spine_y - h/2, w, h, label, fc, ec,
        fontsize=fs, weight=wt)
    xs.append((xL, xL + w, spine_y, label))
    xL = xL + w + gap_x

# spine arrows between consecutive boxes
for i in range(len(xs) - 1):
    arrow(ax, xs[i][1], spine_y, xs[i+1][0], spine_y, color="#333", lw=1.2)

# indices used by the residual arcs
ix_x, ix_add1, ix_add2 = 0, 6, 12

xL_x,   xR_x,   _, _   = xs[ix_x]
xL_a1,  xR_a1,  _, _   = xs[ix_add1]
xL_a2,  xR_a2,  _, _   = xs[ix_add2]
cx_x   = (xL_x + xR_x) / 2
cx_a1  = (xL_a1 + xR_a1) / 2
cx_a2  = (xL_a2 + xR_a2) / 2

# ---- residual_a: x ─ (arc above) ─ ⊕1   with ShortcutDownsample annotation
arc_top_a = spine_y + 1.30
ax.add_patch(FancyArrowPatch(
    (cx_x, spine_y + bh/2), (cx_a1, spine_y + ch/2),
    connectionstyle="arc3,rad=-0.40",
    arrowstyle="-|>", mutation_scale=14,
    color=C_SKIP, lw=1.6, linestyle="--",
))
# ShortcutDownsample annotation centered above arc midpoint
sd_cx = (cx_x + cx_a1) / 2
box(ax, sd_cx - 1.55, arc_top_a + 0.05, 3.10, 0.65,
    "ShortcutDownsample\nAvgPool(s) + 0-pad ch   (if s>1 or Δch)",
    C_SHORT, C_SHORT_E, fontsize=9)
# residual_a label — above the SD annotation
ax.text(sd_cx, arc_top_a + 0.92, "residual_a",
        ha="center", va="center", fontsize=10, color="#444",
        style="italic", weight="bold")

# ---- residual_b: ⊕1 ─ (arc below) ─ ⊕2   (identity)
ax.add_patch(FancyArrowPatch(
    (cx_a1, spine_y - ch/2), (cx_a2, spine_y - ch/2),
    connectionstyle="arc3,rad=0.40",
    arrowstyle="-|>", mutation_scale=14,
    color=C_SKIP, lw=1.6, linestyle="--",
))
mid_b_cx = (cx_a1 + cx_a2) / 2
ax.text(mid_b_cx, spine_y - 1.25, "residual_b   (identity, optional)",
        ha="center", fontsize=10, color="#444",
        style="italic", weight="bold")

# inline legend chips — horizontal strip at the bottom
chips = [
    ("BN",                    C_NORM,  C_NORM_E),
    ("RSign / RPReLU",        C_ACT,   C_ACT_E),
    ("Binary Conv (1-bit)",   C_BIN,   C_BIN_E),
    ("FP tensor",             C_FP,    C_FP_E),
    ("ShortcutDownsample",    C_SHORT, C_SHORT_E),
]
chip_y = 0.35
lx = 0.40
for (lbl, fc, ec) in chips:
    box(ax, lx, chip_y, 0.22, 0.14, "", fc, ec)
    ax.text(lx + 0.30, chip_y + 0.07, lbl, va="center", fontsize=9)
    lx += 0.30 + max(1.6, len(lbl) * 0.10)
# dashed-line legend
ax.plot([lx, lx + 0.22], [chip_y + 0.07, chip_y + 0.07],
        color=C_SKIP, lw=1.5, ls="--")
ax.text(lx + 0.30, chip_y + 0.07, "skip / shortcut",
        va="center", fontsize=9)

_O2 = "/home/yonggi/KAU/3-1/딥러닝/Termp/arch_reactnet_block"
fig.savefig(_O2 + ".png", bbox_inches="tight", dpi=300)
fig.savefig(_O2 + ".svg", bbox_inches="tight")
fig.savefig(_O2 + ".pdf", bbox_inches="tight")
plt.close(fig)

print("wrote arch_reactnet_{overview,block}.{png,svg,pdf}")
