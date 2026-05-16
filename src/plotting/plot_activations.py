"""Plot ReLU / PReLU / GELU / Swish — report §3.6 activation ablation figure."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import erf

mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"]  = 42
mpl.rcParams["svg.fonttype"]  = "none"

x = np.linspace(-4.0, 4.0, 1001)

def relu(x):  return np.maximum(0, x)
def prelu(x, a=0.25): return np.where(x >= 0, x, a * x)
def gelu(x):  return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))   # exact (CDF form)
def swish(x): return x / (1.0 + np.exp(-x))                   # SiLU

curves = [
    ("ReLU",            relu(x),        "#1f77b4", "-"),
    ("PReLU (α = 0.25)", prelu(x, 0.25), "#d62728", "-"),
    ("GELU",            gelu(x),        "#2ca02c", "-"),
    ("Swish (SiLU)",    swish(x),       "#ff7f0e", "-"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

for name, y, c, ls in curves:
    ax.plot(x, y, label=name, color=c, linestyle=ls, linewidth=2.0)

# axes through origin
ax.axhline(0, color="#888", linewidth=0.8)
ax.axvline(0, color="#888", linewidth=0.8)

ax.set_xlim(-4, 4)
ax.set_ylim(-1.2, 4.2)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.set_title("Activation functions compared in §3.6",
             fontsize=13, weight="bold")

ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
ax.legend(loc="upper left", framealpha=0.95, fontsize=11)

# small annotation: result summary
ax.text(0.98, 0.04,
        "Step 5 result (teacher, 80 ep FT)\n"
        "GELU 82.46 %  >  ReLU 82.38 %\n"
        "PReLU 81.24–81.42 %   Swish 80.60 %",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#fafafa", edgecolor="#bbbbbb"))

fig.tight_layout()
_O = "/home/yonggi/KAU/3-1/딥러닝/Termp/activations"
fig.savefig(_O + ".png", bbox_inches="tight", dpi=300)
fig.savefig(_O + ".svg", bbox_inches="tight")
fig.savefig(_O + ".pdf", bbox_inches="tight")
plt.close(fig)
print("wrote activations.{png,svg,pdf}")
