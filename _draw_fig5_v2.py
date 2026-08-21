# Redraw Figure 5 from the R-3 quartile-bin results (no GPU needed).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

bins   = ["[0.60,0.85)", "[0.85,0.89)", "[0.89,0.94)", "[0.94,1.00]"]
n      = [1980, 1927, 2091, 2002]
pos    = [42.4, 46.5, 47.3, 40.4]
acc    = {"Baseline":        [0.6753, 0.7047, 0.6978, 0.7103],
          "SimCSE":          [0.6899, 0.7161, 0.7250, 0.7468],
          "JEPA-Reg (ours)": [0.7692, 0.7888, 0.8092, 0.8327]}
jepa_pos = [0.8560, 0.8896, 0.9040, 0.9209]
jepa_neg = [0.7053, 0.7010, 0.7239, 0.7728]
colors = {"Baseline": "#4C72B0", "SimCSE": "#C44E52", "JEPA-Reg (ours)": "#55A868"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(4); w = 0.26
for i, (name, vals) in enumerate(acc.items()):
    ax1.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name], alpha=0.9)
for xi, (nn, pp) in enumerate(zip(n, pos)):
    ax1.text(xi, 0.60, f"n={nn}\n{pp:.0f}% pos", ha="center", fontsize=7.5, color="#333")
ax1.set_xticks(x); ax1.set_xticklabels(bins, fontsize=8.5)
ax1.set_ylim(0.55, 0.95)
ax1.set_xlabel("Jaccard-overlap quartile (PAWS test)")
ax1.set_ylabel("Accuracy")
ax1.set_title("(a) Accuracy by overlap quartile", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

ax2.plot(x, jepa_pos, "o-", color="#2ECC71", lw=2, label="JEPA-Reg, paraphrase (y=1)")
ax2.plot(x, jepa_neg, "s-", color="#E74C3C", lw=2, label="JEPA-Reg, non-paraphrase (y=0)")
ax2.set_xticks(x); ax2.set_xticklabels(bins, fontsize=8.5)
ax2.set_ylim(0.55, 1.0)
ax2.set_xlabel("Jaccard-overlap quartile (PAWS test)")
ax2.set_ylabel("Per-class accuracy")
ax2.set_title("(b) JEPA-Reg per-class accuracy\n(gain is not a class-prior shift)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_e_overlap_v2.png", dpi=220)
print("saved fig_e_overlap_v2.png")
