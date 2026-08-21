# Regenerate Fig 3 (PAWS seed stability) and Fig 4 (low-resource MRPC) from the fresh run.
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(here, "..", "results", "all_results_fresh_v2.json"), encoding="utf-8"))["bert-base-uncased"]

COL = {"baseline": "#4C72B0", "simcse": "#C44E52", "jepa": "#55A868"}
LBL = {"baseline": "Baseline", "simcse": "SimCSE", "jepa": "JEPA-Reg (ours)"}

# ---- Figure 3: PAWS seed scatter ----
fig, ax = plt.subplots(figsize=(5.2, 4.0))
rng = np.random.default_rng(0)
for i, mt in enumerate(("baseline", "simcse", "jepa")):
    f1s = d["paws"][mt]["f1s"]
    mu, sd = np.mean(f1s), np.std(f1s)
    ax.bar(i, mu, 0.5, color=COL[mt], alpha=0.75, yerr=sd, capsize=5,
           error_kw={"elinewidth": 2}, zorder=2)
    j = rng.uniform(-0.08, 0.08, len(f1s))
    ax.scatter([i + jj for jj in j], f1s, color="black", s=55, zorder=5, alpha=0.85)
    ax.text(i, mu + sd + 0.008, f"$\\sigma$={sd:.3f}", ha="center", fontsize=9,
            fontweight="bold", color=("darkred" if sd > 0.02 else "darkgreen"))
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([LBL[m] for m in ("baseline", "simcse", "jepa")], fontsize=9)
ax.set_ylabel("PAWS test F1"); ax.set_ylim(0.60, 0.85)
worst = min(d["paws"]["baseline"]["f1s"])
ax.annotate("seed\ncollapse", xy=(0, worst), xytext=(0.55, worst - 0.02),
            arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=8, color="darkred")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(here, "figs", "fig3_paws_variance.png"), dpi=220); plt.close()

# ---- Figure 4: low-resource MRPC ----
lr = d["lr_mrpc"]
keys = sorted(lr.keys(), key=float)
pct = [float(k) * 100 for k in keys]
fig, ax = plt.subplots(figsize=(5.2, 4.0))
for mt in ("baseline", "simcse", "jepa"):
    means = [lr[k][mt]["f1_mean"] for k in keys]
    stds  = [lr[k][mt]["f1_std"] for k in keys]
    ax.plot(pct, means, "o-", label=LBL[mt], color=COL[mt], lw=2.2, ms=6)
    ax.fill_between(pct, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)], alpha=0.15, color=COL[mt])
ax.set_xlabel("% of MRPC training data"); ax.set_ylabel("Test F1")
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(here, "figs", "fig4_lowresource.png"), dpi=220); plt.close()
print("fig3 + fig4 regenerated from fresh run")
