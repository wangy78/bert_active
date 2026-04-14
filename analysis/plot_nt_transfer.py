"""
Generate analysis figures for NT cross-task transfer learning experiments.

H3K4me3 (source) -> enhancers / enhancer_types (target)

Fig 1 — A1 (enhancers, binary): 5-condition learning curves, mean ± std
Fig 2 — A2 (enhancer_types, 3-class): 5-condition learning curves, mean ± std
Fig 3 — Transfer gain: baseline vs transfer_random vs transfer_badge (both targets)
Fig 4 — Frozen backbone comparison: frozen_badge vs frozen_doptimal vs baseline (both targets)
Fig 5 — Final performance boxplot: all conditions, both targets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE = Path("analysis/data/nt_transfer_runs.csv")
OUT_DIR   = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline_random":   "#888888",
    "transfer_random":   "#FF9800",
    "transfer_badge":    "#4CAF50",
    "frozen_badge":      "#2196F3",
    "frozen_doptimal":   "#9C27B0",
}
LABELS = {
    "baseline_random":   "Baseline (random, no pretrain)",
    "transfer_random":   "Transfer + Random AL",
    "transfer_badge":    "Transfer + BADGE AL",
    "frozen_badge":      "Frozen backbone + BADGE",
    "frozen_doptimal":   "Frozen backbone + D-Optimal",
}
CONDITION_ORDER = [
    "baseline_random",
    "transfer_random",
    "transfer_badge",
    "frozen_badge",
    "frozen_doptimal",
]
MARKER = dict(marker="o", markersize=3, markerfacecolor="white", markeredgewidth=1.4)

df = pd.read_csv(DATA_FILE)
df = df.rename(columns={
    "al/round":         "round",
    "al/eval_accuracy": "accuracy",
    "al/eval_f1_macro": "f1",
})


def mean_std(sub, x, y):
    grp = sub.groupby(x)[y]
    xs  = np.array(sorted(sub[x].unique()))
    return xs, grp.mean()[xs].values, grp.std()[xs].values


def plot_learning_curves(ax, data, metric, ylabel, title):
    for cond in CONDITION_ORDER:
        sub = data[data["condition"] == cond]
        if sub.empty:
            continue
        xs, ys, ss = mean_std(sub, "n_labeled", metric)
        c = COLORS[cond]
        ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2, **MARKER)
        ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.12)
    ax.set_xlabel("Labeled Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — A1 enhancers (binary): 5-condition learning curves
# ═══════════════════════════════════════════════════════════════════════════════
a1 = df[df["target"] == "enhancers"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("A1 — H3K4me3 → Enhancers (Binary): Transfer Learning Comparison\n"
             "(3 seeds, mean ± std)", fontsize=12, fontweight="bold")

for ax, metric, ylabel in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1 Macro"]):
    plot_learning_curves(ax, a1, metric, ylabel, f"{ylabel} vs Labeled Samples")

plt.tight_layout()
fig.savefig(OUT_DIR / "nt_transfer_fig1_enhancers_curves.png", dpi=200, bbox_inches="tight")
print("Saved nt_transfer_fig1_enhancers_curves.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — A2 enhancer_types (3-class): 5-condition learning curves
# ═══════════════════════════════════════════════════════════════════════════════
a2 = df[df["target"] == "enhancer_types"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("A2 — H3K4me3 → Enhancer Types (3-Class): Transfer Learning Comparison\n"
             "(3 seeds, mean ± std)", fontsize=12, fontweight="bold")

for ax, metric, ylabel in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1 Macro"]):
    plot_learning_curves(ax, a2, metric, ylabel, f"{ylabel} vs Labeled Samples")

plt.tight_layout()
fig.savefig(OUT_DIR / "nt_transfer_fig2_enhancer_types_curves.png", dpi=200, bbox_inches="tight")
print("Saved nt_transfer_fig2_enhancer_types_curves.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Transfer gain: baseline vs transfer_random vs transfer_badge
# ═══════════════════════════════════════════════════════════════════════════════
TRANSFER_CONDITIONS = ["baseline_random", "transfer_random", "transfer_badge"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Transfer Gain: Baseline vs Transfer (Full Fine-Tune)\n"
             "(3 seeds, mean ± std, F1 Macro)", fontsize=12, fontweight="bold")

for ax, (target, title) in zip(axes, [
    ("enhancers",      "A1 — Enhancers (Binary)"),
    ("enhancer_types", "A2 — Enhancer Types (3-Class)"),
]):
    sub = df[df["target"] == target]
    for cond in TRANSFER_CONDITIONS:
        csub = sub[sub["condition"] == cond]
        if csub.empty:
            continue
        xs, ys, ss = mean_std(csub, "n_labeled", "f1")
        c = COLORS[cond]
        ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2, **MARKER)
        ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.12)
    ax.set_xlabel("Labeled Samples")
    ax.set_ylabel("F1 Macro")
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "nt_transfer_fig3_transfer_gain.png", dpi=200, bbox_inches="tight")
print("Saved nt_transfer_fig3_transfer_gain.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Frozen backbone: baseline vs frozen_badge vs frozen_doptimal
# ═══════════════════════════════════════════════════════════════════════════════
FROZEN_CONDITIONS = ["baseline_random", "frozen_badge", "frozen_doptimal"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Frozen Backbone (Linear Probe): BADGE vs D-Optimal vs Baseline\n"
             "(3 seeds, mean ± std, F1 Macro)", fontsize=12, fontweight="bold")

for ax, (target, title) in zip(axes, [
    ("enhancers",      "A1 — Enhancers (Binary)"),
    ("enhancer_types", "A2 — Enhancer Types (3-Class)"),
]):
    sub = df[df["target"] == target]
    for cond in FROZEN_CONDITIONS:
        csub = sub[sub["condition"] == cond]
        if csub.empty:
            continue
        xs, ys, ss = mean_std(csub, "n_labeled", "f1")
        c = COLORS[cond]
        ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2, **MARKER)
        ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.12)
    ax.set_xlabel("Labeled Samples")
    ax.set_ylabel("F1 Macro")
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "nt_transfer_fig4_frozen_backbone.png", dpi=200, bbox_inches="tight")
print("Saved nt_transfer_fig4_frozen_backbone.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Final performance boxplot (all conditions, both targets)
# ═══════════════════════════════════════════════════════════════════════════════
records = []
for target in ["enhancers", "enhancer_types"]:
    for cond in CONDITION_ORDER:
        for seed in df["seed"].unique():
            sub = df[(df["target"] == target) & (df["condition"] == cond) & (df["seed"] == seed)]
            if sub.empty:
                continue
            f1_final = sub.sort_values("n_labeled")["f1"].iloc[-1]
            records.append({"target": target, "condition": cond, "seed": seed, "f1": f1_final})

final_df = pd.DataFrame(records)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Final F1 Macro Distribution (last AL round, all conditions)\n"
             "(3 seeds per condition)", fontsize=12, fontweight="bold")

for ax, (target, title) in zip(axes, [
    ("enhancers",      "A1 — Enhancers (Binary)"),
    ("enhancer_types", "A2 — Enhancer Types (3-Class)"),
]):
    sub = final_df[final_df["target"] == target]
    box_data = [sub[sub["condition"] == c]["f1"].values for c in CONDITION_ORDER]
    bp = ax.boxplot(
        box_data, patch_artist=True, widths=0.45,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, cond in zip(bp["boxes"], CONDITION_ORDER):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.75)

    for i, cond in enumerate(CONDITION_ORDER, start=1):
        ys = sub[sub["condition"] == cond]["f1"].values
        rng = np.random.default_rng(0)
        xs = rng.uniform(i - 0.12, i + 0.12, len(ys))
        ax.scatter(xs, ys, color=COLORS[cond], s=45, zorder=5,
                   edgecolors="white", linewidths=0.8)

    baseline_mean = sub[sub["condition"] == "baseline_random"]["f1"].mean()
    ax.axhline(baseline_mean, color=COLORS["baseline_random"], linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Baseline mean ({baseline_mean:.4f})")

    ax.set_xticks(range(1, len(CONDITION_ORDER) + 1))
    short_labels = ["Baseline", "Transfer\nRandom", "Transfer\nBADGE",
                    "Frozen\nBADGE", "Frozen\nD-Opt"]
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel("F1 Macro (last round)")
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "nt_transfer_fig5_final_boxplot.png", dpi=200, bbox_inches="tight")
print("Saved nt_transfer_fig5_final_boxplot.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary table
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print(f"{'Target':<16} {'Condition':<24} {'F1 mean':>9} {'±std':>7} {'vs baseline':>12}")
print(f"{'='*75}")

for target in ["enhancers", "enhancer_types"]:
    sub = final_df[final_df["target"] == target]
    baseline_mean = sub[sub["condition"] == "baseline_random"]["f1"].mean()
    for cond in CONDITION_ORDER:
        vals = sub[sub["condition"] == cond]["f1"].values
        if len(vals) == 0:
            continue
        fm, fs = vals.mean(), vals.std()
        delta = fm - baseline_mean
        sign = "+" if delta >= 0 else ""
        note = "baseline" if cond == "baseline_random" else f"{sign}{delta:+.4f}"
        print(f"{target:<16} {LABELS[cond]:<24} {fm:>9.4f} {fs:>7.4f} {note:>12}")
    print(f"{'-'*75}")
print(f"{'='*75}")
