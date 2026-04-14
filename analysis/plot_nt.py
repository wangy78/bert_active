"""
Generate analysis figures for NT cross-task transfer experiments.
H3K4me3 -> enhancers (A1) and H3K4me3 -> enhancers_types (A2).

Fig 1 — A1 learning curves: all 5 conditions, mean ± std (F1 macro)
Fig 2 — A2 learning curves: all 5 conditions, mean ± std (F1 macro)
Fig 3 — Transfer gain: baseline vs best transfer, A1 vs A2 side-by-side
Fig 4 — Final performance boxplot: all conditions, both targets
Fig 5 — Frozen backbone comparison: frozen_badge vs frozen_doptimal (A1 & A2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
DATA_A1 = Path("analysis/data/nt_enhancer_runs.csv")
DATA_A2 = Path("analysis/data/nt_enhancer_types_runs.csv")
OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Condition display order and colors
CONDITIONS = [
    "baseline_random",
    "transfer_random",
    "transfer_badge",
    "frozen_badge",
    "frozen_doptimal",
]
COLORS = {
    "baseline_random":  "#888888",
    "transfer_random":  "#2196F3",
    "transfer_badge":   "#4CAF50",
    "frozen_badge":     "#FF9800",
    "frozen_doptimal":  "#F44336",
}
LABELS = {
    "baseline_random":  "Baseline (random, no pretrain)",
    "transfer_random":  "Transfer + Random",
    "transfer_badge":   "Transfer + BADGE",
    "frozen_badge":     "Frozen + BADGE",
    "frozen_doptimal":  "Frozen + D-Optimal",
}
LINESTYLES = {
    "baseline_random":  "-",
    "transfer_random":  "--",
    "transfer_badge":   "-",
    "frozen_badge":     "-.",
    "frozen_doptimal":  ":",
}
MARKER = dict(marker="o", markersize=3, markerfacecolor="white", markeredgewidth=1.4)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={
        "al/round":         "round",
        "al/eval_accuracy": "accuracy",
        "al/eval_f1_macro": "f1",
    })


def mean_std(df, x, y):
    grp = df.groupby(x)[y]
    xs = np.array(sorted(df[x].unique()))
    return xs, grp.mean()[xs].values, grp.std(ddof=1)[xs].fillna(0).values


def plot_learning_curves(df, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, metric, ylabel in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1 Macro"]):
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            xs, ys, ss = mean_std(sub, "n_labeled", metric)
            c = COLORS[cond]
            ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2,
                    linestyle=LINESTYLES[cond], **MARKER)
            ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.12)

        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path.name}")
    plt.close()


# ── Load data ───────────────────────────────────────────────────────────────
df_a1 = load(DATA_A1) if DATA_A1.exists() else None
df_a2 = load(DATA_A2) if DATA_A2.exists() else None

if df_a1 is None and df_a2 is None:
    print("No data files found. Run fetch_data_nt.py first.")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — A1 learning curves
# ═══════════════════════════════════════════════════════════════════════════
if df_a1 is not None:
    plot_learning_curves(
        df_a1,
        "A1 — H3K4me3 → Enhancers (Binary): Learning Curves\n(5 conditions, 3 seeds, mean ± std)",
        OUT_DIR / "nt_fig1_a1_learning_curves.png",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — A2 learning curves
# ═══════════════════════════════════════════════════════════════════════════
if df_a2 is not None:
    plot_learning_curves(
        df_a2,
        "A2 — H3K4me3 → Enhancer Types (3-Class): Learning Curves\n(5 conditions, 3 seeds, mean ± std)",
        OUT_DIR / "nt_fig2_a2_learning_curves.png",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Transfer gain: baseline vs transfer_badge, A1 vs A2
# ═══════════════════════════════════════════════════════════════════════════
pairs = []
if df_a1 is not None:
    pairs.append((df_a1, "A1: Enhancers (Binary)"))
if df_a2 is not None:
    pairs.append((df_a2, "A2: Enhancer Types (3-Class)"))

if pairs:
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle("Transfer Gain: Baseline vs Transfer+BADGE vs Frozen+D-Optimal\n(F1 Macro, mean ± std)",
                 fontsize=12, fontweight="bold")

    highlight = ["baseline_random", "transfer_badge", "frozen_doptimal"]
    for ax, (df, title) in zip(axes, pairs):
        ax.set_title(title, fontsize=10)
        for cond in highlight:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            xs, ys, ss = mean_std(sub, "n_labeled", "f1")
            c = COLORS[cond]
            ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2.5,
                    linestyle=LINESTYLES[cond], **MARKER)
            ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.15)

        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel("F1 Macro")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "nt_fig3_transfer_gain.png", dpi=200, bbox_inches="tight")
    print("Saved nt_fig3_transfer_gain.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — Final performance boxplot (both targets)
# ═══════════════════════════════════════════════════════════════════════════
records = []
for df, target_label in pairs:
    for cond in CONDITIONS:
        for seed in [42, 43, 44]:
            sub = df[(df["condition"] == cond) & (df["seed"] == seed)]
            if sub.empty:
                continue
            f1_final = sub.sort_values("n_labeled")["f1"].iloc[-1]
            records.append({"condition": cond, "target": target_label, "seed": seed, "f1": f1_final})

if records:
    final_df = pd.DataFrame(records)
    targets = final_df["target"].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle("Final F1 Macro Distribution (last round, 2100 labeled samples)\n(3 seeds per condition)",
                 fontsize=12, fontweight="bold")

    for ax, target in zip(axes, targets):
        ax.set_title(target, fontsize=10)
        sub = final_df[final_df["target"] == target]
        conds_present = [c for c in CONDITIONS if c in sub["condition"].values]
        box_data = [sub[sub["condition"] == c]["f1"].values for c in conds_present]

        bp = ax.boxplot(
            box_data, patch_artist=True, widths=0.45,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, cond in zip(bp["boxes"], conds_present):
            patch.set_facecolor(COLORS[cond])
            patch.set_alpha(0.75)

        for i, cond in enumerate(conds_present, start=1):
            ys = sub[sub["condition"] == cond]["f1"].values
            xs = np.random.default_rng(0).uniform(i - 0.12, i + 0.12, len(ys))
            ax.scatter(xs, ys, color=COLORS[cond], s=45, zorder=5,
                       edgecolors="white", linewidths=0.8)

        baseline_mean = sub[sub["condition"] == "baseline_random"]["f1"].mean()
        if not np.isnan(baseline_mean):
            ax.axhline(baseline_mean, color=COLORS["baseline_random"], linestyle="--",
                       linewidth=1.2, alpha=0.7, label=f"Baseline mean ({baseline_mean:.3f})")

        ax.set_xticks(range(1, len(conds_present) + 1))
        ax.set_xticklabels([LABELS[c] for c in conds_present], fontsize=7.5, rotation=15, ha="right")
        ax.set_ylabel("F1 Macro")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "nt_fig4_final_boxplot.png", dpi=200, bbox_inches="tight")
    print("Saved nt_fig4_final_boxplot.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — Frozen backbone: badge vs doptimal (A1 & A2 side-by-side)
# ═══════════════════════════════════════════════════════════════════════════
frozen_conds = ["frozen_badge", "frozen_doptimal", "baseline_random"]
if pairs:
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle("Frozen Backbone: BADGE vs D-Optimal Acquisition\n(F1 Macro, mean ± std)",
                 fontsize=12, fontweight="bold")

    for ax, (df, title) in zip(axes, pairs):
        ax.set_title(title, fontsize=10)
        for cond in frozen_conds:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            xs, ys, ss = mean_std(sub, "n_labeled", "f1")
            c = COLORS[cond]
            ax.plot(xs, ys, color=c, label=LABELS[cond], linewidth=2.5,
                    linestyle=LINESTYLES[cond], **MARKER)
            ax.fill_between(xs, ys - ss, ys + ss, color=c, alpha=0.15)

        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel("F1 Macro")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "nt_fig5_frozen_comparison.png", dpi=200, bbox_inches="tight")
    print("Saved nt_fig5_frozen_comparison.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Console summary table
# ═══════════════════════════════════════════════════════════════════════════
if records:
    print(f"\n{'='*75}")
    for target in final_df["target"].unique():
        sub = final_df[final_df["target"] == target]
        baseline_vals = sub[sub["condition"] == "baseline_random"]["f1"].values
        baseline_mean = baseline_vals.mean() if len(baseline_vals) else float("nan")

        print(f"\n{target}")
        print(f"  {'Condition':<28} {'F1 mean':>9} {'±std':>7}  {'vs baseline':>12}")
        print(f"  {'-'*60}")
        for cond in CONDITIONS:
            vals = sub[sub["condition"] == cond]["f1"].values
            if len(vals) == 0:
                continue
            fm, fs = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
            delta = fm - baseline_mean
            sign = "+" if delta >= 0 else ""
            print(f"  {LABELS[cond]:<28} {fm:>9.4f} {fs:>7.4f}  {sign}{delta:>+10.4f}")
    print(f"\n{'='*75}")
