"""
Threshold-selection helper for the circumscribed-hypersphere checks: reads the
CSV already produced by lexica_hypersphere_mismatch_check.py (true match =
"inlier", closest non-matching prompt = "outlier") and, for each of the two
metrics below, plots inlier vs. outlier boxplots with the "current" threshold
marked, sweeps every value as a candidate cutoff, and reports the one that
best separates the two groups (max Youden's J = TPR - FPR) plus the ROC AUC:

  - raw:       full distance ||X-C|| vs. the fitted radius. Independent of
               `original_prompt_share` -- see check_within_hypersphere.
  - empirical: angle-from-A vs. the largest angle-from-A among samples run
               through the real inv_transform (i.e. WITH the configured
               `original_prompt_share` pull applied) -- see
               check_empirical_reach. This is the one that reflects the
               actually-deployed share=0.5 config.

Does not touch the GPU / re-embed anything -- pure post-hoc analysis on the
existing CSV.

Usage:
    python3 scripts/lexica_hypersphere_threshold_analysis.py \\
        [--csv outputs/lexica_hypersphere_mismatch_check/results_train.csv]
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--strategy", default="cosine_to_a",
                       choices=["cosine_to_a", "full_distance_to_c", "random"],
                       help="Must match the --strategy used by lexica_hypersphere_mismatch_check.py "
                            "when it produced the CSV being read.")
    parser.add_argument("--csv", default=None,
                       help="Overrides the CSV path implied by --split/--strategy.")
    parser.add_argument("--output-dir", default="outputs/lexica_hypersphere_mismatch_check")
    return parser.parse_args()


def analyze(df, inlier_col, outlier_col, current_threshold, title, unit, plot_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    n = len(df)
    inlier_vals = df[inlier_col].to_numpy()
    outlier_vals = df[outlier_col].to_numpy()

    # y_true=1 means "inlier" (true match); score = -value since smaller value => more inlier-like
    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = -np.concatenate([inlier_vals, outlier_vals])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    j_stat = tpr - fpr
    best_idx = int(np.argmax(j_stat))
    best_threshold = -thresholds[best_idx]
    best_tpr, best_fpr = tpr[best_idx], fpr[best_idx]

    current_tpr = (inlier_vals <= current_threshold).mean()
    current_fpr = (outlier_vals <= current_threshold).mean()

    summary_lines = [
        f"[{title}] n={n} inlier/outlier pairs",
        f"  current threshold: {current_threshold:.3f} {unit} -> TPR={current_tpr:.2f}, FPR={current_fpr:.2f}",
        f"  ROC AUC: {roc_auc:.3f}",
        f"  best-separating threshold (max Youden's J): {best_threshold:.3f} {unit} -> "
        f"TPR={best_tpr:.2f}, FPR={best_fpr:.2f}, J={j_stat[best_idx]:.2f}",
    ]
    print("\n".join(summary_lines))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([outlier_vals, inlier_vals], widths=0.6, patch_artist=True,
                   orientation="horizontal",
                   medianprops=dict(color=INK_PRIMARY, linewidth=2),
                   whiskerprops=dict(color=INK_SECONDARY), capprops=dict(color=INK_SECONDARY),
                   flierprops=dict(marker="o", markersize=3, markeredgecolor="none", alpha=0.4))
    for patch, color, flier in zip(bp["boxes"], [SERIES_ORANGE, SERIES_BLUE], bp["fliers"]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
        flier.set_markerfacecolor(color)
    ax.set_yticks([1, 2], ["outliers (closest mismatch)", "inliers (true match)"])
    ax.set_title(title, color=INK_PRIMARY, fontsize=12)
    ax.set_xlabel(unit, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.axvline(current_threshold, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax.text(current_threshold, 1.5, f" current threshold = {current_threshold:.2f}", color=INK_MUTED,
           fontsize=9, va="center", ha="left")
    ax.axvline(best_threshold, color=INK_PRIMARY, linewidth=1.5, linestyle=":")
    ax.text(best_threshold, 1.35, f" best-separating threshold = {best_threshold:.2f}",
           color=INK_PRIMARY, fontsize=9, va="center", ha="left")

    # Robust x-axis: a handful of pathological rows (e.g. a mis-parsed "subject" field hundreds of
    # characters long) can inflate distances by 40x and make the whole plot unreadable otherwise.
    # The boxplot/ROC/AUC above still use the full, unclipped data -- only the axis range here is robust.
    combined = np.concatenate([inlier_vals, outlier_vals])
    lo, hi = np.percentile(combined, [1, 99])
    margin = 0.08 * (hi - lo)
    ax.set_xlim(min(lo - margin, current_threshold, best_threshold) - margin, hi + margin)
    n_clipped = int(((combined < lo) | (combined > hi)).sum())
    if n_clipped:
        ax.text(0.99, 0.02, f"{n_clipped} extreme point(s) outside the shown range (not excluded from AUC/stats)",
               transform=ax.transAxes, color=INK_MUTED, fontsize=8, ha="right", va="bottom", style="italic")

    fig.suptitle(f"n={n}, ROC AUC={roc_auc:.2f}", color=INK_SECONDARY, fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(plot_path, dpi=150)
    print(f"  boxplot written to {plot_path}\n")

    return summary_lines


def main():
    args = parse_args()
    import pandas as pd

    csv_path = args.csv or os.path.join(
        "outputs/lexica_hypersphere_mismatch_check", f"results_{args.split}_{args.strategy}.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} (strategy={args.strategy})\n")

    all_summary = [f"strategy={args.strategy}"]

    # raw: full distance vs. fitted radius, independent of original_prompt_share
    # median, not mean -- a handful of pathological subjects (e.g. mis-parsed multi-hundred-char
    # "subject" fields) can inflate a small number of radii by 40x and skew the mean badly.
    current_radius = df["matched_raw_radius"].median()
    all_summary += analyze(
        df, "matched_raw_full_distance", "mismatched_raw_full_distance", current_radius,
        title=f"Raw (share-independent, mismatch={args.strategy}): full distance ||X-C|| vs. fitted radius",
        unit="embedding-space units",
        plot_path=os.path.join(args.output_dir, f"threshold_boxplot_raw_{args.strategy}.png"),
    )

    # empirical: angle-from-A vs. the max angle among real (share-pulled) samples
    current_angle_threshold = pd.concat([
        df["matched_emp_sample_angle_max"], df["mismatched_emp_sample_angle_max"],
    ]).mean()
    share = df["matched_emp_original_prompt_share"].iloc[0]
    all_summary += analyze(
        df, "matched_emp_angle_b", "mismatched_emp_angle_b", current_angle_threshold,
        title=f"Empirical (share={share}, mismatch={args.strategy}): angle-from-A vs. max sampled angle",
        unit="radians",
        plot_path=os.path.join(args.output_dir, f"threshold_boxplot_empirical_{args.strategy}.png"),
    )

    with open(os.path.join(args.output_dir, f"threshold_analysis_{args.strategy}.txt"), "w") as f:
        f.write("\n".join(all_summary))


if __name__ == "__main__":
    main()
