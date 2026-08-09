"""
Control experiment for lexica_hypersphere_stats.py: for each subject A_i in the
sample, instead of testing its own true variation B_i, find some OTHER prompt
in the sample (excluding B_i itself) and test THAT against A_i's
hypersphere/empirical reach instead.

The point: if a mismatched B is *also* frequently judged "within" / "in
reach", that's evidence the raw radius (or the empirical reach) is too
permissive to be discriminative -- rather than just reflecting that Lexica
prompts for different subjects can legitimately sit close together.

Three mismatch-selection --strategy options:
  - cosine_to_a:        closest B by cosine similarity to A's own bare pooled
                         embedding. NOTE: this is the same quantity as the
                         empirical check's `angle_b` metric, so it's a biased
                         (circular) control for THAT metric specifically --
                         the mismatch is handpicked to score well on the exact
                         axis being tested. Still a fair control for the raw
                         (full-distance-to-C) metric, which uses a different
                         reference point.
  - full_distance_to_c: closest B by plain Euclidean distance to A's fitted
                         sphere center C -- a fair control for the empirical
                         angle-from-A metric, since selection and evaluation
                         are no longer the same quantity.
  - random:              a uniformly random other B, seeded for
                         reproducibility. No selection-metric bias at all, at
                         the cost of being a much weaker (non-adversarial)
                         control.

Writes a CSV (suffixed by strategy) with both the matched and mismatched
result per row, plus a comparison boxplot, so the two can be judged side by
side.

Usage:
    python3 scripts/lexica_hypersphere_mismatch_check.py [--limit N] [--split train] \\
        [--strategy {cosine_to_a,full_distance_to_c,random}]
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline

from prototype.constants import RecommendationType
from prototype.user_profile_host.user_profile_host import UserProfileHost


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=20,
                       help="Sample size to draw closest-mismatch pairs from "
                            "(also the number of rows evaluated).")
    parser.add_argument("--output-dir", default="outputs/lexica_hypersphere_mismatch_check")
    parser.add_argument("--hf-model-name", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache-dir", default="./cache/")
    parser.add_argument("--n-embedding-axis", type=int, default=13)
    parser.add_argument("--axis-style", default="ordered")
    parser.add_argument("--no-embedding-center", action="store_true")
    parser.add_argument("--original-prompt-share", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--prompts-seed", type=int, default=42)
    parser.add_argument("--strategy", default="cosine_to_a",
                       choices=["cosine_to_a", "full_distance_to_c", "random"],
                       help="How to pick the mismatched B for each A -- see module docstring.")
    parser.add_argument("--mismatch-seed", type=int, default=0,
                       help="Seed for --strategy random.")
    parser.add_argument("--log-every", type=int, default=10,
                       help="Print a progress/ETA line every N subjects, in each of the two loops.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading LexicaDataset split={args.split!r}, first {args.limit} rows...")
    dataset = load_dataset("vera365/lexica_dataset", split=args.split)
    dataset = dataset.select_columns(["subject", "prompt"])
    dataset = dataset.select(range(min(args.limit, len(dataset))))
    subjects = dataset["subject"]
    prompts = dataset["prompt"]
    n = len(subjects)
    print(f"{n} rows.")

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.hf_model_name, cache_dir=args.cache_dir, torch_dtype=torch.float16,
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Building one UserProfileHost per subject (also fits each A's hypersphere)...")
    hosts = []
    t0 = time.time()
    for i, subject in enumerate(subjects):
        host = UserProfileHost(
            original_prompt=subject,
            recommendation_type=RecommendationType.HYPERSPHERICAL_RANDOM,
            stable_dif_pipe=pipe,
            n_embedding_axis=args.n_embedding_axis,
            axis_style=args.axis_style,
            use_embedding_center=not args.no_embedding_center,
            n_latent_axis=0,
            prompts_seed=args.prompts_seed,
            cache_dir=args.cache_dir,
            original_prompt_share=args.original_prompt_share,
        )
        hosts.append(host)
        if (i + 1) % args.log_every == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            eta_min = (n - i - 1) / rate / 60 if rate > 0 else float("nan")
            print(f"  [{i + 1}/{n}] fit A = {subject!r}  ({rate:.2f} subjects/s, ETA {eta_min:.1f} min)")

    print("Embedding all B prompts once (reused for the mismatch selection and the checks)...")
    with torch.no_grad():
        b_pooled = torch.stack([hosts[0].clip_embedding(p)[1] for p in prompts]).float()
        a_pooled = torch.stack([h.pooled_prompt_embedding for h in hosts]).float()
        c_pooled = torch.stack([h.hyperspherical_center_pooled for h in hosts]).float()

    # Always computed for reporting, regardless of --strategy.
    a_norm = torch.nn.functional.normalize(a_pooled, dim=-1)
    b_norm = torch.nn.functional.normalize(b_pooled, dim=-1)
    sim = a_norm @ b_norm.T  # sim[i, j] = cosine similarity of A_i to B_j

    if args.strategy == "cosine_to_a":
        score = sim
        pick_highest = True
    elif args.strategy == "full_distance_to_c":
        score = torch.cdist(c_pooled, b_pooled)  # score[i, j] = ||C_i - B_j|| (lower = closer)
        pick_highest = False
    else:  # random
        generator = torch.Generator().manual_seed(args.mismatch_seed)
        score = torch.rand((n, n), generator=generator)  # arbitrary tie-break-free scores; pick highest
        pick_highest = True

    rows = []
    t0 = time.time()
    for i in range(n):
        row_score = score[i].clone()
        row_score[i] = -float("inf") if pick_highest else float("inf")  # exclude the true match
        j_star = int((torch.argmax if pick_highest else torch.argmin)(row_score).item())

        matched = hosts[i].check_within_hypersphere(prompts[i])
        matched_emp = hosts[i].check_empirical_reach(prompts[i], n_samples=args.n_samples)
        mismatched = hosts[i].check_within_hypersphere(prompts[j_star])
        mismatched_emp = hosts[i].check_empirical_reach(prompts[j_star], n_samples=args.n_samples)

        rows.append({
            "row_index": i,
            "subject": subjects[i],
            "matched_prompt": prompts[i],
            "mismatched_prompt": prompts[j_star],
            "mismatch_source_index": j_star,
            "selection_strategy": args.strategy,
            "selection_score_mismatch": score[i, j_star].item(),
            "cosine_sim_mismatch_to_A": sim[i, j_star].item(),
            "cosine_sim_matched_to_A": sim[i, i].item(),
            "full_distance_mismatch_to_C": torch.linalg.norm(b_pooled[j_star] - c_pooled[i]).item(),
            "full_distance_matched_to_C": torch.linalg.norm(b_pooled[i] - c_pooled[i]).item(),
            **{f"matched_raw_{k}": v for k, v in matched.items()},
            **{f"matched_emp_{k}": v for k, v in matched_emp.items()},
            **{f"mismatched_raw_{k}": v for k, v in mismatched.items()},
            **{f"mismatched_emp_{k}": v for k, v in mismatched_emp.items()},
        })
        if (i + 1) % args.log_every == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            eta_min = (n - i - 1) / rate / 60 if rate > 0 else float("nan")
            print(f"  [{i + 1}/{n}] ({args.strategy}) mismatch is row {j_star} "
                 f"(cos={sim[i, j_star].item():.3f}): {prompts[j_star][:80]!r}  "
                 f"({rate:.2f} subjects/s, ETA {eta_min:.1f} min)")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f"results_{args.split}_{args.strategy}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")

    summarize(df, args.output_dir, args.split, args.strategy, args.original_prompt_share, args.n_samples)


def summarize(df, output_dir, split, strategy, share, n_samples):
    import matplotlib.pyplot as plt

    n = len(df)
    print(f"\n[{strategy}] Mean cosine similarity of the closest mismatch to A: "
         f"{df['cosine_sim_mismatch_to_A'].mean():.3f} "
         f"(matched pairs average {df['cosine_sim_matched_to_A'].mean():.3f})")

    pct_raw_matched = 100 * df["matched_raw_within"].astype(bool).mean()
    pct_raw_mismatched = 100 * df["mismatched_raw_within"].astype(bool).mean()
    pct_full_matched = 100 * df["matched_raw_within_full"].astype(bool).mean()
    pct_full_mismatched = 100 * df["mismatched_raw_within_full"].astype(bool).mean()
    pct_emp_matched = 100 * df["matched_emp_within"].astype(bool).mean()
    pct_emp_mismatched = 100 * df["mismatched_emp_within"].astype(bool).mean()

    summary_lines = [
        f"strategy={strategy}, n={n}, original_prompt_share={share}, n_samples={n_samples}",
        f"mean cosine(A, closest mismatch)={df['cosine_sim_mismatch_to_A'].mean():.3f}  "
        f"vs. mean cosine(A, true match)={df['cosine_sim_matched_to_A'].mean():.3f}",
        "",
        f"raw (in-plane) check -- matched within: {pct_raw_matched:.1f}%   "
        f"closest-mismatch within: {pct_raw_mismatched:.1f}%",
        f"full-distance check  -- matched within: {pct_full_matched:.1f}%   "
        f"closest-mismatch within: {pct_full_mismatched:.1f}%",
        f"empirical check      -- matched within: {pct_emp_matched:.1f}%   "
        f"closest-mismatch within: {pct_emp_mismatched:.1f}%",
    ]
    print("\n" + "\n".join(summary_lines))
    with open(os.path.join(output_dir, f"summary_{split}_{strategy}.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    # --- Plot -------------------------------------------------------------
    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#898781"
    gridline = "#e1e0d9"
    baseline = "#c3c2b7"
    series_blue = "#2a78d6"
    series_orange = "#eb6834"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": ink_primary,
        "axes.edgecolor": baseline,
        "axes.labelcolor": ink_primary,
        "xtick.color": ink_muted,
        "ytick.color": ink_muted,
        "figure.facecolor": surface,
        "axes.facecolor": surface,
    })

    def paired_boxplot(ax, matched_values, mismatched_values, title, xlabel, vline=None):
        bp = ax.boxplot([mismatched_values, matched_values], vert=False, widths=0.6, patch_artist=True,
                       medianprops=dict(color=ink_primary, linewidth=2),
                       whiskerprops=dict(color=ink_secondary), capprops=dict(color=ink_secondary),
                       flierprops=dict(marker="o", markersize=3, markeredgecolor="none", alpha=0.4))
        colors = [series_orange, series_blue]
        for patch, color, flier in zip(bp["boxes"], colors, bp["fliers"]):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.35)
            flier.set_markerfacecolor(color)
        ax.set_yticks([1, 2], ["closest mismatch", "true match"])
        ax.set_title(title, color=ink_primary, fontsize=11)
        ax.set_xlabel(xlabel, color=ink_secondary)
        ax.grid(axis="x", color=gridline, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        if vline is not None:
            ax.axvline(vline, color=ink_muted, linewidth=1, linestyle="--")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    fig.suptitle(f"LexicaDataset ({split}): true match vs. mismatch (strategy={strategy}), n={n}\n"
                f"original_prompt_share={share}, n_samples={n_samples}", color=ink_primary)

    paired_boxplot(axes[0], df["matched_raw_relative_distance"], df["mismatched_raw_relative_distance"],
                  "Raw (in-plane only): relative distance to sphere boundary",
                  "distance / radius  (0 = boundary, <0 = inside)", vline=0)
    paired_boxplot(axes[1], df["matched_raw_distance_full"], df["mismatched_raw_distance_full"],
                  "Full Euclidean distance ||X-C|| minus radius",
                  "embedding-space units  (<0 = inside)", vline=0)
    paired_boxplot(axes[2], df["matched_emp_distance"], df["mismatched_emp_distance"],
                  "Empirical: angle to A minus max sampled angle", "radians  (<0 = inside empirical reach)",
                  vline=0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = os.path.join(output_dir, f"boxplots_{split}_{strategy}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Boxplots written to {plot_path}")


if __name__ == "__main__":
    main()
