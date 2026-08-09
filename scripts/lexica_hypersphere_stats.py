"""
For every (subject, prompt) pair in the LexicaDataset
(https://huggingface.co/datasets/vera365/lexica_dataset), fits the circumscribed
hypersphere around `subject` (prompt A) and checks whether `prompt` (variation B)
lies within it, two ways:

  - "raw": UserProfileHost.check_within_hypersphere() -- tests B against the raw,
    pre-slerp fitted sphere (C, r, Q). Independent of `original_prompt_share`.
  - "empirical": UserProfileHost.check_empirical_reach() -- tests B against the
    empirical distribution of what the tool's own recommender could actually
    produce for A, i.e. after the real generation-time slerp-toward-A pull
    controlled by `original_prompt_share` (0.5 in the deployed config). See
    check_empirical_reach's docstring for why this can't be a closed-form
    adjustment of the raw check.

Both are recorded per row (prefixed raw_*/emp_*) so they can be compared
directly. Writes to a resumable CSV and, at the end, prints summary statistics
and saves boxplots.

Usage:
    python3 scripts/lexica_hypersphere_stats.py [--limit N] [--split train]

Can be interrupted and re-run with the same --output-dir to resume: already
processed rows (by row index) are skipped.
"""
import argparse
import csv
import os
import sys
import time

# Make the script runnable from any cwd: `prototype/` loads a JSON via a path
# relative to the repo root (see load_user_profile_host), so both sys.path and
# the working directory need to point at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline

from prototype.constants import RecommendationType
from prototype.user_profile_host.user_profile_host import UserProfileHost

RAW_FIELDS = ["within", "distance", "relative_distance", "in_plane_distance", "radius"]
EMP_FIELDS = ["within", "distance", "percentile", "angle_b", "sample_angle_min",
             "sample_angle_median", "sample_angle_max", "original_prompt_share", "n_samples"]
CSV_FIELDS = (["row_index", "subject", "prompt"]
             + [f"raw_{f}" for f in RAW_FIELDS]
             + [f"emp_{f}" for f in EMP_FIELDS]
             + ["error"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"],
                       help="LexicaDataset split to process.")
    parser.add_argument("--limit", type=int, default=None,
                       help="Only process the first N rows (for quick test runs).")
    parser.add_argument("--output-dir", default="outputs/lexica_hypersphere_stats",
                       help="Directory for the results CSV, summary stats, and plots.")
    parser.add_argument("--hf-model-name", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache-dir", default="./cache/")
    parser.add_argument("--n-embedding-axis", type=int, default=13,
                       help="Number of axis prompts used to fit the circumscribed hypersphere.")
    parser.add_argument("--axis-style", default="ordered")
    parser.add_argument("--no-embedding-center", action="store_true",
                       help="Disable use_embedding_center (see UserProfileHost docstring).")
    parser.add_argument("--original-prompt-share", type=float, default=0.5,
                       help="Matches the deployed config (configs/config.yaml); controls how "
                            "strongly real recommendations are pulled towards A. Used only by "
                            "the empirical check.")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of sampled recommendations for the empirical check.")
    parser.add_argument("--prompts-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                       help="Flush the CSV to disk every N processed rows.")
    return parser.parse_args()


def load_progress(csv_path):
    """Returns the number of rows already written, so a re-run can skip them."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"results_{args.split}.csv")

    print(f"Loading LexicaDataset split={args.split!r}...")
    dataset = load_dataset("vera365/lexica_dataset", split=args.split)
    dataset = dataset.select_columns(["subject", "prompt"])
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    n_rows = len(dataset)
    print(f"{n_rows} rows to process.")

    already_done = load_progress(csv_path)
    if already_done > 0:
        print(f"Resuming: {already_done} rows already in {csv_path}, skipping those.")

    print("Loading SDXL pipeline (shared across all rows)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.hf_model_name, cache_dir=args.cache_dir, torch_dtype=torch.float16,
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    start_time = time.time()
    processed_since_checkpoint = 0
    n_errors = 0
    for row_index in range(already_done, n_rows):
        row = dataset[row_index]
        subject, prompt = row["subject"], row["prompt"]
        record = {"row_index": row_index, "subject": subject, "prompt": prompt, "error": ""}
        try:
            if not subject or not prompt:
                raise ValueError("empty subject or prompt")
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
            raw_result = host.check_within_hypersphere(prompt)
            emp_result = host.check_empirical_reach(prompt, n_samples=args.n_samples)
            record.update({f"raw_{k}": v for k, v in raw_result.items()})
            record.update({f"emp_{k}": v for k, v in emp_result.items()})
        except Exception as e:
            n_errors += 1
            record["error"] = f"{type(e).__name__}: {e}"

        writer.writerow(record)
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= args.checkpoint_every:
            csv_file.flush()
            elapsed = time.time() - start_time
            done = row_index + 1 - already_done
            rate = done / elapsed
            remaining = n_rows - (row_index + 1)
            eta_min = remaining / rate / 60 if rate > 0 else float("nan")
            print(f"[{row_index + 1}/{n_rows}] {rate:.2f} rows/s, "
                 f"{n_errors} errors so far, ETA {eta_min:.1f} min")
            processed_since_checkpoint = 0

    csv_file.close()
    print(f"Done. Results written to {csv_path} ({n_errors} errors out of {n_rows - already_done} new rows).")

    summarize(csv_path, args.output_dir, args.split)


def summarize(csv_path, output_dir, split):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    n_total = len(df)
    df = df[df["error"].fillna("") == ""].copy()
    n_ok = len(df)
    print(f"\n{n_ok}/{n_total} rows had a valid result ({n_total - n_ok} errors).")
    if n_ok == 0:
        print("No valid rows to summarize.")
        return

    df["raw_within"] = df["raw_within"].astype(bool)
    df["emp_within"] = df["emp_within"].astype(bool)
    pct_raw_within = 100 * df["raw_within"].mean()
    pct_emp_within = 100 * df["emp_within"].mean()

    raw_cols = ["raw_distance", "raw_relative_distance", "raw_in_plane_distance", "raw_radius"]
    emp_cols = ["emp_distance", "emp_percentile", "emp_angle_b", "emp_sample_angle_min",
               "emp_sample_angle_median", "emp_sample_angle_max"]
    stats = df[raw_cols + emp_cols].describe()
    print(f"\nRaw (pre-slerp) sphere: {pct_raw_within:.1f}% of B within.")
    print(f"Empirical (real recommender, original_prompt_share={df['emp_original_prompt_share'].iloc[0]}): "
         f"{pct_emp_within:.1f}% of B within reach.\n")
    print(stats.to_string())

    stats_path = os.path.join(output_dir, f"summary_{split}.txt")
    with open(stats_path, "w") as f:
        f.write(f"{n_ok}/{n_total} rows valid ({n_total - n_ok} errors)\n")
        f.write(f"raw: {pct_raw_within:.1f}% within the (pre-slerp) circumscribed hypersphere\n")
        f.write(f"empirical: {pct_emp_within:.1f}% within the tool's actual sampled reach "
               f"(original_prompt_share={df['emp_original_prompt_share'].iloc[0]})\n\n")
        f.write(stats.to_string())
    print(f"\nSummary written to {stats_path}")

    # --- Plots -----------------------------------------------------------
    # Palette: minimal single-hue academic style (blue on a light surface, ink-colored text).
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

    def boxplot(ax, values, title, xlabel, color=series_blue, vline=None):
        ax.boxplot(values, vert=False, widths=0.6, patch_artist=True,
                  medianprops=dict(color=ink_primary, linewidth=2),
                  boxprops=dict(facecolor=color, edgecolor=color, alpha=0.35),
                  whiskerprops=dict(color=ink_secondary),
                  capprops=dict(color=ink_secondary),
                  flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                                 markeredgecolor="none", alpha=0.4))
        ax.set_yticks([])
        ax.set_title(title, color=ink_primary, fontsize=11)
        ax.set_xlabel(xlabel, color=ink_secondary)
        ax.grid(axis="x", color=gridline, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        if vline is not None:
            ax.axvline(vline, color=ink_muted, linewidth=1, linestyle="--")

    share = df["emp_original_prompt_share"].iloc[0]
    n_samples = int(df["emp_n_samples"].iloc[0])
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    fig.suptitle(f"LexicaDataset ({split}): B vs. A's reachable region, n={n_ok}\n"
                f"original_prompt_share={share}, n_samples={n_samples}",
                color=ink_primary)

    boxplot(axes[0][0], df["raw_relative_distance"], "Raw: relative distance to sphere boundary",
           "distance / radius  (0 = boundary, <0 = inside)", color=series_blue, vline=0)
    boxplot(axes[0][1], df["raw_distance"], "Raw: absolute distance to sphere boundary",
           "embedding-space units  (<0 = inside)", color=series_blue, vline=0)
    boxplot(axes[1][0], df["emp_distance"], "Empirical: angle to A minus max sampled angle",
           "radians  (<0 = inside empirical reach)", color=series_orange, vline=0)
    boxplot(axes[1][1], df["emp_percentile"], "Empirical: percentile of B among sampled angles",
           "fraction of samples at least as far as B  (0 = closest, 1 = farthest)",
           color=series_orange, vline=None)
    boxplot(axes[2][0], df["raw_radius"], "Raw: hypersphere radius (per subject)", "embedding-space units",
           color=series_blue)
    boxplot(axes[2][1], df["emp_sample_angle_max"], "Empirical: max sampled angle-from-A (per subject)",
           "radians", color=series_orange)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = os.path.join(output_dir, f"boxplots_{split}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Boxplots written to {plot_path}")


if __name__ == "__main__":
    main()
