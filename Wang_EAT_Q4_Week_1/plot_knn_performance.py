"""KNN performance over training steps. Hoort bij STEP 4/4 in run_wang2025_snellius.slurm."""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _extract_step(path: str):
    base = os.path.basename(path)
    m = re.search(r"step_(\d+)", base)
    if m:
        return int(m.group(1))

    m = re.search(r"checkpoint_(\d+)", base)
    if m:
        return int(m.group(1))

    return None


def _variant_from_path(path: str) -> str:
    """Scheidt baseline-KNN van TSE-KNN voor aparte plotlijnen (geen zigzag op dezelfde step)."""
    base = os.path.basename(path)
    if "_tse_" in base:
        m = re.search(r"_tse_([^.]+)", base)
        suffix = m.group(1) if m else "tse"
        return f"TSE ({suffix})"
    return "Baseline (geen TSE op waveform)"


def _hmean_from_summary(df: pd.DataFrame) -> float:
    valid = df["AUC_all"].notna()
    if not valid.any():
        return float("nan")

    flat = []
    for machine in df.loc[valid].index:
        row = df.loc[machine]
        flat.extend([row["AUC_source"], row["AUC_target"], row["pAUC"]])

    flat = np.maximum(np.array(flat, dtype=float), np.finfo(float).eps)
    return float(stats.hmean(flat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/knn_eval_*.csv")
    parser.add_argument("--out-csv", default="results/knn_performance_over_steps.csv")
    parser.add_argument("--out-png", default="results/knn_performance_over_steps.png")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"Geen resultaten gevonden met pattern: {args.pattern}")

    rows = []
    for path in files:
        step = _extract_step(path)
        if step is None:
            continue

        df = pd.read_csv(path).set_index("machine")
        hmean = _hmean_from_summary(df)
        auc_all_mean = float(df.loc[df["AUC_all"].notna(), "AUC_all"].mean())

        rows.append(
            {
                "step": step,
                "variant": _variant_from_path(path),
                "hmean": hmean,
                "auc_all_mean": auc_all_mean,
                "file": path,
            }
        )

    if not rows:
        raise ValueError("Geen bruikbare csv files met step-nummer gevonden")

    perf = pd.DataFrame(rows)
    perf = perf.sort_values(["variant", "step"]).drop_duplicates(
        subset=["step", "variant"], keep="last"
    )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    perf.to_csv(args.out_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    variants = sorted(
        perf["variant"].unique(),
        key=lambda v: (0 if v.startswith("Baseline") else 1, v),
    )
    markers = ["o", "s", "^", "D", "v"]
    for i, variant in enumerate(variants):
        sub = perf.loc[perf["variant"] == variant].sort_values("step")
        kw = dict(
            linewidth=2,
            label=variant,
            marker=markers[i % len(markers)],
            markersize=5,
        )
        axes[0].plot(sub["step"], sub["hmean"], **kw)
        axes[1].plot(sub["step"], sub["auc_all_mean"], **kw)

    axes[0].set_ylabel("Harmonic mean")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("KNN vs checkpoint (baseline en TSE apart)")

    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Mean AUC_all")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)

    print("Saved:")
    print(f"- {args.out_csv}")
    print(f"- {args.out_png}")
    print("\nTop by harmonic mean (overall):")
    print(
        perf.sort_values("hmean", ascending=False)
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
