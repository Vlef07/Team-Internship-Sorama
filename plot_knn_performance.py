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
                "hmean": hmean,
                "auc_all_mean": auc_all_mean,
                "file": path,
            }
        )

    if not rows:
        raise ValueError("Geen bruikbare csv files met step-nummer gevonden")

    perf = pd.DataFrame(rows).sort_values("step")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    perf.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(perf["step"], perf["hmean"], marker="o", linewidth=2, label="Harmonic mean")
    plt.plot(perf["step"], perf["auc_all_mean"], marker="s", linewidth=2, label="Mean AUC_all")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Score")
    plt.title("KNN performance vs checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)

    print("Saved:")
    print(f"- {args.out_csv}")
    print(f"- {args.out_png}")
    print("\nTop by harmonic mean:")
    print(perf.sort_values("hmean", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
