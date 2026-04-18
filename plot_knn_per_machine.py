import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_step(path: str):
    base = os.path.basename(path)
    m = re.search(r"step_(\d+)", base)
    if m:
        return int(m.group(1))
    return None


def _pick_csv(pattern: str, prefer: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Geen csv gevonden met pattern: {pattern}")

    if prefer == "last":
        last = [f for f in files if "knn_eval_last" in os.path.basename(f)]
        if last:
            return last[0]

    step_files = [(f, _extract_step(f)) for f in files]
    step_files = [(f, s) for f, s in step_files if s is not None]
    if step_files:
        step_files.sort(key=lambda x: x[1])
        return step_files[-1][0]

    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/knn_eval_*.csv")
    parser.add_argument(
        "--prefer",
        choices=["last", "highest_step"],
        default="highest_step",
        help="welke csv pakken als er meerdere zijn: 'last' = checkpoint_last, 'highest_step' = grootste step.",
    )
    parser.add_argument("--input-csv", default="", help="Direct pad naar 1 csv. Overschrijft pattern.")
    parser.add_argument("--out-png", default="results/knn_per_machine.png")
    args = parser.parse_args()

    csv_path = args.input_csv or _pick_csv(args.pattern, args.prefer)
    print(f"Using csv: {csv_path}")

    df = pd.read_csv(csv_path).set_index("machine")
    metrics = ["AUC_source", "AUC_target", "pAUC", "AUC_all"]
    metrics = [m for m in metrics if m in df.columns]

    machines = list(df.index)
    n_machines = len(machines)
    n_metrics = len(metrics)
    if n_machines == 0 or n_metrics == 0:
        raise ValueError("Geen machines of metrics in csv om te plotten")

    x = np.arange(n_machines)
    width = 0.8 / n_metrics

    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_machines), 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, metric in enumerate(metrics):
        values = df[metric].to_numpy(dtype=float)
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"KNN performance per machine\n({os.path.basename(csv_path)})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print(f"Saved: {args.out_png}")

    means = df[metrics].mean(numeric_only=True)
    print("\nGemiddeldes:")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
