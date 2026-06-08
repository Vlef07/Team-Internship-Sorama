"""this script draws a bar chart of knn auc scores per machine from result csv files"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_step(path: str):
    base = os.path.basename(path)
    m = re.search(r"step_(\d+)", base)
    if m:
        return int(m.group(1))
    return None


def _filter_paths(paths: list, variant: str) -> list:
    """variant: baseline = geen _tse_ in bestandsnaam; tse = wel _tse_; any = geen filter."""
    if variant == "any":
        return list(paths)
    out = []
    for p in paths:
        base = os.path.basename(p)
        has_tse = "_tse_" in base
        if variant == "baseline" and not has_tse:
            out.append(p)
        elif variant == "tse" and has_tse:
            out.append(p)
    return out


def _pick_csv(pattern: str, prefer: str, variant: str) -> str:
    files = sorted(glob.glob(pattern))
    files = _filter_paths(files, variant)
    if not files:
        return ""

    if prefer == "last":
        last = [f for f in files if "knn_eval_last" in os.path.basename(f)]
        if last:
            return last[0]

    step_files = [(f, _extract_step(f)) for f in files]
    step_files = [(f, s) for f, s in step_files if s is not None]
    if step_files:
        step_files.sort(key=lambda x: (x[1], x[0]))
        return step_files[-1][0]

    return files[-1]


def main():
    # read one or more knn csv files and save a bar chart png per machine
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/knn_eval_*.csv")
    parser.add_argument(
        "--variant",
        choices=["baseline", "tse", "any"],
        default="baseline",
        help=(
            "baseline: alleen csv zonder _tse_ in de naam (KNN zonder TSE op waveform). "
            "tse: alleen csv met _tse_ (KNN met TSE). "
            "any: oud gedrag, alle bestanden (kan dubbele step mengen, ongeschikt als baseline en TSE bestaan)."
        ),
    )
    parser.add_argument(
        "--prefer",
        choices=["last", "highest_step"],
        default="highest_step",
        help="welke csv pakken als er meerdere zijn: 'last' = checkpoint_last, 'highest_step' = grootste step.",
    )
    parser.add_argument("--input-csv", default="", help="Direct pad naar 1 csv. Overschrijft pattern.")
    parser.add_argument("--out-png", default="results/knn_per_machine.png")
    args = parser.parse_args()

    if args.input_csv:
        csv_path = args.input_csv
    else:
        csv_path = _pick_csv(args.pattern, args.prefer, args.variant)
        if not csv_path:
            print(
                f"Geen csv voor variant={args.variant!r} met pattern {args.pattern!r}; sla over.",
                file=sys.stderr,
            )
            sys.exit(0)

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
    print(f"Opgeslagen: {args.out_png}")

    means = df[metrics].mean(numeric_only=True)
    print("\nGemiddeldes:")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
