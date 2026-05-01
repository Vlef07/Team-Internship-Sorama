"""Plot EAT-treinloss (CSV). Hoort bij STEP 4/4 in run_wang2025_snellius.slurm."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-csv", default="results/training_loss.csv")
    parser.add_argument("--out-png", default="results/training_loss.png")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Aantal logpunten voor moving average. 1 = geen smoothing.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.loss_csv):
        raise FileNotFoundError(f"Loss csv niet gevonden: {args.loss_csv}")

    df = pd.read_csv(args.loss_csv)
    if df.empty:
        raise ValueError(f"Loss csv is leeg: {args.loss_csv}")

    df = df.sort_values("step").drop_duplicates(subset="step", keep="last")

    steps = df["step"].to_numpy()
    losses = df["loss"].to_numpy(dtype=float)
    lrs = df["lr"].to_numpy(dtype=float) if "lr" in df.columns else None

    smoothed = _moving_average(losses, args.smooth_window)
    smoothed_steps = steps[len(steps) - len(smoothed):]

    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if lrs is not None:
        fig, (ax_loss, ax_lr) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_loss = plt.subplots(figsize=(9, 4.5))
        ax_lr = None

    ax_loss.plot(steps, losses, alpha=0.35, color="tab:blue", label="loss (raw)")
    if args.smooth_window > 1 and len(smoothed) > 0:
        ax_loss.plot(
            smoothed_steps,
            smoothed,
            color="tab:blue",
            linewidth=2,
            label=f"loss (MA window={args.smooth_window})",
        )
    ax_loss.set_ylabel("ArcFace loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")

    if ax_lr is not None:
        ax_lr.plot(steps, lrs, color="tab:orange", linewidth=1.5)
        ax_lr.set_ylabel("learning rate")
        ax_lr.set_xlabel("step")
        ax_lr.grid(True, alpha=0.3)
    else:
        ax_loss.set_xlabel("step")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)

    print(f"Saved: {args.out_png}")
    print(f"Final loss: {losses[-1]:.4f} at step {steps[-1]}")
    if len(smoothed) > 0:
        print(f"Final smoothed loss: {smoothed[-1]:.4f}")


if __name__ == "__main__":
    main()
