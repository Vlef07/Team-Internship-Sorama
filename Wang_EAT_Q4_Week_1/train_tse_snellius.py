"""Train one TSE model per machine type.

Dit is STEP 1/4 in run_wang2025_snellius.slurm (in het notebook: stappen 17 t/m 19).

Based on Fujimura et al. (2025), Section 2.1: a per machine TSE model that
learns to extract the target machine signal from a noisy mixture. We train
with a negative SNR reconstruction loss on waveforms (same L_D spirit as the paper).

Usage on Snellius:
    python -u train_tse_snellius.py \
        --data-root "$SCRATCH_DATA_ROOT" \
        --save-dir "$PROJECT_DIR/checkpoints/tse" \
        --num-steps 3000 \
        --loss-csv "$PROJECT_DIR/results/tse_loss.csv"
"""

import argparse
import csv
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tse_snellius import (
    CrossMachineNoiseBank,
    TSEMaskNet,
    TSEWaveformDataset,
    TSE_SAMPLE_RATE,
    TSE_SEGMENT_SECONDS,
    list_machines,
    list_target_wavs,
    mix_at_random_snr,
    neg_snr_loss,
)


def train_one_machine(
    machine: str,
    base_path: str,
    save_path: str,
    num_steps: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    snr_db_range,
    device: torch.device,
    csv_writer,
    log_interval: int,
):
    target_wavs: List[str] = list_target_wavs(base_path, machine)
    if len(target_wavs) == 0:
        print(f"[{machine}] geen target wavs gevonden, sla machine over")
        return

    dataset = TSEWaveformDataset(
        wav_paths=target_wavs,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    noise_bank = CrossMachineNoiseBank(
        base_path=base_path,
        target_machine=machine,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
    )

    model = TSEMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(total=num_steps, desc=f"TSE[{machine}]")
    step = 0
    running = 0.0

    while step < num_steps:
        model.train()
        for clean in loader:
            if step >= num_steps:
                break
            clean = clean.to(device, non_blocking=True)

            noisy_list = []
            for i in range(clean.shape[0]):
                noise = noise_bank.sample(length=clean.shape[-1]).to(device)
                noisy_list.append(mix_at_random_snr(clean[i], noise, snr_db_range))
            noisy = torch.stack(noisy_list, dim=0)

            enhanced = model(noisy)
            loss = neg_snr_loss(enhanced, clean)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running += loss.item()
            step += 1
            pbar.update(1)

            if step % log_interval == 0:
                avg = running / log_interval
                pbar.set_postfix({"neg_snr": f"{avg:.3f}"})
                if csv_writer is not None:
                    csv_writer.writerow([machine, step, f"{avg:.6f}"])
                running = 0.0

    pbar.close()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "machine": machine,
            "num_steps": step,
            "snr_db_range": list(snr_db_range),
        },
        save_path,
    )
    print(f"[{machine}] saved TSE checkpoint -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dcase2025t2/dev_data/raw")
    parser.add_argument("--save-dir", default="checkpoints/tse")
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--loss-csv",
        default="",
        help="Pad naar CSV waar (machine,step,loss) wordt weggeschreven. Leeg = geen CSV.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_path = os.path.normpath(args.data_root)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"data-root bestaat niet: {base_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    machines_all = list_machines(base_path)
    if args.machines:
        wanted = set(args.machines)
        machines = [m for m in machines_all if m in wanted]
    else:
        machines = machines_all
    if not machines:
        raise ValueError("Geen machines gevonden onder data-root")

    csv_file = None
    csv_writer = None
    if args.loss_csv:
        os.makedirs(os.path.dirname(args.loss_csv) or ".", exist_ok=True)
        write_header = not os.path.isfile(args.loss_csv)
        csv_file = open(args.loss_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["machine", "step", "loss"])
            csv_file.flush()

    snr_db_range = (args.snr_min, args.snr_max)

    for machine in machines:
        save_path = os.path.join(args.save_dir, f"{machine}.pt")
        if args.skip_existing and os.path.isfile(save_path):
            print(f"[{machine}] checkpoint bestaat al, sla over ({save_path})")
            continue
        print(f"\n===== Training TSE for {machine} =====")
        try:
            train_one_machine(
                machine=machine,
                base_path=base_path,
                save_path=save_path,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                snr_db_range=snr_db_range,
                device=device,
                csv_writer=csv_writer,
                log_interval=args.log_interval,
            )
        except Exception as ex:
            print(f"[{machine}] TSE training mislukt: {ex}")
            if csv_file is not None:
                csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    print("\nTSE training klaar. Checkpoints in:", args.save_dir)


if __name__ == "__main__":
    main()
