"""Één TSE-model per machinetype trainen. STEP 1/4 in run_wang2025_snellius.slurm
(notebook: ongeveer stappen 17 t/m 19).

We willen een klein netwerk per machine dat rommel op de opname dempt en het
machinesignaal dichter bij "schoon" probeert te brengen. Dat netwerk gebruiken we
later niet tijdens EAT-train (training fase), wél bij kNN-eval (testing fase) op de golfvorm. Fujimura
et.al. (2025) doen iets vergelijkbaars met een zwaardere architectuur en extra loss,
hier trainen we met een negatieve SNR-loss op de golfvorm (L_D).

Stap voor stap wat de code doet:

1) main zoekt alle machines onder data-root. Voor elke machine: pad naar {machine}.pt
   in save-dir, tenzij --skip-existing en het bestand al bestaat.
2) train_one_machine verzamelt train-wavs van dát machinetype als "schone" doelen
   (list_target_wavs → TSEWaveformDataset). Elke minibatch zijn korte stukken audio
   op vaste lengte en sample rate.
3) CrossMachineNoiseBank pakt ruis van andere machines (en bruikbare noise uit de set).
   Zo krijg je realistische mengsels: de machine zelf + storende geluiden die niet van
   die machine hoeven te komen.
4) Voor elk fragment in de batch, mix_at_random_snr koppelt schoon signaal en ruis met
   een willekeurige SNR tussen snr-min en snr-max. Dat maakt het model robuuster.
5) TSEMaskNet(noisy) voorspelt een opgeschoonde golfvorm. De loss is neg_snr_loss(
   enhanced, clean): hoe beter de output op het originele schone segment lijkt, hoe
   lager de loss. AdamW + gradient clip, num_steps updates per machine.
6) torch.save schrijft model_state_dict + machine-naam naar save-dir/{machine}.pt.
   Optioneel append van (machine, step, gemiddelde loss) naar loss-csv.

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
