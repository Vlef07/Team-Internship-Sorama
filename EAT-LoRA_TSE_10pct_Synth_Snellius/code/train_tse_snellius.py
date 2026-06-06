"""
this script trains one tse model per machine type on the additional dataset
tse means target signal enhancement which tries to reduce background noise on a waveform
step one find all machine folders under data root
step two for each machine collect normal train wavs as clean target audio
step three mix each clean clip with noise from other machines at random snr
step four run the small tse network to predict a cleaner waveform
step five save one checkpoint file per machine for use during knn evaluation
tse is used at test time on waveforms not during eat lora training
"""

import argparse
import csv
import os
import random
from typing import List, Optional

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
    extra_noise_roots: Optional[List[str]] = None,
):
    # step 1 collect normal train wav paths for this machine
    target_wavs: List[str] = list_target_wavs(base_path, machine)
    if len(target_wavs) == 0:
        print(f"[{machine}] geen target wavs gevonden, sla machine over")
        return

    dataset = TSEWaveformDataset(
        wav_paths=target_wavs,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
    )
    # step 2 load short clean audio segments in batches
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    noise_extra = [p for p in (extra_noise_roots or []) if os.path.normpath(p) != os.path.normpath(base_path)]
    # step 3 build a pool of noise clips from other machines
    noise_bank = CrossMachineNoiseBank(
        base_path=base_path,
        target_machine=machine,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
        extra_noise_roots=noise_extra,
    )

    model = TSEMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # step 4 train loop mix clean signal with noise then predict cleaner output
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

    # step 5 save one checkpoint file per machine
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
    print(f"[{machine}] TSE checkpoint opgeslagen in {save_path}")


def _endless_batches(loader: DataLoader):
    """Oneindige herhaling van batches uit een DataLoader voor mix training."""
    while True:
        for batch in loader:
            yield batch


def train_one_machine_mixed(
    machine: str,
    base_path_primary: str,
    base_path_secondary: str,
    save_path: str,
    num_steps: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    snr_db_range,
    device: torch.device,
    csv_writer,
    log_interval: int,
    mix_primary_fraction: float,
):
    """Zelfde TSE loss als train_one_machine maar elke stap willekeurig primary of secondary boom.

    mix_primary_fraction is de kans op primary (data-root). Standaard 0.1 is ongeveer
    10 procent gesynthetiseerde train en 90 procent ruwe DCASE train als primary synth is.
    """
    target_primary = list_target_wavs(base_path_primary, machine)
    target_secondary = list_target_wavs(base_path_secondary, machine)
    # step 1 if one data root is empty fall back to single root training
    if len(target_primary) == 0 and len(target_secondary) == 0:
        print(f"[{machine}] geen wavs op beide roots, sla over")
        return
    if len(target_primary) == 0:
        print(f"[{machine}] alleen secondary boom heeft wavs voor TSE mix, gebruik alleen secondary")
        train_one_machine(
            machine=machine,
            base_path=base_path_secondary,
            save_path=save_path,
            num_steps=num_steps,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            snr_db_range=snr_db_range,
            device=device,
            csv_writer=csv_writer,
            log_interval=log_interval,
            extra_noise_roots=[base_path_primary],
        )
        return
    if len(target_secondary) == 0:
        print(f"[{machine}] alleen primary boom heeft wavs voor TSE mix, gebruik alleen primary")
        train_one_machine(
            machine=machine,
            base_path=base_path_primary,
            save_path=save_path,
            num_steps=num_steps,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            snr_db_range=snr_db_range,
            device=device,
            csv_writer=csv_writer,
            log_interval=log_interval,
            extra_noise_roots=[base_path_secondary],
        )
        return

    ds_p = TSEWaveformDataset(
        wav_paths=target_primary,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
    )
    ds_s = TSEWaveformDataset(
        wav_paths=target_secondary,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
    )
    loader_p = DataLoader(
        ds_p,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_s = DataLoader(
        ds_s,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Cross-machine noise pool uses raw data first and synth only as extra fallback
    noise_extra = []
    if os.path.normpath(base_path_primary) != os.path.normpath(base_path_secondary):
        noise_extra.append(base_path_primary)
    bank_p = CrossMachineNoiseBank(
        base_path=base_path_secondary,
        target_machine=machine,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
        extra_noise_roots=noise_extra,
    )
    bank_s = CrossMachineNoiseBank(
        base_path=base_path_secondary,
        target_machine=machine,
        target_sr=TSE_SAMPLE_RATE,
        segment_seconds=TSE_SEGMENT_SECONDS,
        extra_noise_roots=noise_extra,
    )

    it_p = _endless_batches(loader_p)
    it_s = _endless_batches(loader_s)

    model = TSEMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # step 4 each step randomly picks synth or raw clean audio then trains the mask net
    pbar = tqdm(total=num_steps, desc=f"TSE-mix[{machine}]")
    step = 0
    running = 0.0

    while step < num_steps:
        model.train()
        if random.random() < mix_primary_fraction:
            clean = next(it_p).to(device, non_blocking=True)
            noise_bank = bank_p
        else:
            clean = next(it_s).to(device, non_blocking=True)
            noise_bank = bank_s

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

    # step 5 save one checkpoint file per machine after mixed training
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "machine": machine,
            "num_steps": step,
            "snr_db_range": list(snr_db_range),
            "mix_second_root": base_path_secondary,
            "mix_primary_fraction": float(mix_primary_fraction),
        },
        save_path,
    )
    print(f"[{machine}] TSE mix checkpoint opgeslagen in {save_path}")


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
    parser.add_argument(
        "--mix-second-data-root",
        default="",
        help=(
            "Tweede data boom voor TSE. "
            "Per update wordt willekeurig gekozen of de patch uit de eerste of tweede boom komt. "
            "De verhouding stel je met mix-primary-fraction."
        ),
    )
    parser.add_argument(
        "--mix-primary-fraction",
        type=float,
        default=0.1,
        help=(
            "Alleen bij mix-second-data-root. Kans dat een stap primary (data-root) gebruikt. "
            "Default 0.1 is ongeveer 10 procent synth en 90 procent tweede boom als primary synth is."
        ),
    )
    args = parser.parse_args()

    mpf = float(args.mix_primary_fraction)
    if not (0.0 <= mpf <= 1.0):
        raise ValueError("mix-primary-fraction moet tussen 0 en 1 liggen")

    base_path = os.path.normpath(args.data_root)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"data-root bestaat niet: {base_path}")

    mix_second = (args.mix_second_data_root or "").strip()
    if mix_second:
        mix_second = os.path.normpath(mix_second)
        if not os.path.isdir(mix_second):
            raise FileNotFoundError(f"mix-second-data-root bestaat niet: {mix_second}")
        print(
            "TSE mix training. Primary root is",
            base_path,
            "secondary root is",
            mix_second,
            "mix_primary_fraction",
            mpf,
        )

    os.makedirs(args.save_dir, exist_ok=True)

    # step 1 pick gpu if available else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

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

    # step 2 train one tse model per machine and write optional loss csv
    for machine in machines:
        save_path = os.path.join(args.save_dir, f"{machine}.pt")
        if args.skip_existing and os.path.isfile(save_path):
            print(f"[{machine}] checkpoint bestaat al, sla over ({save_path})")
            continue
        print(f"\nTSE trainen voor {machine}")
        try:
            if mix_second:
                train_one_machine_mixed(
                    machine=machine,
                    base_path_primary=base_path,
                    base_path_secondary=mix_second,
                    save_path=save_path,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    num_workers=args.num_workers,
                    snr_db_range=snr_db_range,
                    device=device,
                    csv_writer=csv_writer,
                    log_interval=args.log_interval,
                    mix_primary_fraction=mpf,
                )
            else:
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
