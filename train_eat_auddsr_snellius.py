import argparse
import csv
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from transformers import AutoModel
import transformers.modeling_utils as _hf_modeling_utils
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


def _patch_hf_eat_tied_weights():
    pt = _hf_modeling_utils.PreTrainedModel
    if not getattr(_hf_modeling_utils, "_eat_tied_weights_compat", False) and hasattr(
        pt, "_adjust_tied_keys_with_tied_pointers"
    ):
        adjust_orig = pt._adjust_tied_keys_with_tied_pointers

        def _adjust_tied_keys_eat_compat(self, missing_keys):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return adjust_orig(self, missing_keys)

        pt._adjust_tied_keys_with_tied_pointers = _adjust_tied_keys_eat_compat
        _hf_modeling_utils._eat_tied_weights_compat = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EATBackbone(nn.Module):
    """
    EAT model wrapper following HuggingFace usage.
    Loads the pretrained model and optionally applies LoRA adapters.
    """

    def __init__(
        self,
        embed_dim=768,
        model_id="worstchan/EAT-base_epoch30_pretrain",
        apply_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_id = model_id
        self.apply_lora = apply_lora

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=False,
        ).eval()

        # Freeze the backbone; LoRA adapters remain trainable
        for p in self.model.parameters():
            p.requires_grad = False

        if apply_lora:
            self._apply_lora_to_model(
                r=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable}/{total}")

    def _apply_lora_to_model(self, target_keywords=("qkv", "proj"), r=8, alpha=16, dropout=0.05):
        """Apply LoRA adapters to selected linear layers."""

        class LoRALinear(nn.Module):
            def __init__(self, base_layer, r=8, alpha=16, dropout=0.05):
                super().__init__()
                if not isinstance(base_layer, nn.Linear):
                    raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")

                self.base_layer = base_layer
                self.base_layer.weight.requires_grad = False
                if self.base_layer.bias is not None:
                    self.base_layer.bias.requires_grad = False

                self.r = r
                self.alpha = alpha
                self.scaling = alpha / r
                self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
                self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

            def forward(self, x):
                return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        for module_name, module in list(self.model.named_modules()):
            for child_name, child in list(module.named_children()):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                if isinstance(child, nn.Linear) and any(k in full_name for k in target_keywords):
                    setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    def forward(self, x):
        """
        Input: (B, T, F) or (B, 1, T, F)
        Output: (B, T, D)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.apply_lora:
            feat = self.model.extract_features(x)
        else:
            with torch.no_grad():
                feat = self.model.extract_features(x)

        return feat


def band_attenuation(mel, prob=0.5, min_band=4, max_band=24, atten_min=0.1, atten_max=0.4):
    """
    Apply frequency-band attenuation on mel spectrograms.
    mel: (B, T, F)
    """
    if mel.dim() != 3:
        raise ValueError("mel must be (B, T, F)")

    mel_aug = mel.clone()
    bsz, _, n_bins = mel_aug.shape
    max_band = min(max_band, n_bins)
    min_band = min(min_band, max_band)

    for b in range(bsz):
        if random.random() > prob:
            continue
        band_width = random.randint(min_band, max_band)
        start = random.randint(0, n_bins - band_width)
        atten = random.uniform(atten_min, atten_max)
        mel_aug[b, :, start : start + band_width] *= atten

    return mel_aug


def simulate_anomaly(embeddings, anomaly_ratio=0.2):
    """
    Simulate synthetic anomalies by replacing random frames with noise.
    embeddings: (B, T, D)
    Returns: (B, T, D) with some frames replaced
    """
    bsz, t_len, d_dim = embeddings.shape
    embeddings_aug = embeddings.clone()

    for b in range(bsz):
        num_anom = max(1, int(t_len * anomaly_ratio))
        idx = torch.randperm(t_len, device=embeddings.device)[:num_anom]
        noise = torch.randn(num_anom, d_dim, device=embeddings.device)
        embeddings_aug[b, idx] = noise

    return embeddings_aug


class ProjectionHead(nn.Module):
    """Simple 2-layer projection head for contrastive learning"""

    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def contrastive_loss(z1, z2, temperature=0.05):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    z1, z2: (B, D) normalized embeddings
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)


def _resolve_machine_root(data_root, machine):
    direct = data_root / machine
    if direct.exists():
        return direct
    alt = data_root / f"dev_{machine}" / machine
    if alt.exists():
        return alt
    return None


def _detect_machines(data_root):
    machines = set()
    if not data_root.exists():
        return []
    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("dev_"):
            for sub in entry.iterdir():
                if sub.is_dir():
                    machines.add(sub.name)
        else:
            machines.add(entry.name)
    return sorted(machines)


def _collect_split(data_root, machines, split):
    files = []
    missing = []
    for machine in machines:
        root = _resolve_machine_root(data_root, machine)
        if root is None:
            missing.append(machine)
            continue
        split_dir = root / split
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*.wav")))
    if missing:
        print(f"Warning: missing machines under data_root: {missing}")
    return files


def _split_train_val(files, val_split, seed):
    if val_split <= 0:
        return files, []
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * val_split)
    if val_size < 1:
        return shuffled, []
    val_files = shuffled[:val_size]
    train_files = shuffled[val_size:]
    return train_files, val_files


def _load_wav(path, sample_rate):
    """Load WAV file using scipy to avoid torchcodec dependencies."""
    sr, data = wavfile.read(str(path))

    # Convert to float and normalize to [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(max_int)
    else:
        data = data.astype(np.float32)

    # Handle stereo -> mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Convert to tensor
    waveform = torch.from_numpy(data).unsqueeze(0)

    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Remove DC offset
    waveform = waveform - waveform.mean()
    return waveform


class WavToMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wav_paths,
        target_length=1024,
        sample_rate=16000,
        num_mel_bins=128,
        norm_mean=-4.268,
        norm_std=4.569,
    ):
        self.wav_paths = wav_paths
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        waveform = _load_wav(str(self.wav_paths[idx]), self.sample_rate)

        mel = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )

        n_frames = mel.shape[0]
        if n_frames < self.target_length:
            pad_amount = self.target_length - n_frames
            mel = F.pad(mel, (0, 0, 0, pad_amount), "constant", 0)
        else:
            mel = mel[: self.target_length, :]

        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        return mel


def _get_trainable_state_dict(module):
    return {name: p.detach().cpu() for name, p in module.named_parameters() if p.requires_grad}


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(path, model, proj, optimizer, epoch, best_loss, stale_epochs, args):
    payload = {
        "model_state_dict": _get_trainable_state_dict(model),
        "proj_state_dict": proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "stale_epochs": stale_epochs,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path, model, proj, optimizer, device):
    if not path or not os.path.isfile(path):
        return 0, float("inf"), 0

    ckpt = torch.load(path, map_location=device)

    model_state = ckpt.get("model_state_dict")
    if model_state:
        model.load_state_dict(model_state, strict=False)

    proj_state = ckpt.get("proj_state_dict")
    if proj_state:
        proj.load_state_dict(proj_state)

    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state:
        optimizer.load_state_dict(opt_state)
        _move_optimizer_state_to_device(optimizer, device)

    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_loss", float("inf"))), int(
        ckpt.get("stale_epochs", 0)
    )


def knn_score(query, memory_bank, k=3):
    """
    Compute KNN anomaly score.
    query: (B, D) query embeddings
    memory_bank: (M, D) normal training embeddings
    Returns: (B,) anomaly scores (0=normal, 1=anomaly)
    """
    query = F.normalize(query, dim=-1)
    memory = F.normalize(memory_bank.to(query.device), dim=-1)

    # Compute similarity
    sim = torch.matmul(query, memory.T)  # (B, M)

    # Get top-k similarities
    topk_sim, _ = torch.topk(sim, k=min(k, sim.shape[1]), dim=1)
    topk_sim = topk_sim.mean(dim=1)  # Average of top-k

    # Anomaly score: 1 - similarity (lower similarity = higher anomaly score)
    score = 1 - topk_sim

    return score


def evaluate_model(model, test_dataset, test_loader, memory_bank, device, data_root, machines):
    """
    Evaluate model on test set and compute metrics.
    """
    model.eval()

    # Compute test scores
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)
            emb = model(x)  # (B, T, D)
            pooled = emb.mean(dim=1)  # (B, D)
            s = knn_score(pooled, memory_bank, k=3)
            scores.append(s.cpu())

    scores = torch.cat(scores).numpy()

    # Parse filenames and compute metrics
    paths = [str(p) for p in test_dataset.wav_paths]
    machine_list = []
    y_true = []
    y_score_keep = []

    for i, p in enumerate(paths):
        name = Path(p).name.lower()
        if "source" not in name and "target" not in name:
            continue

        machine_list.append(Path(p).parents[1].name)
        y_true.append(1 if ("anomaly" in name or "abnormal" in name) else 0)
        y_score_keep.append(scores[i])

    if len(y_true) == 0:
        print("Warning: No source/target test files found for evaluation.")
        return

    y_true = np.asarray(y_true)
    y_score_keep = np.asarray(y_score_keep)

    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class in test data, cannot compute AUC.")
        return

    # Global metrics
    auc_global = roc_auc_score(y_true, y_score_keep)
    pauc_global = roc_auc_score(y_true, y_score_keep, max_fpr=0.1)

    prec, rec, thr = precision_recall_curve(y_true, y_score_keep)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1))
    best_threshold = thr[best_idx] if len(thr) > 0 else 0.0
    recall_best = rec[:-1][best_idx]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Global AUC:  {auc_global:.4f}")
    print(f"Global pAUC: {pauc_global:.4f}")
    print(f"Recall@bestF1: {recall_best:.4f} (threshold={best_threshold:.4f})")
    print(f"Score range: [{y_score_keep.min():.4f}, {y_score_keep.max():.4f}]")

    # Per-machine metrics
    print("\nPer-machine metrics (global best-F1 threshold):")
    print("machine | n_norm | n_anom | AUC | pAUC | TPR | TNR | mean_norm | mean_anom")
    for m in sorted(set(machine_list)):
        idx = [i for i, mm in enumerate(machine_list) if mm == m]
        y_m = y_true[idx]
        s_m = y_score_keep[idx]

        n_norm = int(np.sum(y_m == 0))
        n_anom = int(np.sum(y_m == 1))

        if n_norm > 0 and n_anom > 0:
            auc_m = roc_auc_score(y_m, s_m)
            pauc_m = roc_auc_score(y_m, s_m, max_fpr=0.1)
        else:
            auc_m = float("nan")
            pauc_m = float("nan")

        y_pred = (s_m >= best_threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_m == 1)))
        tn = int(np.sum((y_pred == 0) & (y_m == 0)))
        fn = int(np.sum((y_pred == 0) & (y_m == 1)))
        fp = int(np.sum((y_pred == 1) & (y_m == 0)))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        mean_norm = float(np.mean(s_m[y_m == 0])) if n_norm > 0 else float("nan")
        mean_anom = float(np.mean(s_m[y_m == 1])) if n_anom > 0 else float("nan")

        print(
            f"{m:10s} | {n_norm:5d} | {n_anom:6d} | {auc_m:0.4f} | {pauc_m:0.4f} | "
            f"{tpr:0.4f} | {tnr:0.4f} | {mean_norm:0.6f} | {mean_anom:0.6f}"
        )


def plot_loss_curves(epoch_losses, val_losses, save_path):
    """
    Plot training and validation loss curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses, marker="o", label="Train Loss", linewidth=2)

    if val_losses:
        ax.plot(epochs, val_losses, marker="s", label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to: {save_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EAT-based AudDSR (contrastive) with early stopping"
    )
    parser.add_argument("--data-root", default="data/dcase2025t2/dev_data/raw")
    parser.add_argument("--save-dir", default="checkpoints/eat_auddsr")
    parser.add_argument("--model-id", default="worstchan/EAT-base_epoch30_pretrain")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--num-mel-bins", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-max-files", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--mask-prob", type=float, default=0.7)
    parser.add_argument("--min-band", type=int, default=4)
    parser.add_argument("--max-band", type=int, default=36)
    parser.add_argument("--atten-min", type=float, default=0.05)
    parser.add_argument("--atten-max", type=float, default=0.6)
    parser.add_argument("--no-band-mask", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", default="")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--loss-csv",
        default="",
        help="Path to CSV file for epoch metrics (empty disables).",
    )
    parser.add_argument("--no-save-best", action="store_true")
    parser.add_argument("--no-save-last", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--machines", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data-root does not exist: {data_root}")

    machines = args.machines or _detect_machines(data_root)
    if not machines:
        raise ValueError("No machines detected. Pass --machines or check data-root.")

    train_files = _collect_split(data_root, machines, "train")
    if not train_files:
        raise FileNotFoundError("No training WAVs found. Check data-root and machine folders.")

    if args.train_max_files and args.train_max_files > 0:
        rng = random.Random(args.seed)
        rng.shuffle(train_files)
        train_files = train_files[: args.train_max_files]

    train_files, val_files = _split_train_val(train_files, args.val_split, args.seed)
    if args.val_max_files and args.val_max_files > 0:
        val_files = val_files[: args.val_max_files]

    if not train_files:
        raise ValueError("Training split is empty. Reduce --val-split or check data-root.")

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _patch_hf_eat_tied_weights()

    model = EATBackbone(
        model_id=args.model_id,
        apply_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    proj = ProjectionHead(dim=model.embed_dim).to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(proj.parameters()),
        lr=args.lr,
    )

    train_dataset = WavToMelDataset(
        train_files,
        target_length=args.target_length,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
    )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    val_loader = None
    if val_files:
        val_dataset = WavToMelDataset(
            val_files,
            target_length=args.target_length,
            sample_rate=args.sample_rate,
            num_mel_bins=args.num_mel_bins,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    start_epoch, best_loss, stale_epochs = load_checkpoint(
        args.resume, model, proj, optimizer, device
    )
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch} (best_loss={best_loss:.6f})")

    csv_file = None
    csv_writer = None
    if args.loss_csv:
        os.makedirs(os.path.dirname(args.loss_csv) or ".", exist_ok=True)
        write_header = not os.path.isfile(args.loss_csv) or start_epoch == 0
        csv_file = open(args.loss_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "best_loss"])
            csv_file.flush()

    use_band_mask = not args.no_band_mask

    print(
        "Training with settings:\n"
        f"  data_root={data_root}\n"
        f"  machines={machines}\n"
        f"  train_files={len(train_files)}\n"
        f"  val_files={len(val_files)}\n"
        f"  batch_size={args.batch_size}\n"
        f"  target_length={args.target_length}\n"
        f"  use_band_mask={use_band_mask}\n"
        f"  apply_lora={not args.no_lora}\n"
    )

    # Track losses for plotting
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.eval()
        proj.train()

        total_loss = 0.0
        batch_count = 0
        running_loss = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            x = batch.to(device, non_blocking=True)

            emb = model(x)
            if use_band_mask:
                x_anom = band_attenuation(
                    x,
                    prob=args.mask_prob,
                    min_band=args.min_band,
                    max_band=args.max_band,
                    atten_min=args.atten_min,
                    atten_max=args.atten_max,
                )
                emb_anom = model(x_anom)
            else:
                emb_anom = simulate_anomaly(emb)

            z1 = proj(emb.mean(dim=1))
            z2 = proj(emb_anom.mean(dim=1))

            loss = contrastive_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()
            batch_count += 1

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"Epoch {epoch:02d} | Step {batch_idx:05d} | Loss {avg:.4f}")
                running_loss = 0.0

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        val_loss = None
        if val_loader is not None:
            model.eval()
            proj.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    x_val = val_batch.to(device, non_blocking=True)
                    emb_val = model(x_val)
                    if use_band_mask:
                        x_anom_val = band_attenuation(
                            x_val,
                            prob=args.mask_prob,
                            min_band=args.min_band,
                            max_band=args.max_band,
                            atten_min=args.atten_min,
                            atten_max=args.atten_max,
                        )
                        emb_anom_val = model(x_anom_val)
                    else:
                        emb_anom_val = simulate_anomaly(emb_val)

                    z1_val = proj(emb_val.mean(dim=1))
                    z2_val = proj(emb_anom_val.mean(dim=1))
                    loss_val = contrastive_loss(z1_val, z2_val, temperature=args.temperature)
                    val_total += loss_val.item()
                    val_batches += 1

            val_loss = val_total / val_batches if val_batches > 0 else 0.0
            print(
                f"Epoch {epoch:02d} complete | Train {avg_loss:.6f} | Val {val_loss:.6f}"
            )
        else:
            print(f"Epoch {epoch:02d} complete | Train {avg_loss:.6f}")

        # Track losses for plotting
        epoch_train_losses.append(avg_loss)
        if val_loss is not None:
            epoch_val_losses.append(val_loss)

        if csv_writer is not None:
            csv_writer.writerow([epoch, f"{avg_loss:.6f}", f"{val_loss:.6f}" if val_loss is not None else "", f"{best_loss:.6f}"])
            csv_file.flush()

        monitor_loss = val_loss if val_loss is not None else avg_loss

        if best_loss - monitor_loss > args.min_delta:
            best_loss = monitor_loss
            stale_epochs = 0
            if not args.no_save_best:
                save_checkpoint(
                    os.path.join(args.save_dir, "checkpoint_best.pt"),
                    model,
                    proj,
                    optimizer,
                    epoch,
                    best_loss,
                    stale_epochs,
                    args,
                )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} (best_loss={best_loss:.6f})."
                )
                break

        if args.save_every and args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt"),
                model,
                proj,
                optimizer,
                epoch,
                best_loss,
                stale_epochs,
                args,
            )

    if not args.no_save_last:
        save_checkpoint(
            os.path.join(args.save_dir, "checkpoint_last.pt"),
            model,
            proj,
            optimizer,
            epoch,
            best_loss,
            stale_epochs,
            args,
        )

    if csv_file is not None:
        csv_file.close()

    print("Training complete")

    # Plot loss curves
    if epoch_train_losses:
        loss_plot_path = os.path.join(args.save_dir, "loss_curves.png")
        plot_loss_curves(epoch_train_losses, epoch_val_losses, loss_plot_path)

    # Run evaluation if not disabled
    if not args.no_eval:
        # Load best checkpoint for evaluation
        best_ckpt_path = os.path.join(args.save_dir, "checkpoint_best.pt")
        if os.path.isfile(best_ckpt_path):
            print(f"\nLoading best checkpoint: {best_ckpt_path}")
            load_checkpoint(best_ckpt_path, model, proj, optimizer, device)

        # Load test data
        test_files = _collect_split(data_root, machines, "test")
        if not test_files:
            print("Warning: No test WAV files found. Skipping evaluation.")
        else:
            print(f"Found {len(test_files)} test files. Running evaluation...")

            test_dataset = WavToMelDataset(
                test_files,
                target_length=args.target_length,
                sample_rate=args.sample_rate,
                num_mel_bins=args.num_mel_bins,
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
            )

            # Build memory bank from training data
            print("Building memory bank from training embeddings...")
            memory_bank = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch.to(device, non_blocking=True)
                    emb = model(x)  # (B, T, D)
                    pooled = emb.mean(dim=1)  # (B, D)
                    memory_bank.append(pooled.cpu())

            memory_bank = torch.cat(memory_bank, dim=0)  # (M, D)
            print(f"Memory bank shape: {memory_bank.shape}")

            # Evaluate
            evaluate_model(model, test_dataset, test_loader, memory_bank, device, data_root, machines)
    else:
        print("Evaluation skipped (--no-eval flag set).")


if __name__ == "__main__":
    main()
