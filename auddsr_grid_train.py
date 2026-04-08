#!/usr/bin/env python3
"""
Standalone AudDSR training script with grid search and checkpointing.

This script ports the full notebook pipeline to Python:
1) Stage 1: Discrete VQ autoencoder training on all machine types.
2) Stage 2: Freeze discrete modules, train object-specific decoder + detector.
3) Evaluate with AUROC/pAUC and anomaly-score histograms/statistics.
4) Hyperparameter grid search across multiple option sets.
5) Save checkpoints that can be reloaded and reconstructed easily.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Option presets from notebook
# -----------------------------
QUANTIZER_OPTIONS = {
    "tiny": {"q1_embeddings": 128, "q2_embeddings": 128},
    "small": {"q1_embeddings": 256, "q2_embeddings": 256},
    "base": {"q1_embeddings": 512, "q2_embeddings": 512},
    "large": {"q1_embeddings": 1024, "q2_embeddings": 1024},
    "xlarge": {"q1_embeddings": 2048, "q2_embeddings": 2048},
}

FRONTEND_OPTIONS = {
    "transient": {"n_fft": 512, "hop_length": 128, "n_mels": 128, "norm": "per_bin"},
    "balanced": {"n_fft": 1024, "hop_length": 256, "n_mels": 128, "norm": "per_bin"},
    "harmonic": {"n_fft": 2048, "hop_length": 512, "n_mels": 128, "norm": "global"},
    "low_compute": {"n_fft": 512, "hop_length": 256, "n_mels": 64, "norm": "none"},
}

CORRUPTION_OPTIONS = {
    "perlin_like": {"mode": "perlin_like", "ratio": 0.20, "block_hw": (8, 8), "low_res": (8, 8)},
    "band_time": {"mode": "band_time", "ratio": 0.20, "block_hw": (8, 8), "low_res": (8, 8)},
    "point": {"mode": "point", "ratio": 0.20, "block_hw": (4, 4), "low_res": (8, 8)},
    "time_stripe": {"mode": "time_stripe", "ratio": 0.20, "block_hw": (4, 16), "low_res": (8, 8)},
    "freq_stripe": {"mode": "freq_stripe", "ratio": 0.20, "block_hw": (16, 4), "low_res": (8, 8)},
    "block_medium": {"mode": "block", "ratio": 0.20, "block_hw": (8, 8), "low_res": (8, 8)},
}

DETECTOR_OPTIONS = {
    "light": {"width": 32},
    "wide": {"width": 64},
}


@dataclass
class TrialConfig:
    quantizer_name: str
    frontend_name: str
    corruption_name: str
    detector_name: str
    q1_embeddings: int
    q2_embeddings: int
    frontend_cfg: Dict
    corruption_cfg: Dict
    detector_cfg: Dict
    stage1_epochs: int
    stage2_epochs: int
    lr_stage1: float
    lr_stage2: float
    lambda_x: float
    lambda_k: float
    lambda_stage2_rec: float
    lambda_stage2_det: float
    focal_alpha: float
    focal_gamma: float
    stage2_anomaly_batch_ratio: float


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_wav_files(folder: str) -> List[str]:
    wav_paths: List[str] = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(".wav"):
                wav_paths.append(os.path.join(root, file_name))
    return sorted(wav_paths)


def collect_machine_split_files(data_root: str, machine_type: str, split: str) -> List[str]:
    split_dir = os.path.join(data_root, f"dev_{machine_type}", machine_type, split)
    return collect_wav_files(split_dir)


def collect_multi_machine_split_files(data_root: str, machine_types: Sequence[str], split: str) -> List[str]:
    files: List[str] = []
    for machine in machine_types:
        files.extend(collect_machine_split_files(data_root, machine, split))
    return sorted(files)


def collect_multi_machine_split_files_limited(
    data_root: str,
    machine_types: Sequence[str],
    split: str,
    max_per_machine: int,
    seed: int,
) -> List[str]:
    if max_per_machine <= 0:
        return collect_multi_machine_split_files(data_root, machine_types, split)

    files: List[str] = []
    for machine in machine_types:
        machine_files = collect_machine_split_files(data_root, machine, split)
        rng = random.Random(seed + (abs(hash((machine, split))) % 1_000_000))
        rng.shuffle(machine_files)
        files.extend(sorted(machine_files[:max_per_machine]))
    return sorted(files)


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    try:
        waveform, sample_rate = torchaudio.load(path)
    except (RuntimeError, ImportError, OSError):
        sample_rate, data = wavfile.read(path)
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)
        waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.squeeze(0)


def normalize_spec(spec: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return spec
    if mode == "global":
        mean = spec.mean()
        std = spec.std().clamp_min(1e-6)
        return (spec - mean) / std
    if mode == "per_bin":
        mean = spec.mean(dim=-1, keepdim=True)
        std = spec.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (spec - mean) / std
    raise ValueError(f"Unknown normalization mode: {mode}")


def build_mel_transform(target_sr: int, frontend_cfg: Dict) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=frontend_cfg["n_fft"],
        hop_length=frontend_cfg["hop_length"],
        n_mels=frontend_cfg["n_mels"],
    )


def waveform_to_model_input(
    waveform: torch.Tensor,
    mel_transform: T.MelSpectrogram,
    frontend_cfg: Dict,
    spec_size: int,
) -> torch.Tensor:
    spec = torch.log1p(mel_transform(waveform)).unsqueeze(0)
    spec = normalize_spec(spec, frontend_cfg["norm"])
    spec = F.interpolate(
        spec.unsqueeze(0),
        size=(spec_size, spec_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return spec


class SpectrogramFileDataset(Dataset):
    def __init__(
        self,
        file_paths: Sequence[str],
        mel_transform: T.MelSpectrogram,
        frontend_cfg: Dict,
        target_sr: int,
        spec_size: int,
    ):
        self.file_paths = list(file_paths)
        self.mel_transform = mel_transform
        self.frontend_cfg = frontend_cfg
        self.target_sr = target_sr
        self.spec_size = spec_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        waveform = load_waveform(file_path, self.target_sr)
        spec = waveform_to_model_input(waveform, self.mel_transform, self.frontend_cfg, self.spec_size)
        return spec, file_path


def make_corruption_mask(
    z_q: torch.Tensor,
    mode: str = "band_time",
    ratio: float = 0.2,
    block_hw: Tuple[int, int] = (8, 8),
    low_res: Tuple[int, int] = (8, 8),
    anomaly_batch_ratio: float = 1.0,
) -> torch.Tensor:
    bsz, _, h, w = z_q.shape
    device = z_q.device
    mask = torch.zeros((bsz, 1, h, w), device=device, dtype=torch.bool)

    if anomaly_batch_ratio <= 0:
        return mask

    n_anom = int(round(anomaly_batch_ratio * bsz))
    if anomaly_batch_ratio > 0 and n_anom == 0:
        n_anom = 1
    n_anom = max(0, min(bsz, n_anom))

    active = torch.zeros((bsz,), device=device, dtype=torch.bool)
    if n_anom > 0:
        active_idx = torch.randperm(bsz, device=device)[:n_anom]
        active[active_idx] = True

    for b in range(bsz):
        if not bool(active[b].item()):
            continue

        if mode == "point":
            mask[b] = torch.rand((1, h, w), device=device) < ratio
            continue

        if mode == "time_stripe":
            stripe_w = max(1, int(w * ratio))
            start_w = int(torch.randint(0, max(1, w - stripe_w + 1), (1,), device=device).item())
            mask[b, :, :, start_w : start_w + stripe_w] = True
            continue

        if mode == "freq_stripe":
            stripe_h = max(1, int(h * ratio))
            start_h = int(torch.randint(0, max(1, h - stripe_h + 1), (1,), device=device).item())
            mask[b, :, start_h : start_h + stripe_h, :] = True
            continue

        if mode == "block":
            block_h, block_w = block_hw
            target_area = max(1, int(ratio * h * w))
            block_area = max(1, block_h * block_w)
            num_blocks = max(1, target_area // block_area)

            for _ in range(num_blocks):
                top = int(torch.randint(0, max(1, h - block_h + 1), (1,), device=device).item())
                left = int(torch.randint(0, max(1, w - block_w + 1), (1,), device=device).item())
                mask[b, :, top : top + block_h, left : left + block_w] = True
            continue

        if mode == "perlin_like":
            low_h = max(2, min(h, int(low_res[0])))
            low_w = max(2, min(w, int(low_res[1])))
            coarse = torch.rand((1, 1, low_h, low_w), device=device)
            smooth = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)[0, 0]
            threshold = torch.quantile(smooth.flatten(), 1.0 - ratio)
            mask[b, 0] = smooth >= threshold
            continue

        if mode == "band_time":
            band_h_min = max(1, int(0.10 * h))
            band_h_max = max(band_h_min, int(0.50 * h))
            band_h = int(torch.randint(band_h_min, band_h_max + 1, (1,), device=device).item())
            top = int(torch.randint(0, max(1, h - band_h + 1), (1,), device=device).item())

            seg_min = max(1, int(0.05 * w))
            seg_max = max(seg_min, int(0.25 * w))
            num_segments = int(torch.randint(2, 6, (1,), device=device).item())
            for _ in range(num_segments):
                seg_w = int(torch.randint(seg_min, seg_max + 1, (1,), device=device).item())
                left = int(torch.randint(0, max(1, w - seg_w + 1), (1,), device=device).item())
                mask[b, :, top : top + band_h, left : left + seg_w] = True
            continue

        raise ValueError(f"Unknown corruption mode: {mode}")

    return mask


# -----------------------------
# Models
# -----------------------------
class FeatureEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, ch1: int = 64, ch2: int = 128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        return f1, f2


class VectorQuantizerSimple(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, ch, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, ch)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(bsz, h, w, ch).permute(0, 3, 1, 2)
        return z_q, indices


class GeneralDecoderModule1(nn.Module):
    def __init__(self, ch1: int = 64, ch2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch1, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, q1: torch.Tensor) -> torch.Tensor:
        return self.net(q1)


class QuantizedDecoderModule2(nn.Module):
    def __init__(self, ch1: int = 64, ch2: int = 128):
        super().__init__()
        self.q1_proj = nn.Conv2d(ch2, ch1, kernel_size=1)
        self.to_img = nn.Sequential(
            nn.Conv2d(ch1 * 2, ch1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        q1_up = F.interpolate(q1, size=q2.shape[-2:], mode="bilinear", align_corners=False)
        q1_up = self.q1_proj(q1_up)
        fused = torch.cat([q2, q1_up], dim=1)
        return self.to_img(fused)


class DiscreteAutoencoderPaper(nn.Module):
    def __init__(self, in_ch: int = 1, ch1: int = 64, ch2: int = 128, q1_embeddings: int = 512, q2_embeddings: int = 512):
        super().__init__()
        self.ch1 = ch1
        self.ch2 = ch2

        self.encoder = FeatureEncoder(in_ch=in_ch, ch1=ch1, ch2=ch2)
        self.vq1 = VectorQuantizerSimple(num_embeddings=q1_embeddings, embedding_dim=ch2)
        self.general_dec1 = GeneralDecoderModule1(ch1=ch1, ch2=ch2)

        self.pre_q2_proj = nn.Conv2d(ch1 * 2, ch1, kernel_size=1)
        self.vq2 = VectorQuantizerSimple(num_embeddings=q2_embeddings, embedding_dim=ch1)
        self.general_dec2 = QuantizedDecoderModule2(ch1=ch1, ch2=ch2)

    def encode_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def quantize_features(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1, _ = self.vq1(f2)
        f_u = self.general_dec1(q1)
        pre_q2 = self.pre_q2_proj(torch.cat([f1, f_u], dim=1))
        q2, _ = self.vq2(pre_q2)
        return q1, q2, pre_q2

    def decode_general(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return self.general_dec2(q1, q2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1, f2 = self.encode_features(x)
        q1, q2, pre_q2 = self.quantize_features(f1, f2)
        x_out = self.decode_general(q1, q2)
        return {
            "x_out": x_out,
            "f1": f1,
            "f2": f2,
            "pre_q2": pre_q2,
            "q1": q1,
            "q2": q2,
        }


class PaperAnomalyDetector(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, x_general: torch.Tensor, x_specific: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_general, x_specific], dim=1)
        return torch.sigmoid(self.net(x))


class AudDSRPaperLike(nn.Module):
    def __init__(
        self,
        ch1: int = 64,
        ch2: int = 128,
        q1_embeddings: int = 512,
        q2_embeddings: int = 512,
        corruption_cfg: Optional[Dict] = None,
        detector_cfg: Optional[Dict] = None,
        discrete_ae: Optional[DiscreteAutoencoderPaper] = None,
    ):
        super().__init__()

        if discrete_ae is None:
            self.discrete_ae = DiscreteAutoencoderPaper(
                in_ch=1,
                ch1=ch1,
                ch2=ch2,
                q1_embeddings=q1_embeddings,
                q2_embeddings=q2_embeddings,
            )
        else:
            self.discrete_ae = discrete_ae
            ch1 = self.discrete_ae.ch1
            ch2 = self.discrete_ae.ch2

        self.object_specific_decoder = QuantizedDecoderModule2(ch1=ch1, ch2=ch2)
        self.object_specific_decoder.load_state_dict(self.discrete_ae.general_dec2.state_dict())

        detector_cfg = detector_cfg or {"width": 32}
        self.detector = PaperAnomalyDetector(width=detector_cfg["width"])

        self.corruption_cfg = corruption_cfg or {
            "mode": "band_time",
            "ratio": 0.2,
            "block_hw": (8, 8),
            "low_res": (8, 8),
        }

    def freeze_discrete_modules(self) -> None:
        self.discrete_ae.eval()
        for p in self.discrete_ae.parameters():
            p.requires_grad = False

    def _sample_random_codes(self, quantizer: VectorQuantizerSimple, ref_tensor: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = ref_tensor.shape
        idx = torch.randint(0, quantizer.embedding.num_embeddings, (bsz, h, w), device=ref_tensor.device)
        sampled = quantizer.embedding(idx).permute(0, 3, 1, 2).contiguous()
        return sampled

    def forward(self, x: torch.Tensor, apply_anomaly: bool = False, anomaly_batch_ratio: float = 1.0) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dae_out = self.discrete_ae(x)
            q1 = dae_out["q1"]
            q2 = dae_out["q2"]
            x_general = dae_out["x_out"]

        if apply_anomaly:
            mask_q1 = make_corruption_mask(
                q1,
                mode=self.corruption_cfg["mode"],
                ratio=self.corruption_cfg["ratio"],
                block_hw=tuple(self.corruption_cfg["block_hw"]),
                low_res=tuple(self.corruption_cfg.get("low_res", (8, 8))),
                anomaly_batch_ratio=anomaly_batch_ratio,
            )
            mask_q2 = F.interpolate(mask_q1.float(), size=q2.shape[-2:], mode="nearest").bool()

            random_q1 = self._sample_random_codes(self.discrete_ae.vq1, q1)
            random_q2 = self._sample_random_codes(self.discrete_ae.vq2, q2)

            q1_aug = torch.where(mask_q1.expand_as(q1), random_q1, q1)
            q2_aug = torch.where(mask_q2.expand_as(q2), random_q2, q2)
        else:
            mask_q1 = torch.zeros((q1.shape[0], 1, q1.shape[2], q1.shape[3]), device=q1.device, dtype=torch.bool)
            mask_q2 = F.interpolate(mask_q1.float(), size=q2.shape[-2:], mode="nearest").bool()
            q1_aug = q1
            q2_aug = q2

        x_specific = self.object_specific_decoder(q1_aug, q2_aug)
        anomaly_map = self.detector(x_general, x_specific)

        return {
            "x_general": x_general,
            "x_specific": x_specific,
            "x_out": x_general,
            "x_clean": x_specific,
            "anomaly_map": anomaly_map,
            "mask": mask_q1,
            "mask_q2": mask_q2,
            "q1": q1,
            "q2": q2,
            "q1_aug": q1_aug,
            "q2_aug": q2_aug,
        }


# -----------------------------
# Loss and eval
# -----------------------------
def binary_focal_loss(prob: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(min=eps, max=1.0 - eps)
    ce = -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))
    pt = target * prob + (1.0 - target) * (1.0 - prob)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    loss = alpha_t * ((1.0 - pt) ** gamma) * ce
    return loss.mean()


def stage1_vqvae_loss(x: torch.Tensor, outputs: Dict[str, torch.Tensor], lambda_x: float = 1.0, lambda_k: float = 0.25):
    rec = F.mse_loss(outputs["x_out"], x)

    q1_codebook = F.mse_loss(outputs["f2"].detach(), outputs["q1"])
    q1_commit = F.mse_loss(outputs["f2"], outputs["q1"].detach())

    q2_codebook = F.mse_loss(outputs["pre_q2"].detach(), outputs["q2"])
    q2_commit = F.mse_loss(outputs["pre_q2"], outputs["q2"].detach())

    total = lambda_x * rec + q1_codebook + lambda_k * q1_commit + q2_codebook + lambda_k * q2_commit

    return total, {
        "total": float(total.item()),
        "rec": float(rec.item()),
        "q1_cb": float(q1_codebook.item()),
        "q1_cm": float(q1_commit.item()),
        "q2_cb": float(q2_codebook.item()),
        "q2_cm": float(q2_commit.item()),
    }


def stage2_auddsr_loss(
    x: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    lambda_rec: float = 1.0,
    lambda_det: float = 1.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    rec_specific = F.mse_loss(outputs["x_specific"], x)
    det_target = F.interpolate(outputs["mask"].float(), size=x.shape[-2:], mode="nearest")
    det_loss = binary_focal_loss(outputs["anomaly_map"], det_target, alpha=alpha, gamma=gamma)

    total = lambda_rec * rec_specific + lambda_det * det_loss
    return total, {
        "total": float(total.item()),
        "rec_specific": float(rec_specific.item()),
        "det": float(det_loss.item()),
    }


def anomaly_map_to_file_score(anomaly_map: torch.Tensor, topk_ratio: float = 0.05) -> torch.Tensor:
    flat = anomaly_map.flatten(1)
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals = torch.topk(flat, k=k, dim=1).values
    return topk_vals.mean(dim=1)


def score_file_list(
    model_obj: AudDSRPaperLike,
    file_list: Sequence[str],
    mel_transform: T.MelSpectrogram,
    frontend_cfg: Dict,
    topk_ratio: float,
    device: torch.device,
    target_sr: int,
    spec_size: int,
) -> np.ndarray:
    model_obj.eval()
    scores: List[float] = []

    with torch.no_grad():
        for file_path in file_list:
            waveform = load_waveform(file_path, target_sr)
            x = waveform_to_model_input(waveform, mel_transform, frontend_cfg, spec_size).unsqueeze(0).to(device)
            out = model_obj(x, apply_anomaly=False)
            file_score = anomaly_map_to_file_score(out["anomaly_map"], topk_ratio=topk_ratio)
            scores.append(float(file_score.item()))

    return np.asarray(scores, dtype=np.float32)


def infer_label_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path).lower()
    if "anomaly" in name:
        return 1
    if "normal" in name:
        return 0
    return None


def compute_auc_metrics(file_paths: Sequence[str], scores: np.ndarray) -> Tuple[Optional[float], Optional[float], int]:
    labels = np.asarray([infer_label_from_filename(p) for p in file_paths], dtype=object)
    valid = np.asarray([lbl is not None for lbl in labels])

    if valid.sum() == 0:
        return None, None, 0

    y_true = labels[valid].astype(int)
    y_score = scores[valid].astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return None, None, int(valid.sum())

    auroc = float(roc_auc_score(y_true, y_score))
    pauc_01 = float(roc_auc_score(y_true, y_score, max_fpr=0.1))
    return auroc, pauc_01, int(valid.sum())


# -----------------------------
# Train / save / reload
# -----------------------------
def train_trial(
    trial_cfg: TrialConfig,
    data_root: str,
    all_machine_types: Sequence[str],
    stage1_machine_types: Sequence[str],
    stage2_machine_types: Sequence[str],
    subset_train_per_machine: int,
    subset_test_per_machine: int,
    batch_size: int,
    num_workers: int,
    target_sr: int,
    spec_size: int,
    topk_ratio: float,
    device: torch.device,
    seed: int,
) -> Dict:
    mel_transform = build_mel_transform(target_sr, trial_cfg.frontend_cfg)

    stage1_train_files = collect_multi_machine_split_files_limited(
        data_root,
        stage1_machine_types,
        split="train",
        max_per_machine=subset_train_per_machine,
        seed=seed,
    )
    if not stage1_train_files:
        raise FileNotFoundError("No Stage 1 wav files found.")

    stage2_train_files = collect_multi_machine_split_files_limited(
        data_root,
        stage2_machine_types,
        split="train",
        max_per_machine=subset_train_per_machine,
        seed=seed + 17,
    )
    if not stage2_train_files:
        raise FileNotFoundError("No Stage 2 wav files found.")

    test_files = collect_multi_machine_split_files_limited(
        data_root,
        stage2_machine_types,
        split="test",
        max_per_machine=subset_test_per_machine,
        seed=seed + 33,
    )
    if not test_files:
        raise FileNotFoundError("No test wav files found.")

    stage1_ds = SpectrogramFileDataset(stage1_train_files, mel_transform, trial_cfg.frontend_cfg, target_sr, spec_size)
    stage2_ds = SpectrogramFileDataset(stage2_train_files, mel_transform, trial_cfg.frontend_cfg, target_sr, spec_size)

    stage1_loader = DataLoader(stage1_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    stage2_loader = DataLoader(stage2_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(
        f"Stage1 files={len(stage1_train_files)} | Stage2 files={len(stage2_train_files)} | Test files={len(test_files)} | "
        f"subset_train_per_machine={subset_train_per_machine} | subset_test_per_machine={subset_test_per_machine}"
    )

    # Stage 1
    discrete_ae = DiscreteAutoencoderPaper(
        in_ch=1,
        ch1=64,
        ch2=128,
        q1_embeddings=trial_cfg.q1_embeddings,
        q2_embeddings=trial_cfg.q2_embeddings,
    ).to(device)

    opt_stage1 = torch.optim.Adam(discrete_ae.parameters(), lr=trial_cfg.lr_stage1)

    stage1_history: List[Dict[str, float]] = []
    discrete_ae.train()
    for epoch in range(1, trial_cfg.stage1_epochs + 1):
        run = {"total": 0.0, "rec": 0.0, "q1_cb": 0.0, "q1_cm": 0.0, "q2_cb": 0.0, "q2_cm": 0.0}
        for batch in stage1_loader:
            x = batch[0].to(device)
            opt_stage1.zero_grad()
            outputs = discrete_ae(x)
            loss, parts = stage1_vqvae_loss(x, outputs, lambda_x=trial_cfg.lambda_x, lambda_k=trial_cfg.lambda_k)
            loss.backward()
            opt_stage1.step()
            for k in run:
                run[k] += parts[k]

        n_batches = max(1, len(stage1_loader))
        epoch_stats = {
            "epoch": float(epoch),
            "total": run["total"] / n_batches,
            "rec": run["rec"] / n_batches,
            "q1_cb": run["q1_cb"] / n_batches,
            "q1_cm": run["q1_cm"] / n_batches,
            "q2_cb": run["q2_cb"] / n_batches,
            "q2_cm": run["q2_cm"] / n_batches,
        }
        stage1_history.append(epoch_stats)
        print(
            f"[Stage1] {epoch}/{trial_cfg.stage1_epochs} "
            f"total={epoch_stats['total']:.4f} rec={epoch_stats['rec']:.4f} "
            f"q1_cb={epoch_stats['q1_cb']:.4f} q2_cb={epoch_stats['q2_cb']:.4f}"
        )

    # Stage 2
    paper_model = AudDSRPaperLike(
        corruption_cfg=trial_cfg.corruption_cfg,
        detector_cfg=trial_cfg.detector_cfg,
        discrete_ae=discrete_ae,
    ).to(device)
    paper_model.freeze_discrete_modules()

    stage2_params = [p for p in paper_model.parameters() if p.requires_grad]
    opt_stage2 = torch.optim.Adam(stage2_params, lr=trial_cfg.lr_stage2)

    stage2_history: List[Dict[str, float]] = []
    paper_model.train()
    for epoch in range(1, trial_cfg.stage2_epochs + 1):
        run = {"total": 0.0, "rec_specific": 0.0, "det": 0.0}
        for batch in stage2_loader:
            x = batch[0].to(device)
            opt_stage2.zero_grad()
            outputs = paper_model(
                x,
                apply_anomaly=True,
                anomaly_batch_ratio=trial_cfg.stage2_anomaly_batch_ratio,
            )
            loss, parts = stage2_auddsr_loss(
                x,
                outputs,
                lambda_rec=trial_cfg.lambda_stage2_rec,
                lambda_det=trial_cfg.lambda_stage2_det,
                alpha=trial_cfg.focal_alpha,
                gamma=trial_cfg.focal_gamma,
            )
            loss.backward()
            opt_stage2.step()
            for k in run:
                run[k] += parts[k]

        n_batches = max(1, len(stage2_loader))
        epoch_stats = {
            "epoch": float(epoch),
            "total": run["total"] / n_batches,
            "rec_specific": run["rec_specific"] / n_batches,
            "det": run["det"] / n_batches,
        }
        stage2_history.append(epoch_stats)
        print(
            f"[Stage2] {epoch}/{trial_cfg.stage2_epochs} "
            f"total={epoch_stats['total']:.4f} rec={epoch_stats['rec_specific']:.4f} det={epoch_stats['det']:.4f}"
        )

    # Evaluation
    train_scores = score_file_list(
        paper_model,
        stage2_train_files,
        mel_transform,
        trial_cfg.frontend_cfg,
        topk_ratio=topk_ratio,
        device=device,
        target_sr=target_sr,
        spec_size=spec_size,
    )
    test_scores = score_file_list(
        paper_model,
        test_files,
        mel_transform,
        trial_cfg.frontend_cfg,
        topk_ratio=topk_ratio,
        device=device,
        target_sr=target_sr,
        spec_size=spec_size,
    )

    train_threshold = float(np.percentile(train_scores, 95))
    auroc, pauc_01, labeled_count = compute_auc_metrics(test_files, test_scores)

    test_std = float(np.std(test_scores))
    train_std = float(np.std(train_scores))

    metrics = {
        "train_files": len(stage2_train_files),
        "test_files": len(test_files),
        "train_threshold_p95": train_threshold,
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": train_std,
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": test_std,
        "auroc": auroc,
        "pauc_0_1": pauc_01,
        "labeled_test_files": labeled_count,
    }

    return {
        "paper_model": paper_model,
        "discrete_ae": discrete_ae,
        "mel_transform": mel_transform,
        "trial_cfg": trial_cfg,
        "metrics": metrics,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "train_files": stage2_train_files,
        "test_files": test_files,
    }


def save_training_curves(result: Dict, save_dir: Path) -> None:
    stage1_history: List[Dict[str, float]] = result.get("stage1_history", [])
    stage2_history: List[Dict[str, float]] = result.get("stage2_history", [])

    curves_payload = {
        "stage1": stage1_history,
        "stage2": stage2_history,
    }
    with open(save_dir / "training_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves_payload, f, indent=2)

    if not stage1_history and not stage2_history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if stage1_history:
        x1 = [int(ep["epoch"]) for ep in stage1_history]
        axes[0].plot(x1, [ep["total"] for ep in stage1_history], label="total")
        axes[0].plot(x1, [ep["rec"] for ep in stage1_history], label="rec")
        axes[0].plot(x1, [ep["q1_cb"] for ep in stage1_history], label="q1_cb")
        axes[0].plot(x1, [ep["q2_cb"] for ep in stage1_history], label="q2_cb")
        axes[0].set_title("Stage 1 Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].set_title("Stage 1 Loss Curves")
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")

    if stage2_history:
        x2 = [int(ep["epoch"]) for ep in stage2_history]
        axes[1].plot(x2, [ep["total"] for ep in stage2_history], label="total")
        axes[1].plot(x2, [ep["rec_specific"] for ep in stage2_history], label="rec_specific")
        axes[1].plot(x2, [ep["det"] for ep in stage2_history], label="det")
        axes[1].set_title("Stage 2 Loss Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_title("Stage 2 Loss Curves")
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def save_trial_checkpoint(result: Dict, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint.pt"

    payload = {
        "trial_cfg": asdict(result["trial_cfg"]),
        "metrics": result["metrics"],
        "stage1_history": result.get("stage1_history", []),
        "stage2_history": result.get("stage2_history", []),
        "paper_model_state": result["paper_model"].state_dict(),
    }
    torch.save(payload, ckpt_path)

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    with open(save_dir / "trial_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result["trial_cfg"]), f, indent=2)

    return ckpt_path


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[AudDSRPaperLike, Dict, Dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload["trial_cfg"]

    dae = DiscreteAutoencoderPaper(
        in_ch=1,
        ch1=64,
        ch2=128,
        q1_embeddings=int(cfg["q1_embeddings"]),
        q2_embeddings=int(cfg["q2_embeddings"]),
    )

    model = AudDSRPaperLike(
        corruption_cfg=cfg["corruption_cfg"],
        detector_cfg=cfg["detector_cfg"],
        discrete_ae=dae,
    ).to(device)

    model.load_state_dict(payload["paper_model_state"], strict=True)
    model.eval()
    metrics = payload.get("metrics", {})
    metrics["stage1_history"] = payload.get("stage1_history", [])
    metrics["stage2_history"] = payload.get("stage2_history", [])
    return model, cfg, metrics


# -----------------------------
# Grid search
# -----------------------------
def parse_csv_list(raw: str, cast_fn):
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return [cast_fn(x) for x in items]


def build_trial_grid(args) -> List[TrialConfig]:
    q_names = parse_csv_list(args.quantizer_grid, str)
    f_names = parse_csv_list(args.frontend_grid, str)
    c_names = parse_csv_list(args.corruption_grid, str)
    d_names = parse_csv_list(args.detector_grid, str)

    stage1_epochs_grid = parse_csv_list(args.stage1_epochs_grid, int)
    stage2_epochs_grid = parse_csv_list(args.stage2_epochs_grid, int)
    lr_stage1_grid = parse_csv_list(args.lr_stage1_grid, float)
    lr_stage2_grid = parse_csv_list(args.lr_stage2_grid, float)
    lambda_det_grid = parse_csv_list(args.lambda_stage2_det_grid, float)

    grid: List[TrialConfig] = []
    for qn, fn, cn, dn, s1e, s2e, lr1, lr2, lam_det in product(
        q_names,
        f_names,
        c_names,
        d_names,
        stage1_epochs_grid,
        stage2_epochs_grid,
        lr_stage1_grid,
        lr_stage2_grid,
        lambda_det_grid,
    ):
        if qn not in QUANTIZER_OPTIONS:
            raise ValueError(f"Unknown quantizer option: {qn}")
        if fn not in FRONTEND_OPTIONS:
            raise ValueError(f"Unknown frontend option: {fn}")
        if cn not in CORRUPTION_OPTIONS:
            raise ValueError(f"Unknown corruption option: {cn}")
        if dn not in DETECTOR_OPTIONS:
            raise ValueError(f"Unknown detector option: {dn}")

        q_cfg = QUANTIZER_OPTIONS[qn]
        f_cfg = FRONTEND_OPTIONS[fn]
        c_cfg = CORRUPTION_OPTIONS[cn]
        d_cfg = DETECTOR_OPTIONS[dn]

        trial = TrialConfig(
            quantizer_name=qn,
            frontend_name=fn,
            corruption_name=cn,
            detector_name=dn,
            q1_embeddings=int(q_cfg["q1_embeddings"]),
            q2_embeddings=int(q_cfg["q2_embeddings"]),
            frontend_cfg=dict(f_cfg),
            corruption_cfg=dict(c_cfg),
            detector_cfg=dict(d_cfg),
            stage1_epochs=int(s1e),
            stage2_epochs=int(s2e),
            lr_stage1=float(lr1),
            lr_stage2=float(lr2),
            lambda_x=float(args.lambda_x),
            lambda_k=float(args.lambda_k),
            lambda_stage2_rec=float(args.lambda_stage2_rec),
            lambda_stage2_det=float(lam_det),
            focal_alpha=float(args.focal_alpha),
            focal_gamma=float(args.focal_gamma),
            stage2_anomaly_batch_ratio=float(args.stage2_anomaly_batch_ratio),
        )
        grid.append(trial)

    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    return grid


def trial_score(metrics: Dict) -> float:
    # Primary objective: AUROC (if available). Fallback: test score std (avoid collapse).
    auroc = metrics.get("auroc")
    if auroc is not None:
        return float(auroc)
    return float(metrics.get("test_score_std", 0.0))


# -----------------------------
# CLI
# -----------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AudDSR grid search trainer")

    p.add_argument("--data-root", type=str, required=True, help="Root of DCASE dev_* folders")
    p.add_argument("--output-dir", type=str, default="outputs_auddsr_grid")

    p.add_argument("--all-machine-types", type=str, default="bearing,fan,gearbox,slider,ToyCar,ToyTrain,valve")
    p.add_argument("--stage1-machine-types", type=str, default="bearing,fan,gearbox,slider,ToyCar,ToyTrain,valve")
    p.add_argument("--stage2-machine-types", type=str, default="bearing,fan,gearbox,slider,ToyCar,ToyTrain,valve")

    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--spec-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--topk-ratio", type=float, default=0.05)
    p.add_argument("--subset-train-per-machine", type=int, default=0, help="0 = use all train files; else cap files per machine")
    p.add_argument("--subset-test-per-machine", type=int, default=0, help="0 = use all test files; else cap files per machine")

    # Grid axes
    p.add_argument("--quantizer-grid", type=str, default="base,large")
    p.add_argument("--frontend-grid", type=str, default="balanced,harmonic")
    p.add_argument("--corruption-grid", type=str, default="band_time,perlin_like")
    p.add_argument("--detector-grid", type=str, default="light,wide")
    p.add_argument("--stage1-epochs-grid", type=str, default="5,10")
    p.add_argument("--stage2-epochs-grid", type=str, default="5,10")
    p.add_argument("--lr-stage1-grid", type=str, default="2e-4")
    p.add_argument("--lr-stage2-grid", type=str, default="2e-4")
    p.add_argument("--lambda-stage2-det-grid", type=str, default="1.0,5.0")

    # Fixed loss params
    p.add_argument("--lambda-x", type=float, default=1.0)
    p.add_argument("--lambda-k", type=float, default=0.25)
    p.add_argument("--lambda-stage2-rec", type=float, default=1.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--stage2-anomaly-batch-ratio", type=float, default=0.5)

    p.add_argument("--max-trials", type=int, default=0, help="0 means all combinations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--resume-checkpoint", type=str, default="", help="Load one checkpoint and only run eval print")
    p.add_argument("--no-save-curves", action="store_true", help="Disable saving training curve plots/json per trial")
    p.add_argument("--preset", type=str, default="none", choices=["none", "laptop30"], help="Apply a runtime preset")
    return p


def apply_preset(args) -> None:
    if args.preset == "none":
        return

    if args.preset == "laptop30":
        # Tuned for a quick run likely within ~30 min on a typical laptop GPU.
        args.quantizer_grid = "small"
        args.frontend_grid = "balanced"
        args.corruption_grid = "band_time"
        args.detector_grid = "light"
        args.stage1_epochs_grid = "2"
        args.stage2_epochs_grid = "2"
        args.lr_stage1_grid = "2e-4"
        args.lr_stage2_grid = "2e-4"
        args.lambda_stage2_det_grid = "3.0"
        args.max_trials = 1
        args.batch_size = 8
        args.num_workers = 2
        args.subset_train_per_machine = 150
        args.subset_test_per_machine = 60
        return

    raise ValueError(f"Unknown preset: {args.preset}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = make_parser().parse_args()
    apply_preset(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Preset: {args.preset}")

    if args.resume_checkpoint:
        model, cfg, metrics = load_model_from_checkpoint(args.resume_checkpoint, device)
        print("Checkpoint loaded successfully")
        print(json.dumps({"cfg": cfg, "metrics": metrics}, indent=2))
        _ = model  # keep explicit for clarity
        return

    all_machine_types = parse_csv_list(args.all_machine_types, str)
    stage1_machine_types = parse_csv_list(args.stage1_machine_types, str)
    stage2_machine_types = parse_csv_list(args.stage2_machine_types, str)

    trial_grid = build_trial_grid(args)
    if not trial_grid:
        raise RuntimeError("Trial grid is empty")

    print(f"Total trials: {len(trial_grid)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: List[Dict] = []
    best_record: Optional[Dict] = None

    for idx, trial_cfg in enumerate(trial_grid, start=1):
        trial_name = (
            f"trial_{idx:03d}_q-{trial_cfg.quantizer_name}_f-{trial_cfg.frontend_name}_"
            f"c-{trial_cfg.corruption_name}_d-{trial_cfg.detector_name}_"
            f"s1-{trial_cfg.stage1_epochs}_s2-{trial_cfg.stage2_epochs}_"
            f"detw-{trial_cfg.lambda_stage2_det:g}"
        )
        trial_dir = output_dir / trial_name

        print("=" * 100)
        print(f"[{idx}/{len(trial_grid)}] {trial_name}")
        print("=" * 100)

        result = train_trial(
            trial_cfg=trial_cfg,
            data_root=args.data_root,
            all_machine_types=all_machine_types,
            stage1_machine_types=stage1_machine_types,
            stage2_machine_types=stage2_machine_types,
            subset_train_per_machine=args.subset_train_per_machine,
            subset_test_per_machine=args.subset_test_per_machine,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_sr=args.target_sr,
            spec_size=args.spec_size,
            topk_ratio=args.topk_ratio,
            device=device,
            seed=args.seed,
        )

        ckpt_path = save_trial_checkpoint(result, trial_dir)
        if not args.no_save_curves:
            save_training_curves(result, trial_dir)
        metrics = result["metrics"]

        record = {
            "trial_name": trial_name,
            "checkpoint": str(ckpt_path),
            "trial_cfg": asdict(trial_cfg),
            "metrics": metrics,
            "score": trial_score(metrics),
        }
        leaderboard.append(record)

        if best_record is None or record["score"] > best_record["score"]:
            best_record = record
            with open(output_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
                f.write(record["checkpoint"] + "\n")

        with open(output_dir / "leaderboard.json", "w", encoding="utf-8") as f:
            json.dump(sorted(leaderboard, key=lambda x: x["score"], reverse=True), f, indent=2)

        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

    assert best_record is not None
    print("\nGrid search completed")
    print(f"Best trial: {best_record['trial_name']}")
    print(f"Best checkpoint: {best_record['checkpoint']}")
    print(f"Best score: {best_record['score']:.6f}")


if __name__ == "__main__":
    main()
