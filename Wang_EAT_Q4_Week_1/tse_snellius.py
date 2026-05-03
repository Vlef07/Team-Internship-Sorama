"""Target signal enhancement (TSE) module, gebaseerd op Fujimura et al. (2025).

TSE-module naar het idee van Fujimura e.a. (2025), NU technical report DCASE 2025 Task 2.

Gedeeld door het notebook, train_tse_snellius.py en (optioneel) de eval.

In het paper gebruiken ze per machine een TSE-model, reconstructieloss L_D (negatieve SNR)
plus een gebalanceerde classificatieterm met een frontend-classifier, architectuur o.a.
TF-Locoformer met STFT (bv. DFT 512, shift 128).

We gebruiken een eenvoudigere variant dan in het Fujimura paper. 
De golfvorm gaat naar een STFT (het geluid wordt per kort tijdstukje uitgesplitst 
in frequenties naar een spectrogram). Alleen de magnitudes (hoe sterk elk frequentie-onderdeel is) 
gaan door een klein U-Net (een netwerk dat beeld patronen kan leren). Het net leert een 
mask tussen 0 en 1, per vakje is de vraag hoeveel van dit frequentie-onderdeel we door laten. Daarna, 
maskeermagnitude → inverse STFT (iSTFT) terug naar geluid, de fase (de fijne timing/klank van de 
originele opname) nemen we van de ingang, zodat het beeld niet volledig kunstmatig klinkt. 
Trainen minemen een SNR-loss, we hebben geen extra classificatie-loss zoals in het paper, die het daar nog naast 
de reconstructie zet. Het doel blijft om achtergrondruis iets terug te dringen zodat het 
machinesignaal duidelijker wordt voor de volgende stappen.


Reference
  Fujimura, T., Kuroyanagi, I., and Toda, T. (2025). The NU systems for DCASE
  2025 challenge task 2 (Technical report). Nagoya University.
"""

import os
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


TSE_SAMPLE_RATE = 16000
TSE_SEGMENT_SECONDS = 6.0
TSE_N_FFT = 512
TSE_HOP_LENGTH = 128


def _find_wavs(folder: str) -> List[str]:
    """Return sorted .wav paths directly in folder (not recursive)."""
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".wav"):
            out.append(os.path.join(folder, name))
    return out


def load_wav_mono(path: str, target_sr: int = TSE_SAMPLE_RATE) -> torch.Tensor:
    """Load a wav, convert to mono float tensor at target_sr, shape [T]."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).float()


def pad_or_crop(wav: torch.Tensor, length: int, random_crop: bool = True) -> torch.Tensor:
    """Return a fixed length segment from a 1D waveform."""
    if wav.shape[-1] >= length:
        if random_crop:
            start = random.randint(0, wav.shape[-1] - length)
        else:
            start = 0
        return wav[start : start + length]
    pad = length - wav.shape[-1]
    return F.pad(wav, (0, pad))


class TSEWaveformDataset(torch.utils.data.Dataset):
    """Yields fixed length clean waveforms for one target machine type.

    In Fujimura et al. (2025) the target is "clean machine sound" from the
    supplementary data. When supplementary clean data is not available, we use
    the regular training wavs of the target machine as the best local proxy,
    because they are the cleanest recordings we have of that machine.
    """

    def __init__(
        self,
        wav_paths: List[str],
        target_sr: int = TSE_SAMPLE_RATE,
        segment_seconds: float = TSE_SEGMENT_SECONDS,
    ):
        self.wav_paths = wav_paths
        self.target_sr = target_sr
        self.segment_length = max(1, int(segment_seconds * target_sr))

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = load_wav_mono(self.wav_paths[idx], self.target_sr)
        wav = pad_or_crop(wav, self.segment_length, random_crop=True)
        wav = wav - wav.mean()
        return wav


class CrossMachineNoiseBank:
    """Pool of noise waveforms taken from other machine types.

    Rationale: Fujimura et al. (2025) mix the target signal with AudioSet
    noise at random SNR. When AudioSet is not available, the closest local
    substitute is wavs from machine types other than the target, so the TSE
    model learns to remove "sounds that are not the target machine".
    """

    def __init__(
        self,
        base_path: str,
        target_machine: str,
        target_sr: int = TSE_SAMPLE_RATE,
        max_noise_files: int = 200,
        segment_seconds: float = TSE_SEGMENT_SECONDS,
        seed: int = 0,
    ):
        self.target_sr = target_sr
        self.segment_length = max(1, int(segment_seconds * target_sr))

        rng = random.Random(seed)
        candidates: List[str] = []
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"base_path bestaat niet: {base_path}")
        for machine in sorted(os.listdir(base_path)):
            if machine == target_machine:
                continue
            train_dir = os.path.join(base_path, machine, "train")
            for p in _find_wavs(train_dir):
                name = os.path.basename(p).lower()
                if "anomaly" in name:
                    continue
                candidates.append(p)
        if not candidates:
            raise RuntimeError(
                f"Geen noise wavs gevonden voor target {target_machine!r} onder {base_path}"
            )
        rng.shuffle(candidates)
        self.noise_paths = candidates[:max_noise_files]

    def sample(self, length: Optional[int] = None) -> torch.Tensor:
        length = self.segment_length if length is None else length
        path = random.choice(self.noise_paths)
        noise = load_wav_mono(path, self.target_sr)
        noise = noise - noise.mean()
        return pad_or_crop(noise, length, random_crop=True)


def mix_at_random_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db_range: Tuple[float, float] = (-5.0, 5.0),
) -> torch.Tensor:
    """Mix clean and noise at a random SNR in snr_db_range.

    clean and noise share the same 1D shape. Uses the Fujimura range [-5, 5) dB.
    """
    eps = 1e-8
    snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
    clean_p = (clean ** 2).mean() + eps
    noise_p = (noise ** 2).mean() + eps
    target_noise_p = clean_p / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(target_noise_p / noise_p)
    return clean + noise * scale


def neg_snr_loss(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative reconstruction SNR in dB (Fujimura et al. 2025, L_D as negative SNR loss).

    SNR_dB = 10*log10( mean(y^2) / mean((y_hat - y)^2) ) per item; return
    -mean(SNR_dB) over batch. Lower loss => higher SNR. Shapes: [B, T] or [T].
    """
    if est.dim() == 1:
        est = est.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    err = est - target
    p_sig = (target ** 2).mean(dim=-1) + eps
    p_err = (err ** 2).mean(dim=-1) + eps
    snr_db = 10.0 * torch.log10(p_sig / p_err)
    return -snr_db.mean()


@torch.no_grad()
def mean_snr_db(
    est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean reconstruction SNR in dB; **higher is better** (same definition as :func:`neg_snr_loss`)."""
    if est.dim() == 1:
        est = est.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    err = est - target
    p_sig = (target ** 2).mean(dim=-1) + eps
    p_err = (err ** 2).mean(dim=-1) + eps
    return (10.0 * torch.log10(p_sig / p_err)).mean()


class TSEMaskNet(nn.Module):
    """Small U-Net that predicts a magnitude mask on the STFT.

    Input: batch of waveforms, shape [B, T].
    Output: batch of enhanced waveforms, shape [B, T].

    The encoder halves the frequency axis twice, the decoder mirrors that. We
    use bilinear interpolation to match skip shapes if the strided conv picks
    up an extra frame due to padding. The output mask is bounded in [0, 1] via
    sigmoid, so the enhanced magnitude can never exceed the noisy magnitude.
    """

    def __init__(
        self,
        n_fft: int = TSE_N_FFT,
        hop_length: int = TSE_HOP_LENGTH,
        base_channels: int = 16,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        c = base_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * c, 4 * c, 3, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(c, 1, 1)

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
            center=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]

        spec = self._stft(wav)
        mag = spec.abs()
        phase = spec / (mag + 1e-8)

        x = mag.unsqueeze(1)

        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bottleneck(d2)

        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="nearest")
        u2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(u2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="nearest")
        u1 = self.dec1(torch.cat([u1, e1], dim=1))

        mask = torch.sigmoid(self.mask_head(u1)).squeeze(1)

        enhanced_mag = mask * mag
        enhanced_spec = enhanced_mag * phase
        return self._istft(enhanced_spec, length=length)


def list_target_wavs(base_path: str, machine: str) -> List[str]:
    """List training wavs of the target machine, excluding anomalies."""
    train_dir = os.path.join(base_path, machine, "train")
    out: List[str] = []
    for p in _find_wavs(train_dir):
        low = os.path.basename(p).lower()
        if "anomaly" in low:
            continue
        out.append(p)
    return out


def list_machines(base_path: str) -> List[str]:
    return sorted(
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    )


@torch.no_grad()
def enhance_waveform(
    tse_model: TSEMaskNet, wav: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Apply TSE to a single waveform [T] or batch [B, T], return same shape."""
    tse_model.eval()
    single = wav.dim() == 1
    if single:
        wav = wav.unsqueeze(0)
    out = tse_model(wav.to(device)).cpu()
    return out.squeeze(0) if single else out


def load_tse_models(
    tse_checkpoint_dir: str, device: torch.device
) -> dict:
    """Load a {machine_name: TSEMaskNet} dict from a directory of {machine}.pt files."""
    models: dict = {}
    if not tse_checkpoint_dir or not os.path.isdir(tse_checkpoint_dir):
        return models
    for name in sorted(os.listdir(tse_checkpoint_dir)):
        if not name.lower().endswith(".pt"):
            continue
        machine = os.path.splitext(name)[0]
        path = os.path.join(tse_checkpoint_dir, name)
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model = TSEMaskNet().to(device)
        model.load_state_dict(state)
        model.eval()
        models[machine] = model
    return models
