# this file reads wav audio files for the whole pipeline
# dcase wav files sometimes use a special format that normal loaders miss
# we try soundfile first and fall back to torchaudio if needed
# all audio is converted to mono at 16 khz because the models expect that

from __future__ import annotations

import torch
import torchaudio


def load_wav_mono(
    file_path: str,
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int]:
    # step one read the file with soundfile when possible
    try:
        import soundfile as sf

        audio, source_sr = sf.read(file_path, dtype="float32", always_2d=True)
        # step two mix down to one channel if the file has stereo
        waveform = torch.from_numpy(audio.mean(axis=1)).float().unsqueeze(0)
        source_sr = int(source_sr)
    except Exception:
        # step three use torchaudio as backup
        waveform, source_sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

    # step four resample if the file is not already at target sample rate
    if int(source_sr) != int(target_sr):
        waveform = torchaudio.functional.resample(waveform, int(source_sr), int(target_sr))
    return waveform, int(target_sr)


def load_wav_mono_1d(path: str, target_sr: int = 16000) -> torch.Tensor:
    # same as load wav mono but returns a flat one dimensional tensor for tse
    waveform, _ = load_wav_mono(path, target_sr=target_sr)
    return waveform.squeeze(0).float()
