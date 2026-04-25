import argparse
import os
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile


def _resolve_dataset_root(input_root: Path) -> Path:
    direct_pump = input_root / "Pump"
    direct_bearing = input_root / "Bearing"
    if direct_pump.is_dir() and direct_bearing.is_dir():
        return input_root

    nested = input_root / "Data Sorama"
    nested_pump = nested / "Pump"
    nested_bearing = nested / "Bearing"
    if nested_pump.is_dir() and nested_bearing.is_dir():
        return nested

    raise FileNotFoundError(
        f"Could not find Pump/ and Bearing/ under: {input_root}"
    )


def _iter_wav_files(root: Path) -> Iterable[Path]:
    for current_root, _, files in os.walk(root):
        for filename in sorted(files):
            if filename.lower().endswith(".wav"):
                yield Path(current_root) / filename


def _load_audio_mono(file_path: Path) -> Tuple[np.ndarray, int]:
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            sr = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)

        if sample_width == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(
                f"Unsupported sample width {sample_width} in {file_path}"
            )

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio.astype(np.float32), sr
    except wave.Error:
        # Fallback for non-PCM WAV formats (e.g., IEEE float, format tag 3).
        try:
            sr, raw_audio = wavfile.read(str(file_path))
        except Exception as ex:
            raise ValueError(f"Could not decode WAV file {file_path}: {ex}") from ex

        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)

        if np.issubdtype(raw_audio.dtype, np.floating):
            audio = raw_audio.astype(np.float32)
        elif raw_audio.dtype == np.uint8:
            audio = (raw_audio.astype(np.float32) - 128.0) / 128.0
        elif raw_audio.dtype == np.int16:
            audio = raw_audio.astype(np.float32) / 32768.0
        elif raw_audio.dtype == np.int32:
            audio = raw_audio.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV dtype {raw_audio.dtype} in {file_path}")

        return audio.astype(np.float32), int(sr)


def _write_wav_pcm16(file_path: Path, audio: np.ndarray, sr: int) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm16.tobytes())


def _resample_linear(audio: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) == target_len:
        return audio.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-12:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def _normal_variant(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    stretch = rng.uniform(0.95, 1.05)
    stretched_len = max(1, int(round(len(audio) * stretch)))
    y = _resample_linear(audio, stretched_len)
    y = _resample_linear(y, len(audio))

    if y.size:
        shift = int(rng.integers(-max(1, len(y) // 30), max(1, len(y) // 30) + 1))
        y = np.roll(y, shift)

    y *= rng.uniform(0.85, 1.12)
    y += rng.normal(0.0, rng.uniform(0.001, 0.015), size=y.shape[0]).astype(np.float32)
    return _normalize(y)


def _anomaly_variant(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    y = _normal_variant(audio, rng)
    n = len(y)
    if n == 0:
        return y

    t = np.arange(n, dtype=np.float32) / float(sr)
    tone_freq = rng.uniform(1200.0, 6000.0)
    y += 0.12 * np.sin(2.0 * np.pi * tone_freq * t + rng.uniform(0.0, 2.0 * np.pi)).astype(np.float32)

    burst_count = int(rng.integers(2, 6))
    for _ in range(burst_count):
        start = int(rng.integers(0, max(1, n - max(2, sr // 5))))
        burst_len = int(rng.integers(max(2, sr // 80), max(3, sr // 10)))
        end = min(n, start + burst_len)
        if end <= start:
            continue
        window = np.hanning(end - start).astype(np.float32)
        burst = rng.normal(0.0, rng.uniform(0.35, 0.9), size=end - start).astype(np.float32)
        y[start:end] += burst * window

    if rng.random() < 0.5:
        d_start = int(rng.integers(0, max(1, n - max(2, sr // 3))))
        d_len = int(rng.integers(max(2, sr // 90), max(3, sr // 20)))
        d_end = min(n, d_start + d_len)
        y[d_start:d_end] *= rng.uniform(0.0, 0.35)

    return _normalize(y)


def _target_sr(src_sr: int, requested_sr: int) -> int:
    return src_sr if requested_sr <= 0 else requested_sr


def _maybe_resample_to_sr(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    target_len = max(1, int(round(len(audio) * dst_sr / float(src_sr))))
    return _resample_linear(audio, target_len)


def _category_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if "0_normal" in parts:
        return "normal"
    if "1_anomaly" in parts:
        return "anomaly"
    return None


def _build_destination_path(
    source_file: Path,
    source_dataset_root: Path,
    destination_dataset_root: Path,
    copy_index: int,
) -> Path:
    rel = source_file.relative_to(source_dataset_root)
    stem = source_file.stem
    new_name = f"{stem}_syn{copy_index:03d}.wav"
    return destination_dataset_root / rel.parent / new_name


def _synthesize_root(
    source_dataset_root: Path,
    destination_dataset_root: Path,
    copies_per_source: int,
    requested_sr: int,
    seed: int,
    dry_run: bool,
) -> Tuple[int, int]:
    copied = 0
    skipped = 0

    for wav_path in _iter_wav_files(source_dataset_root):
        category = _category_from_path(wav_path)
        if category is None:
            skipped += 1
            continue

        if dry_run:
            copied += copies_per_source
            continue

        try:
            audio, src_sr = _load_audio_mono(wav_path)
        except Exception as ex:
            skipped += 1
            print(f"[WARN] skipping unreadable file: {wav_path} ({ex})")
            continue

        dst_sr = _target_sr(src_sr, requested_sr)
        base_audio = _maybe_resample_to_sr(audio, src_sr, dst_sr)

        for i in range(copies_per_source):
            rng = np.random.default_rng(seed + (hash(str(wav_path)) % 1000003) + i)
            if category == "normal":
                out_audio = _normal_variant(base_audio, rng)
            else:
                out_audio = _anomaly_variant(base_audio, dst_sr, rng)

            dst_path = _build_destination_path(
                source_file=wav_path,
                source_dataset_root=source_dataset_root,
                destination_dataset_root=destination_dataset_root,
                copy_index=i,
            )

            _write_wav_pcm16(dst_path, out_audio, dst_sr)

            copied += 1

    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic Sorama-style audio from existing Sorama WAV files, "
            "preserving Pump/Bearing, Asset, and 0_normal/1_anomaly folder structure."
        )
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help=(
            "One or more roots. Each root can either directly contain Pump/Bearing or "
            "contain a nested 'Data Sorama' folder with Pump/Bearing inside."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination folder for generated synthetic dataset.",
    )
    parser.add_argument(
        "--prefix-with-root-name",
        action="store_true",
        help="Write each input root under output_root/<input_root_name> to avoid collisions.",
    )
    parser.add_argument(
        "--copies-per-source",
        type=int,
        default=1,
        help="How many synthetic files to generate per source file.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        help="Target sample rate. 0 keeps each source file's original sample rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthesis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file counts without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.copies_per_source < 1:
        raise ValueError("--copies-per-source must be >= 1")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_skipped = 0

    for input_root_str in args.input_roots:
        input_root = Path(input_root_str).expanduser().resolve()
        if not input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {input_root}")

        dataset_root = _resolve_dataset_root(input_root)
        destination_root = (
            output_root / input_root.name if args.prefix_with_root_name else output_root
        )

        generated, skipped = _synthesize_root(
            source_dataset_root=dataset_root,
            destination_dataset_root=destination_root,
            copies_per_source=args.copies_per_source,
            requested_sr=args.sample_rate,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        total_generated += generated
        total_skipped += skipped

        mode = "DRY-RUN" if args.dry_run else "GENERATE"
        print(
            f"[{mode}] {input_root} -> {destination_root} | generated={generated}, skipped={skipped}"
        )

    print("\nDone")
    print(f"Total generated files: {total_generated}")
    print(f"Total skipped files  : {total_skipped}")


if __name__ == "__main__":
    main()
