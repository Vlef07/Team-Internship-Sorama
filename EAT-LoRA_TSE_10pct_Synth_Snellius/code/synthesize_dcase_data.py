"""
this script builds synthetic training audio from real additional dataset wav files
step one walk each machine folder and read normal train clips from disk
step two for each source clip create new variants with small time stretch noise and tone changes
step three write new train test normal and test anomaly wav files into the synth output folder
step four copy machine attributes from attributes_00.csv so labels stay consistent
step five optionally add background noise at a fixed snr such as 15 db before peak normalization
only train clips from the input dataset are used as sources
the synth output gives the pipeline more training variety while the raw additional dataset stays unchanged
"""
import argparse
import json
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SourceClip:
    audio_path: str
    stem: str
    domain: str
    attributes: Dict[str, str]


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_wav_pcm16(path: str, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs < 1e-12:
        return x.astype(np.float32)
    return (x / max_abs * peak).astype(np.float32)


def _infer_domain(text: str) -> str:
    lowered = str(text).lower()
    if "source" in lowered:
        return "source"
    if "target" in lowered:
        return "target"
    return "source"


def _extract_attribute_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("v")]


def _normalize_attr_value(v) -> str:
    if pd.isna(v):
        return "noAttribute"
    s = str(v).strip()
    if not s:
        return "noAttribute"
    try:
        x = float(s.replace(",", "."))
        if x == int(x):
            return str(int(x))
    except ValueError:
        pass
    return s


def _load_audio_mono(file_path: str, sample_rate: int) -> np.ndarray:
    """Load mono float audio. Uses soundfile for DCASE WAV (incl. format 65534 / EXTENSIBLE)."""
    try:
        import soundfile as sf

        audio, source_sr = sf.read(file_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
    except Exception as sf_err:
        try:
            with wave.open(file_path, "rb") as wav_file:
                source_sr = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                raw_audio = wav_file.readframes(n_frames)

            if sample_width == 1:
                audio = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0
            elif sample_width == 2:
                audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(
                    f"Unsupported WAV sample width: {sample_width} bytes in {file_path}"
                ) from sf_err

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            audio = audio.astype(np.float32)
        except Exception as wave_err:
            raise RuntimeError(
                f"Could not read WAV {file_path} with soundfile or wave module"
            ) from wave_err

    if int(source_sr) != int(sample_rate):
        target_length = max(1, int(round(len(audio) * sample_rate / float(source_sr))))
        audio = _resample_audio(audio, target_length)

    return audio.astype(np.float32)


def _resample_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(audio) == target_length:
        return audio.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _match_length(audio: np.ndarray, target_samples: int, rng: np.random.Generator) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if len(audio) > target_samples:
        start = int(rng.integers(0, len(audio) - target_samples + 1))
        return audio[start : start + target_samples].astype(np.float32)

    if len(audio) < target_samples:
        repeats = int(np.ceil(target_samples / len(audio)))
        tiled = np.tile(audio, repeats)
        return tiled[:target_samples].astype(np.float32)

    return audio.astype(np.float32)


# Zachte volume-schommeling over de hele clip zodat normale varianten niet allemaal plat klinken.
def _additive_noise_for_variant(
    variant_before_noise: np.ndarray,
    target_samples: int,
    rng: np.random.Generator,
    synth_noise_snr_db: Optional[float],
) -> np.ndarray:
    """Voeg witte Gaussian ruis toe. Als synth_noise_snr_db gezet is, kies ruis zodat
    signaal t.o.v. ruis het gevraagde SNR heeft vóór normalisatie, zelfde idee als
    vermogen SNR gebruikt in Fujimura TSE mixing."""

    variant_before_noise = np.asarray(variant_before_noise, dtype=np.float64)
    if synth_noise_snr_db is not None:
        eps = 1e-12
        p_sig = float(np.mean(variant_before_noise**2)) + eps
        p_noise = p_sig / (10.0 ** (float(synth_noise_snr_db) / 10.0))
        sigma = np.sqrt(max(p_noise, eps))
        return rng.normal(0.0, sigma, size=target_samples).astype(np.float32)
    return rng.normal(0.0, rng.uniform(0.002, 0.02), size=target_samples).astype(np.float32)


def _slow_amplitude_envelope(target_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(target_samples, dtype=np.float32) / float(max(target_samples, 1))
    freq = rng.uniform(0.2, 1.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return (0.92 + 0.08 * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)


# Licht bewerkte normale golfvorm: minieme stretch, shift, ruis en envelop, daarna genormaliseerd.
def _make_normal_variant(
    audio: np.ndarray,
    sample_rate: int,
    target_samples: int,
    rng: np.random.Generator,
    synth_noise_snr_db: Optional[float] = None,
) -> np.ndarray:
    stretched_len = max(1, int(round(len(audio) * rng.uniform(0.94, 1.06))))
    variant = _resample_audio(audio, stretched_len)
    variant = _match_length(variant, target_samples, rng)

    if rng.random() < 0.7:
        shift = int(rng.integers(-target_samples // 20, target_samples // 20 + 1))
        variant = np.roll(variant, shift)

    variant = variant * rng.uniform(0.8, 1.15)
    variant = variant * _slow_amplitude_envelope(target_samples, rng)
    variant = variant + _additive_noise_for_variant(
        variant, target_samples, rng, synth_noise_snr_db=synth_noise_snr_db
    )
    return _normalize_audio(variant)


# Start als normale variant, voegt sinusstorende korte ruisbursts en soms een gedempt stuk toe.
def _make_anomaly_variant(
    audio: np.ndarray,
    sample_rate: int,
    target_samples: int,
    rng: np.random.Generator,
    synth_noise_snr_db: Optional[float] = None,
) -> np.ndarray:
    variant = _make_normal_variant(
        audio, sample_rate, target_samples, rng, synth_noise_snr_db=synth_noise_snr_db
    ).astype(np.float32)
    t = np.arange(target_samples, dtype=np.float32) / float(sample_rate)

    off_freq = rng.uniform(1200.0, 6000.0)
    variant += 0.14 * np.sin(2.0 * np.pi * off_freq * t + rng.uniform(0.0, 2.0 * np.pi)).astype(np.float32)

    burst_count = int(rng.integers(2, 6))
    for _ in range(burst_count):
        start = int(rng.integers(0, max(1, target_samples - sample_rate // 3)))
        length = int(rng.integers(sample_rate // 60, sample_rate // 8))
        end = min(target_samples, start + length)
        if end <= start:
            continue
        window = np.hanning(end - start).astype(np.float32)
        burst = rng.normal(0.0, rng.uniform(0.35, 0.95), size=end - start).astype(np.float32)
        variant[start:end] += burst * window

    if rng.random() < 0.5:
        dropout_start = int(rng.integers(0, max(1, target_samples - sample_rate // 4)))
        dropout_length = int(rng.integers(sample_rate // 80, sample_rate // 15))
        dropout_end = min(target_samples, dropout_start + dropout_length)
        variant[dropout_start:dropout_end] *= rng.uniform(0.0, 0.3)

    if rng.random() < 0.6:
        variant = np.clip(variant * rng.uniform(1.05, 1.4), -1.0, 1.0)

    return _normalize_audio(variant)


# Elke map met een train-submap telt als één machine (bearing, valve, enzovoort).
def _iter_machine_roots(root: str):
    for current_root, dirnames, _ in os.walk(root):
        if os.path.isdir(os.path.join(current_root, "train")):
            yield current_root
            dirnames[:] = []


def _has_machine_layout(root: str) -> bool:
    return any(True for _ in _iter_machine_roots(root))


def _find_machine_roots(input_root: str, requested: Optional[List[str]]) -> List[tuple[str, str]]:
    if not os.path.isdir(input_root):
        return []

    requested_set = set(requested or [])
    found: Dict[str, str] = {}

    for machine_root in _iter_machine_roots(input_root):
        machine_name = os.path.basename(machine_root)
        if requested_set and machine_name not in requested_set:
            continue
        found.setdefault(machine_name, machine_root)

    if requested_set:
        return [(name, found[name]) for name in sorted(found) if name in requested_set]

    return [(name, found[name]) for name in sorted(found)]


def _discover_input_root(preferred: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []

    if preferred:
        candidates.append(preferred)

    env_root = os.environ.get("DCASE_INPUT_ROOT", "").strip()
    if env_root:
        candidates.append(env_root)

    cwd = os.getcwd()
    candidates.extend(
        [
            cwd,
            os.path.join(cwd, "data", "dcase2025t2", "dev_data", "raw"),
            os.path.join(cwd, "data", "dcase2025t2", "dev_data", "raw_synth_small"),
            os.path.join(cwd, "..", "data", "dcase2025t2", "dev_data", "raw"),
        ]
    )

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if _has_machine_layout(normalized):
            return normalized

    return None


def _load_source_clips(machine_root: str) -> tuple[List[SourceClip], List[str]]:
    csv_path = os.path.join(machine_root, "attributes_00.csv")
    train_dir = os.path.join(machine_root, "train")

    if not os.path.isdir(train_dir):
        return [], []

    attr_cols: List[str] = []
    source_df: Optional[pd.DataFrame] = None
    if os.path.isfile(csv_path):
        source_df = pd.read_csv(csv_path)
        attr_cols = _extract_attribute_columns(source_df)

    if not attr_cols:
        attr_cols = ["attribute_1v"]

    attr_lookup: Dict[str, Dict[str, str]] = {}
    if source_df is not None and "file_name" in source_df.columns:
        train_rows = source_df[source_df["file_name"].str.contains("train", na=False)].copy()
        for _, row in train_rows.iterrows():
            stem = os.path.splitext(os.path.basename(str(row["file_name"])))[0]
            attrs = {col: _normalize_attr_value(row[col]) if col in row.index else "noAttribute" for col in attr_cols}
            attr_lookup[stem] = attrs

    clips: List[SourceClip] = []
    for wav_name in sorted(os.listdir(train_dir)):
        if not wav_name.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(wav_name)[0]
        audio_path = os.path.join(train_dir, wav_name)
        if not os.path.isfile(audio_path):
            continue

        attrs = attr_lookup.get(stem, {col: "noAttribute" for col in attr_cols})
        clips.append(
            SourceClip(
                audio_path=audio_path,
                stem=stem,
                domain=_infer_domain(stem),
                attributes=attrs,
            )
        )

    return clips, attr_cols


def _source_stem_to_train_stem(source_stem: str, copy_index: int) -> str:
    stem = source_stem
    if copy_index > 0:
        stem = f"{stem}_syn{copy_index:03d}"
    return stem


def _source_stem_to_test_stem(source_stem: str, anomaly: bool, copy_index: int) -> str:
    stem = source_stem
    if "_train_normal_" in stem:
        replacement = "_test_anomaly_" if anomaly else "_test_normal_"
        stem = stem.replace("_train_normal_", replacement)
    elif "train_normal" in stem:
        replacement = "test_anomaly" if anomaly else "test_normal"
        stem = stem.replace("train_normal", replacement)
    else:
        stem = f"{stem}_{'test_anomaly' if anomaly else 'test_normal'}"

    if copy_index > 0:
        stem = f"{stem}_syn{copy_index:03d}"
    return stem


def _build_output_rows(
    machine: str,
    source_clips: List[SourceClip],
    copies_per_source: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for clip in source_clips:
        for copy_index in range(copies_per_source):
            train_stem = _source_stem_to_train_stem(clip.stem, copy_index)
            normal_test_stem = _source_stem_to_test_stem(clip.stem, anomaly=False, copy_index=copy_index)
            anomaly_test_stem = _source_stem_to_test_stem(clip.stem, anomaly=True, copy_index=copy_index)

            row_attrs = dict(clip.attributes)
            if not row_attrs:
                row_attrs = {"attribute_1v": "noAttribute"}

            rows.append({"file_name": f"{machine}/train/{train_stem}.wav", **row_attrs})
            rows.append({"file_name": f"{machine}/test/{normal_test_stem}.wav", **row_attrs})
            rows.append({"file_name": f"{machine}/test/{anomaly_test_stem}.wav", **row_attrs})

    return rows


def _synthesize_machine_outputs(
    output_root: str,
    machine: str,
    source_clips: List[SourceClip],
    copies_per_source: int,
    sample_rate: int,
    duration_sec: float,
    rng: np.random.Generator,
    synth_noise_snr_db: Optional[float],
) -> int:
    # for each real train clip make several synthetic train and test wav files
    target_samples = int(sample_rate * duration_sec)
    generated_files = 0

    for clip in source_clips:
        source_audio = _load_audio_mono(clip.audio_path, sample_rate)
        for copy_index in range(copies_per_source):
            train_stem = _source_stem_to_train_stem(clip.stem, copy_index)
            normal_test_stem = _source_stem_to_test_stem(clip.stem, anomaly=False, copy_index=copy_index)
            anomaly_test_stem = _source_stem_to_test_stem(clip.stem, anomaly=True, copy_index=copy_index)

            train_audio = _make_normal_variant(
                source_audio, sample_rate, target_samples, rng, synth_noise_snr_db=synth_noise_snr_db
            )
            test_normal_audio = _make_normal_variant(
                source_audio, sample_rate, target_samples, rng, synth_noise_snr_db=synth_noise_snr_db
            )
            test_anomaly_audio = _make_anomaly_variant(
                source_audio, sample_rate, target_samples, rng, synth_noise_snr_db=synth_noise_snr_db
            )

            train_path = os.path.join(output_root, machine, "train", f"{train_stem}.wav")
            test_normal_path = os.path.join(output_root, machine, "test", f"{normal_test_stem}.wav")
            test_anomaly_path = os.path.join(output_root, machine, "test", f"{anomaly_test_stem}.wav")

            _safe_makedirs(os.path.dirname(train_path))
            _safe_makedirs(os.path.dirname(test_normal_path))
            _safe_makedirs(os.path.dirname(test_anomaly_path))

            _write_wav_pcm16(train_path, train_audio, sample_rate)
            _write_wav_pcm16(test_normal_path, test_normal_audio, sample_rate)
            _write_wav_pcm16(test_anomaly_path, test_anomaly_audio, sample_rate)
            generated_files += 3

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DCASE-style synthetic audio based on real train WAV files from an input root."
    )
    parser.add_argument(
        "--input-root",
        default="",
        help="Root that contains one folder per machine, each with train/ and optionally attributes_00.csv.",
    )
    parser.add_argument(
        "--template-root",
        default="",
        help="Backward-compatible alias for --input-root. If both are set, --input-root wins.",
    )
    parser.add_argument(
        "--output-root",
        default="data/dcase2025t2/dev_data/raw_synth",
        help="Destination root for the generated machine/train/test structure.",
    )
    parser.add_argument(
        "--machines",
        nargs="*",
        default=[],
        help="Optional subset of machine folders to process. If empty, all machine folders under the input root are used.",
    )
    parser.add_argument(
        "--copies-per-source",
        type=int,
        default=1,
        help="How many synthetic train/test triplets to create from each source train clip.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synth-noise-snr-db",
        type=float,
        default=None,
        help=(
            "Optioneel. Als gezet (dB): vervang de willekeurige ruisschaal door witte ruismix met dit "
            "vermogens SNR tussen signaal vlak voor de ruis en de toegevoegde ruis. "
            "Hogere dB betekent zachtere ruis. Niet gezet behoudt de oude sigma in [0.002,0.02]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # step 1 find the input folder with real additional dataset wav files
    preferred = (args.input_root or args.template_root or "").strip()
    if preferred:
        input_root = os.path.normpath(preferred)
        if not os.path.isdir(input_root):
            raise FileNotFoundError(f"Input root bestaat niet of is geen map: {input_root}")
        if not _has_machine_layout(input_root):
            raise FileNotFoundError(
                f"Geen machine-mappen met train/ gevonden onder: {input_root}. "
                "Controleer --input-root."
            )
    else:
        input_root = _discover_input_root(None)
        if not input_root:
            raise FileNotFoundError(
                "Geen input gevonden. Geef --input-root mee, of zet DCASE_INPUT_ROOT, "
                "of werk vanuit een map met data/dcase2025t2/dev_data/raw."
            )

    output_root = os.path.normpath(args.output_root)
    _safe_makedirs(output_root)

    # step 2 save a small json file so we remember how synth was built
    meta = {
        "input_root": input_root,
        "output_root": output_root,
        "copies_per_source": args.copies_per_source,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "synth_noise_snr_db": args.synth_noise_snr_db,
        "noise_mode": "fixed_power_snr" if args.synth_noise_snr_db is not None else "legacy_uniform_sigma",
    }
    manifest_path = os.path.join(output_root, "synth_generation_meta.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    machine_roots = _find_machine_roots(input_root, args.machines or None)
    if not machine_roots:
        raise ValueError(
            f"Geen machine folders gevonden onder: {input_root}. "
            "Controleer dat het pad machine-subfolders bevat met een train/ map."
        )

    # step 3 loop over each machine type and build synthetic copies
    total_files = 0
    for machine, machine_root in machine_roots:
        # step 3a read real train clips from disk for this machine
        source_clips, attr_cols = _load_source_clips(machine_root)
        if not source_clips:
            print(f"[{machine}] skip: geen train wavs gevonden in {machine_root}")
            continue

        rows = _build_output_rows(machine, source_clips, args.copies_per_source)
        generated_files = _synthesize_machine_outputs(
            output_root=output_root,
            machine=machine,
            source_clips=source_clips,
            copies_per_source=args.copies_per_source,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            rng=rng,
            synth_noise_snr_db=args.synth_noise_snr_db,
        )

        df = pd.DataFrame(rows)
        ordered_cols = ["file_name"] + [c for c in attr_cols if c in df.columns]
        df = df.reindex(columns=ordered_cols).sort_values("file_name").reset_index(drop=True)

        machine_output_root = os.path.join(output_root, machine)
        _safe_makedirs(machine_output_root)
        csv_out = os.path.join(machine_output_root, "attributes_00.csv")
        df.to_csv(csv_out, index=False)

        total_files += generated_files
        print(
            f"[{machine}] wrote {generated_files} wav files derived from {len(source_clips)} source train clips and {csv_out}"
        )

    print(f"\nKlaar. Totaal {total_files} wav bestanden gegenereerd.")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()