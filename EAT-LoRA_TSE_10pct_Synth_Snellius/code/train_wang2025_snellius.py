"""
this script fine tunes the eat audio model with lora layers for dcase anomaly detection
step one scan disk for train wav files and match them to labels from attributes csv
step two build integer class labels from machine name domain and attribute columns
step three load each wav as mono 16 khz and convert it to a mel spectrogram
step four pass the mel through the eat backbone to get an embedding vector
step five use arcface loss to pull embeddings toward the correct machine class
step six update only lora weights and save checkpoints every few thousand steps
training runs for a fixed 6000 steps
you can mix two data folders for example ten percent synth and ninety percent raw
"""

import argparse
import csv
import os
import random
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from audio_io import load_wav_mono
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel
import transformers.modeling_utils as _hf_modeling_utils

try:
    from tse_snellius import load_tse_models
except Exception:
    load_tse_models = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None
    _TB_AVAILABLE = False

# fixed number of training steps for eat lora training
EAT_TRAIN_NUM_STEPS = 6000


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


def _normalize_attr_value(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        x = float(s.replace(",", "."))
        if x == int(x):
            return str(int(x))
    except ValueError:
        pass
    return s


def _iter_machine_attribute_csvs(base_path):
    if not os.path.isdir(base_path):
        return
    for name in sorted(os.listdir(base_path)):
        p = os.path.join(base_path, name)
        if not os.path.isdir(p):
            continue
        ap = os.path.join(p, "attributes_00.csv")
        if os.path.isfile(ap):
            yield name, ap


_CORE_KEY_RE = re.compile(
    r"(section_\d+_(?:source|target)_(?:train|test)_(?:normal|anomaly)_\d+)",
    re.IGNORECASE,
)


def _core_key(stem):
    """Gedeelde sleutel tussen schijf-naam en CSV file_name (negeert attribuut-suffix)."""
    m = _CORE_KEY_RE.search(stem)
    return m.group(1).lower() if m else stem.lower()


def _machine_attr_lookup(csv_path):
    """core_key -> attribuutwaarden uit attributes_00.csv (alleen train-regels)."""
    lookup = {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return lookup
    if "file_name" not in df.columns:
        return lookup
    attr_cols = [c for c in df.columns if c.endswith("v")]
    train_df = df[df["file_name"].astype(str).str.contains("train", na=False)]
    for _, row in train_df.iterrows():
        stem = os.path.splitext(os.path.basename(str(row["file_name"]).replace("\\", "/")))[0]
        vals = [_normalize_attr_value(row[c]) for c in attr_cols if pd.notna(row[c])]
        vals = [v for v in vals if v and v.lower() != "noattribute"]
        lookup[_core_key(stem)] = vals
    return lookup


def _iter_disk_train_records(base_path):
    # list train wav files from disk and attach attribute labels from the csv when possible
    # this works even when csv file names differ from the short names on disk
    if not os.path.isdir(base_path):
        return
    for machine in sorted(os.listdir(base_path)):
        machine_dir = os.path.join(base_path, machine)
        train_dir = os.path.join(machine_dir, "train")
        if not os.path.isdir(train_dir):
            continue
        csv_path = os.path.join(machine_dir, "attributes_00.csv")
        attr_lookup = _machine_attr_lookup(csv_path) if os.path.isfile(csv_path) else {}
        for wav_name in sorted(os.listdir(train_dir)):
            if not wav_name.lower().endswith(".wav"):
                continue
            stem = os.path.splitext(wav_name)[0]
            if "anomaly" in stem.lower():
                continue
            domain = "source" if "source" in stem.lower() else "target"
            attr_values = attr_lookup.get(_core_key(stem), [])
            yield {
                "machine": machine,
                "domain": domain,
                "attr_values": list(attr_values),
                "file_name": f"{machine}/train/{wav_name}",
            }


def _config_label(machine, domain, attr_values):
    if attr_values:
        return f"{machine}_{domain}_" + "_".join(attr_values)
    return f"{machine}_{domain}"


def get_dcase_num_classes(base_path):
    all_configs = [
        _config_label(r["machine"], r["domain"], r["attr_values"])
        for r in _iter_disk_train_records(base_path)
    ]
    unique_classes = sorted(set(all_configs))
    return unique_classes, len(unique_classes)


class DCASELabelEncoder:
    def __init__(self, unique_configs):
        self.str_to_int = {label: i for i, label in enumerate(unique_configs)}
        self.int_to_str = {i: label for i, label in enumerate(unique_configs)}
        self.num_classes = len(unique_configs)
        self._machine_canonical = {}
        for label in unique_configs:
            parts = label.split("_")
            if len(parts) >= 2 and parts[1] in ("source", "target"):
                m = parts[0]
                self._machine_canonical.setdefault(m.lower(), m)

    def encode(self, machine, domain, attributes=None):
        machine = str(machine).strip()
        machine = self._machine_canonical.get(machine.lower(), machine)
        domain = domain.strip().lower()

        if not attributes or attributes == ["noAttribute"]:
            label_str = f"{machine}_{domain}"
        else:
            clean_attrs = [_normalize_attr_value(a) for a in attributes]
            clean_attrs = [a for a in clean_attrs if a and a.lower() != "noattribute"]
            attrs_str = "_".join(clean_attrs)
            label_str = f"{machine}_{domain}_{attrs_str}"

        return self.str_to_int.get(label_str, -1)


def _resolve_train_wav_path(base_path, row):
    fn = str(row["file_name"]).replace("\\\\", "/").replace("\\", "/")
    if "/" in fn:
        return os.path.normpath(os.path.join(base_path, *fn.split("/")))
    machine = row.get("_machine")
    if machine is None or (isinstance(machine, float) and pd.isna(machine)):
        raise ValueError(
            f"file_name zonder map maar geen _machine in rij: {fn!r}. draai create_master_dataframe opnieuw."
        )
    stem = os.path.basename(fn)
    if not stem.lower().endswith(".wav"):
        stem = stem + ".wav"
    return os.path.normpath(os.path.join(base_path, str(machine), "train", stem))


def _waveform_mono_16k_zero_mean(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Match eval: mono, 16 khz, dc offset removed. shape [1, T]."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform - waveform.mean()


class EATDCASEDataset(Dataset):
    def __init__(
        self,
        df,
        encoder,
        base_path,
        target_length=1024,
        tse_models=None,
    ):
        self.df = df
        self.encoder = encoder
        self.base_path = base_path
        self.target_length = target_length
        self.tse_models = tse_models if tse_models else None
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = _resolve_train_wav_path(self.base_path, row)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"wav ontbreekt: {file_path}")

        machine, domain, attr_values = _row_label_fields(self.df, row)

        target_id = self.encoder.encode(machine, domain, attr_values)

        # step one load wav as mono 16 khz and remove dc offset
        waveform, sr = load_wav_mono(file_path, target_sr=16000)
        waveform = _waveform_mono_16k_zero_mean(waveform, sr)

        tse_net = None
        if self.tse_models:
            tse_net = self.tse_models.get(machine)

        if tse_net is not None:
            with torch.no_grad():
                wav = waveform.to(dtype=torch.float32)
                enhanced = tse_net(wav)
                if enhanced.dim() == 1:
                    waveform = enhanced.unsqueeze(0)
                else:
                    waveform = enhanced

        # step two turn waveform into a fixed length mel spectrogram
        mel = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )

        n_frames = mel.shape[0]
        if n_frames < self.target_length:
            pad_amount = self.target_length - n_frames
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_amount), "constant", 0)
        else:
            mel = mel[: self.target_length, :]

        # step three normalize mel the same way as during evaluation
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        return mel, torch.tensor(target_id).long()


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, label)


class EATAnomalousTrainer(nn.Module):
    def __init__(self, eat_backbone):
        super().__init__()
        self.backbone = eat_backbone

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        base = (
            self.backbone.get_base_model()
            if hasattr(self.backbone, "get_base_model")
            else self.backbone
        )
        if hasattr(base, "extract_features"):
            feat_seq = base.extract_features(x)
            cls_vec = feat_seq[:, 0]
            inner = getattr(base, "model", base)
            if hasattr(inner, "fc_norm"):
                return inner.fc_norm(cls_vec)
            return cls_vec
        outputs = self.backbone(x)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def _merged_labels_for_two_roots(base_path_a: str, base_path_b: str):
    """Maak één gesorteerde lijst ArcFace labels uit twee DCASE bomen. Synth en raw delen dezelfde namen."""
    la, _ = get_dcase_num_classes(base_path_a)
    lb, _ = get_dcase_num_classes(base_path_b)
    return sorted(set(la) | set(lb))


def _endless_batches(loader):
    """Herhaal batches oneindig voor gemengde training uit twee loaders."""
    while True:
        for batch in loader:
            yield batch


def create_master_dataframe(base_path):
    records = list(_iter_disk_train_records(base_path))
    if not records:
        raise FileNotFoundError(f"geen train-wavs gevonden onder {base_path}")
    return pd.DataFrame(
        {
            "file_name": [r["file_name"] for r in records],
            "_machine": [r["machine"] for r in records],
            "_domain": [r["domain"] for r in records],
            "_attr_values": [r["attr_values"] for r in records],
        }
    )


def _row_label_fields(df, row):
    """Bepaal (machine, domain, attr_values) uit een rij, ongeacht bron-formaat."""
    fn = str(row["file_name"]).replace("\\\\", "/").replace("\\", "/")
    if "_machine" in df.columns and pd.notna(row.get("_machine")):
        machine = str(row["_machine"]).strip()
    elif "/" in fn:
        machine = fn.split("/")[0]
    else:
        machine = str(row.get("_machine", "")).strip()

    if "_domain" in df.columns and pd.notna(row.get("_domain")):
        domain = str(row["_domain"]).strip()
    else:
        domain = "source" if "source" in fn else "target"

    if "_attr_values" in df.columns:
        av = row["_attr_values"]
        attr_values = list(av) if isinstance(av, (list, tuple)) else []
    else:
        attr_cols = [col for col in df.columns if col.endswith("v")]
        attr_values = [_normalize_attr_value(row[col]) for col in attr_cols if pd.notna(row[col])]
        attr_values = [v for v in attr_values if v and v.lower() != "noattribute"]
    return machine, domain, attr_values


def _validate_training_df(df, encoder, base_path, label):
    """Fail fast with a clear message before DataLoader workers start."""
    missing_paths = []
    bad_labels = 0
    for _, row in df.iterrows():
        file_path = _resolve_train_wav_path(base_path, row)
        if not os.path.isfile(file_path):
            missing_paths.append(file_path)
            continue
        machine, domain, attr_values = _row_label_fields(df, row)
        if encoder.encode(machine, domain, attr_values) < 0:
            bad_labels += 1

    if missing_paths:
        preview = "\n  ".join(missing_paths[:8])
        extra = f"\n  ... en {len(missing_paths) - 8} meer" if len(missing_paths) > 8 else ""
        raise FileNotFoundError(
            f"[{label}] {len(missing_paths)} train-wav(s) ontbreken onder {base_path}.\n  {preview}{extra}"
        )
    if bad_labels:
        raise ValueError(
            f"[{label}] {bad_labels} train-regels hebben geen geldig ArcFace-label. "
            "Controleer attributes_00.csv en label-merge tussen synth/raw."
        )
    print(f"[{label}] dataset OK: {len(df)} train-regels, alle wavs aanwezig", flush=True)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint(path, eat_lora, criterion, optimizer, scheduler, device, labels_list):
    if not path:
        return 0
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint niet gevonden: {path}")
    ckpt = torch.load(path, map_location="cpu")
    saved_labels = ckpt.get("labels_list")
    if saved_labels is not None and list(saved_labels) != list(labels_list):
        print(
            f"Checkpoint {path} heeft andere labels ({len(saved_labels)} vs {len(labels_list)}); "
            "start opnieuw vanaf step 0",
            flush=True,
        )
        return 0
    saved_arcface = ckpt.get("arcface_state_dict", {})
    saved_weight = saved_arcface.get("weight")
    if saved_weight is not None and saved_weight.shape[0] != criterion.weight.shape[0]:
        print(
            f"Checkpoint {path} past niet bij huidige ArcFace ({saved_weight.shape[0]} vs "
            f"{criterion.weight.shape[0]} klassen); start opnieuw vanaf step 0",
            flush=True,
        )
        return 0
    try:
        eat_lora.load_state_dict(ckpt["model_state_dict"])
        criterion.load_state_dict(ckpt["arcface_state_dict"])
    except RuntimeError as exc:
        print(
            f"Checkpoint {path} niet compatible ({exc}); start opnieuw vanaf step 0",
            flush=True,
        )
        return 0
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = int(ckpt.get("step", 0))
    print(f"Checkpoint geladen: {path} (step {step})", flush=True)
    return step


def save_checkpoint(path, eat_lora, criterion, optimizer, scheduler, step, labels_list):
    torch.save(
        {
            "model_state_dict": eat_lora.state_dict(),
            "arcface_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "labels_list": labels_list,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dcase2025t2/dev_data/raw")
    parser.add_argument("--save-dir", default="checkpoints/eat_lora_system1")
    parser.add_argument("--model-id", default="worstchan/EAT-base_epoch30_finetune_AS2M")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-step", type=int, default=5000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--loss-csv",
        default="",
        help="Pad naar CSV waar (step,loss,lr) wordt weggeschreven. Leeg = geen csv.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default="",
        help="TensorBoard logdir. Leeg = TensorBoard uit.",
    )
    parser.add_argument(
        "--train-tse-checkpoint-dir",
        default="",
        help=(
            "Map met TSE checkpoints ({machine}.pt). Indien gezet: zelfde keten als eval, "
            "waveform naar TSE naar mel. Machines zonder checkpoint vallen terug op ruwe waveform."
        ),
    )
    parser.add_argument(
        "--mix-second-data-root",
        default="",
        help=(
            "Tweede DCASE boom naast data root. Per stap kiest het script willekeurig of de batch "
            "uit de eerste of de tweede boom komt. Zie mix-primary-fraction voor de verhouding. "
            "kNN blijft apart op EVAL_DATA_ROOT draaien."
        ),
    )
    parser.add_argument(
        "--mix-primary-fraction",
        type=float,
        default=0.1,
        help=(
            "Alleen bij mix-second-data-root. Kans dat een stap de eerste boom (data-root) gebruikt. "
            "Default 0,1 geeft ca. 10 procent synth en 90 procent tweede boom als primary synth is."
        ),
    )
    args = parser.parse_args()
    args.num_steps = EAT_TRAIN_NUM_STEPS

    # step one read command line paths and validate the data folders exist
    mix_primary_fraction = float(args.mix_primary_fraction)
    if not (0.0 <= mix_primary_fraction <= 1.0):
        raise ValueError("mix-primary-fraction moet tussen 0 en 1 liggen")

    data_root = os.path.normpath(args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data-root bestaat niet: {data_root}")

    mix_second = (args.mix_second_data_root or "").strip()
    if mix_second:
        mix_second = os.path.normpath(mix_second)
        if not os.path.isdir(mix_second):
            raise FileNotFoundError(f"mix-second-data-root bestaat niet: {mix_second}")
        if os.path.normpath(mix_second) == data_root:
            raise ValueError("mix-second-data-root moet een andere map zijn dan data root")
        print(
            "EAT mix training. Eerste boom is",
            data_root,
            "tweede boom is",
            mix_second,
            "mix_primary_fraction",
            mix_primary_fraction,
        )

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if mix_second:
        labels_list = _merged_labels_for_two_roots(data_root, mix_second)
    else:
        labels_list, _ = get_dcase_num_classes(data_root)
    encoder = DCASELabelEncoder(labels_list)
    master_df = create_master_dataframe(data_root)
    _validate_training_df(master_df, encoder, data_root, "primary")

    tse_train_models = None
    tse_dir = (args.train_tse_checkpoint_dir or "").strip()
    if tse_dir:
        if load_tse_models is None:
            raise RuntimeError("tse_snellius ontbreekt, kan train-tse-checkpoint-dir niet gebruiken")
        if not os.path.isdir(tse_dir):
            raise FileNotFoundError(f"train-tse-checkpoint-dir bestaat niet: {tse_dir}")
        tse_train_models = load_tse_models(tse_dir, torch.device("cpu"))
        if not tse_train_models:
            print(f"waarschuwing: geen .pt in {tse_dir}, EAT traint zonder TSE-voorbewerking")
            tse_train_models = None
        else:
            print(f"EAT training met TSE op cpu voor {len(tse_train_models)} machines (zelfde volgorde als eval)")

    train_dataset = EATDCASEDataset(
        df=master_df,
        encoder=encoder,
        base_path=data_root,
        tse_models=tse_train_models,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    mix_loader = None
    mix_iter = None
    if mix_second:
        master_df_b = create_master_dataframe(mix_second)
        _validate_training_df(master_df_b, encoder, mix_second, "mix-secondary")
        train_dataset_b = EATDCASEDataset(
            df=master_df_b,
            encoder=encoder,
            base_path=mix_second,
            tse_models=tse_train_models,
        )
        mix_loader = DataLoader(
            train_dataset_b,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )
        mix_iter = _endless_batches(mix_loader)

    _patch_hf_eat_tied_weights()
    # step two load pretrained eat from hugging face and attach lora adapters
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).eval()

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"ArcFace klassen: {encoder.num_classes}, batch: {args.batch_size}, "
        f"workers: {args.num_workers}",
        flush=True,
    )

    eat_lora = EATAnomalousTrainer(model).to(device)
    criterion = ArcFaceLoss(in_features=768, out_features=encoder.num_classes).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": eat_lora.parameters()},
            {"params": criterion.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    resume_path = args.resume
    if args.auto_resume and not resume_path:
        candidate = os.path.join(args.save_dir, "checkpoint_last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate

    current_step = load_checkpoint(
        resume_path, eat_lora, criterion, optimizer, scheduler, device, labels_list
    )
    if current_step > 0:
        print(f"Hervatten vanaf step {current_step} ({resume_path})", flush=True)
    elif resume_path:
        print(f"Geen geldig checkpoint geladen van {resume_path}; start vanaf step 0", flush=True)

    csv_writer = None
    csv_file = None
    if args.loss_csv:
        os.makedirs(os.path.dirname(args.loss_csv) or ".", exist_ok=True)
        write_header = not os.path.isfile(args.loss_csv) or current_step == 0
        csv_file = open(args.loss_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["step", "loss", "lr"])
            csv_file.flush()

    tb_writer = None
    if args.tensorboard_dir:
        if not _TB_AVAILABLE:
            print("TensorBoard niet beschikbaar, sla logdir over")
        else:
            os.makedirs(args.tensorboard_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    running_loss = 0.0
    pbar = tqdm(total=args.num_steps, initial=current_step, desc="Training System-1")

    def one_opt_step(mel_tensor, labels_tensor):
        nonlocal running_loss, current_step
        mel_tensor = mel_tensor.to(device, non_blocking=True)
        labels_tensor = labels_tensor.to(device, non_blocking=True)
        embeddings = eat_lora(mel_tensor)
        loss = criterion(embeddings, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        current_step += 1
        pbar.update(1)
        if current_step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
            if csv_writer is not None:
                csv_writer.writerow([current_step, f"{avg_loss:.6f}", f"{current_lr:.6e}"])
                csv_file.flush()
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, current_step)
                tb_writer.add_scalar("train/lr", current_lr, current_step)
            running_loss = 0.0
        if current_step % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.save_dir, f"checkpoint_step_{current_step}.pt"),
                eat_lora,
                criterion,
                optimizer,
                scheduler,
                current_step,
                labels_list,
            )

    if mix_iter is not None:
        primary_iter = _endless_batches(loader)
        while current_step < args.num_steps:
            eat_lora.train()
            if random.random() < mix_primary_fraction:
                mel, labels = next(primary_iter)
            else:
                mel, labels = next(mix_iter)
            one_opt_step(mel, labels)
    else:
        while current_step < args.num_steps:
            eat_lora.train()
            for mel, labels in loader:
                if current_step >= args.num_steps:
                    break
                one_opt_step(mel, labels)

    pbar.close()
    if csv_file is not None:
        csv_file.close()
    if tb_writer is not None:
        tb_writer.close()

    save_checkpoint(
        os.path.join(args.save_dir, "checkpoint_last.pt"),
        eat_lora,
        criterion,
        optimizer,
        scheduler,
        current_step,
        labels_list,
    )
    print("Training klaar")


if __name__ == "__main__":
    main()
