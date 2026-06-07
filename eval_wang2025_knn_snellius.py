import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from peft import LoraConfig, get_peft_model
from scipy import stats
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoModel

from train_EAT_LoRa_snellius import (
    EATAnomalousTrainer,
    _patch_hf_eat_tied_weights,
    _resolve_train_wav_path,
)


def _dev_row_to_label_and_domain(fn: str):
    fn = str(fn).replace("\\", "/")
    if "test_anomaly" in fn:
        y = 1
    elif "test_normal" in fn:
        y = 0
    else:
        return None, None
    domain = "source" if "source" in fn else "target"
    return y, domain


def load_mel_from_wav_path(file_path: str, target_length: int = 1024):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    waveform = waveform - waveform.mean()
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
    if n_frames < target_length:
        mel = torch.nn.functional.pad(
            mel, (0, 0, 0, target_length - n_frames), "constant", 0
        )
    else:
        mel = mel[:target_length, :]

    norm_mean, norm_std = -4.268, 4.569
    mel = (mel - norm_mean) / (norm_std * 2)
    return mel


@torch.no_grad()
def embed_mel_batch(eat_lora, mels_cpu: torch.Tensor, device: torch.device):
    eat_lora.eval()
    x = mels_cpu.to(device, non_blocking=True)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    e = eat_lora(x)
    return torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy()


def knn_min_cosine_distance_scores(bank: np.ndarray, queries: np.ndarray):
    sim = queries @ bank.T
    max_sim = sim.max(axis=1)
    return (1.0 - max_sim).astype(np.float64)


def auc_pauc_all(y_true, y_score, max_fpr: float):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auc = metrics.roc_auc_score(y_true, y_score)
    pauc = metrics.roc_auc_score(y_true, y_score, max_fpr=max_fpr)
    return auc, pauc


def auc_domain_like_evaluator(y_true, y_score, y_domain_int, domain_idx: int):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_domain_int = np.asarray(y_domain_int, dtype=int)
    mask = (y_domain_int == domain_idx) | (y_true != 0)
    yt = y_true[mask]
    ys = y_score[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return metrics.roc_auc_score(yt, ys)


def wav_rows_train_normal(machine: str, base_path: str) -> pd.DataFrame:
    train_dir = os.path.join(base_path, machine, "train")
    if not os.path.isdir(train_dir):
        return pd.DataFrame()

    rows = []
    for wav_name in sorted(os.listdir(train_dir)):
        if not wav_name.lower().endswith(".wav"):
            continue
        low = wav_name.lower()
        if "anomaly" in low:
            continue
        if "train" not in low or "normal" not in low:
            continue
        rel = f"{machine}/train/{wav_name}"
        rows.append({"file_name": rel, "_machine": machine})
    return pd.DataFrame(rows)


def wav_rows_test_labeled(machine: str, base_path: str) -> pd.DataFrame:
    test_dir = os.path.join(base_path, machine, "test")
    if not os.path.isdir(test_dir):
        return pd.DataFrame()

    rows = []
    for wav_name in sorted(os.listdir(test_dir)):
        if not wav_name.lower().endswith(".wav"):
            continue
        rel = f"{machine}/test/{wav_name}"
        if _dev_row_to_label_and_domain(rel)[0] is None:
            continue
        rows.append({"file_name": rel, "_machine": machine})
    return pd.DataFrame(rows)


def _load_eat_lora_from_checkpoint(
    checkpoint_path: str,
    model_id: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
):
    _patch_hf_eat_tied_weights()
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    base_model = get_peft_model(base_model, config)

    eat_lora = EATAnomalousTrainer(base_model).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("model_state_dict ontbreekt in checkpoint")

    eat_lora.load_state_dict(ckpt["model_state_dict"], strict=True)
    eat_lora.eval()

    step = int(ckpt.get("step", -1))
    print(f"Loaded checkpoint: {checkpoint_path}")
    if step >= 0:
        print(f"Checkpoint step: {step}")

    return eat_lora


def _embed_train_bank(
    eat_lora,
    train_df: pd.DataFrame,
    base_path: str,
    device: torch.device,
    embed_batch_size: int,
    machine: str,
):
    bank_emb = []
    pending_mels = []
    skipped_tr = 0

    iterator = tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc=f"{machine} train",
        leave=False,
    )

    for _, row in iterator:
        fp = _resolve_train_wav_path(base_path, row)
        if not os.path.isfile(fp):
            skipped_tr += 1
            continue
        try:
            pending_mels.append(load_mel_from_wav_path(fp))
        except Exception as ex:
            skipped_tr += 1
            print(f"[{machine}] train wav fout {fp}: {ex}")
            continue

        if len(pending_mels) >= embed_batch_size:
            batch = torch.stack(pending_mels, dim=0)
            bank_emb.append(embed_mel_batch(eat_lora, batch, device))
            pending_mels = []

    if pending_mels:
        batch = torch.stack(pending_mels, dim=0)
        bank_emb.append(embed_mel_batch(eat_lora, batch, device))

    if not bank_emb:
        return None, skipped_tr

    return np.concatenate(bank_emb, axis=0), skipped_tr


def _embed_test_queries(
    eat_lora,
    test_df: pd.DataFrame,
    base_path: str,
    device: torch.device,
    embed_batch_size: int,
    machine: str,
):
    q_emb = []
    pending_mels = []
    pending_labels = []
    pending_domains = []

    y_list = []
    dom_list = []
    skipped_te = 0

    iterator = tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc=f"{machine} test",
        leave=False,
    )

    for _, row in iterator:
        fp = _resolve_train_wav_path(base_path, row)
        lab, dom = _dev_row_to_label_and_domain(row["file_name"])

        if lab is None:
            continue
        if not os.path.isfile(fp):
            skipped_te += 1
            continue

        try:
            pending_mels.append(load_mel_from_wav_path(fp))
            pending_labels.append(int(lab))
            pending_domains.append(0 if dom == "source" else 1)
        except Exception as ex:
            skipped_te += 1
            print(f"[{machine}] test wav fout {fp}: {ex}")
            continue

        if len(pending_mels) >= embed_batch_size:
            batch = torch.stack(pending_mels, dim=0)
            q_emb.append(embed_mel_batch(eat_lora, batch, device))
            y_list.extend(pending_labels)
            dom_list.extend(pending_domains)
            pending_mels, pending_labels, pending_domains = [], [], []

    if pending_mels:
        batch = torch.stack(pending_mels, dim=0)
        q_emb.append(embed_mel_batch(eat_lora, batch, device))
        y_list.extend(pending_labels)
        dom_list.extend(pending_domains)

    if not q_emb:
        return None, None, None, skipped_te

    q_np = np.concatenate(q_emb, axis=0)
    y_true = np.array(y_list, dtype=int)
    y_domain_int = np.array(dom_list, dtype=int)
    return q_np, y_true, y_domain_int, skipped_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dcase2025t2/dev_data/raw")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="worstchan/EAT-base_epoch30_finetune_AS2M")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--output-csv", default="results/knn_eval_summary.csv")
    args = parser.parse_args()

    base_path = os.path.normpath(args.data_root)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"data-root bestaat niet: {base_path}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint bestaat niet: {args.checkpoint}")

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eat_lora = _load_eat_lora_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    machine_dirs = sorted(
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    )
    if args.machines:
        allowed = set(args.machines)
        machine_dirs = [m for m in machine_dirs if m in allowed]

    if not machine_dirs:
        raise ValueError("Geen machine directories gevonden om te evalueren")

    rows_out = []

    for machine in machine_dirs:
        train_df = wav_rows_train_normal(machine, base_path)
        test_df = wav_rows_test_labeled(machine, base_path)

        if len(train_df) == 0:
            print(f"[{machine}] skip: geen train/*.wav (train+normal in naam, geen anomaly)")
            continue
        if len(test_df) == 0:
            print(f"[{machine}] skip: geen test/*.wav met test_normal/test_anomaly in naam")
            continue

        bank_np, skipped_tr = _embed_train_bank(
            eat_lora=eat_lora,
            train_df=train_df,
            base_path=base_path,
            device=device,
            embed_batch_size=args.embed_batch_size,
            machine=machine,
        )
        if bank_np is None:
            print(f"[{machine}] skip: geen train wav geladen (overgeslagen {skipped_tr})")
            continue

        q_np, y_true, y_domain_int, skipped_te = _embed_test_queries(
            eat_lora=eat_lora,
            test_df=test_df,
            base_path=base_path,
            device=device,
            embed_batch_size=args.embed_batch_size,
            machine=machine,
        )
        if q_np is None:
            print(f"[{machine}] skip: geen test wav (overgeslagen {skipped_te})")
            continue

        y_score = knn_min_cosine_distance_scores(bank_np, q_np)

        auc_all, pauc = auc_pauc_all(y_true, y_score, max_fpr=args.max_fpr)
        auc_src = auc_domain_like_evaluator(y_true, y_score, y_domain_int, 0)
        auc_tgt = auc_domain_like_evaluator(y_true, y_score, y_domain_int, 1)

        rows_out.append(
            {
                "machine": machine,
                "n_bank": len(bank_np),
                "n_test": len(y_true),
                "skipped_train_wav": skipped_tr,
                "skipped_test_wav": skipped_te,
                "AUC_all": auc_all,
                "pAUC": pauc,
                "AUC_source": auc_src,
                "AUC_target": auc_tgt,
            }
        )

        print(
            f"{machine}: AUC_all={auc_all:.4f} pAUC={pauc:.4f} "
            f"AUC_src={auc_src:.4f} AUC_tgt={auc_tgt:.4f}  "
            f"bank={len(bank_np)} test={len(y_true)}"
        )

    if not rows_out:
        print("Geen machines geevalueerd. Controleer BASE_PATH en of .wav bestanden bestaan.")
        print("BASE_PATH gebruikt:", os.path.abspath(base_path))
        return

    summary = pd.DataFrame(rows_out).set_index("machine")
    print("\nSummary per machine:")
    print(summary)

    summary.to_csv(args.output_csv)
    print(f"\nSaved summary csv to: {args.output_csv}")

    valid = summary["AUC_all"].notna()
    if valid.any():
        print(
            "\nArithmetic mean over machines - AUC_all:",
            float(summary.loc[valid, "AUC_all"].mean()),
            " pAUC:",
            float(summary.loc[valid, "pAUC"].mean()),
            " AUC_source:",
            float(summary.loc[valid, "AUC_source"].mean()),
            " AUC_target:",
            float(summary.loc[valid, "AUC_target"].mean()),
        )

        flat = []
        for m in summary.loc[valid].index:
            r = summary.loc[m]
            flat.extend([r["AUC_source"], r["AUC_target"], r["pAUC"]])

        flat = np.maximum(np.array(flat, dtype=np.float64), sys.float_info.epsilon)
        print(
            "Harmonic mean (AUC_src, AUC_tgt, pAUC per machine):",
            float(stats.hmean(flat)),
        )


if __name__ == "__main__":
    main()
