"""DCASE 2025 Task 2 inference op de Evaluation dataset (officiele submission-CSV's).

Hoort bij de pijplijn die EAT-LoRA + TSE op de **Additional dataset** traint en daarna
op de **Evaluation dataset** scoort, zodat de officiele ``dcase2025_task2_evaluator.py``
een ranking-score kan berekenen.

Verschil met ``eval_wang2025_knn_snellius.py``:
  * de KNN-bank komt uit de Additional-dataset train-clips (``--train-data-root``);
  * de queries zijn de Evaluation test-clips (``--eval-data-root``), die GEEN
    normal/anomaly- of domain-label in de bestandsnaam hebben;
  * we berekenen hier zelf geen AUC (de labels zijn verborgen). In plaats daarvan
    schrijven we per machine/sectie twee CSV's in het DCASE-submissieformaat:
        anomaly_score_<machine>_<section>_test.csv   (filename,score)
        decision_result_<machine>_<section>_test.csv (filename,0|1)
    De drempel voor de 0/1-beslissing volgt de DCASE-baseline: fit een
    gamma-verdeling op de train-anomaliescores en neem het 90e percentiel.

De output gaat naar ``--out-dir`` (bijv. ``.../dcase2025_task2_evaluator/teams/<team>/<system>``)
zodat de evaluator hem met ``glob(teams/*/*)`` (dir_depth=2) vindt.

TSE-modus (``--tse-mode``) is identiek aan de KNN-eval: ``both`` (bank + test enhanced,
cluster-default), ``test-only`` of ``off``.

Zie: Fujimura, T., Kuroyanagi, I., and Toda, T. (2025). The NU systems for DCASE 2025
Challenge Task 2. Technical report, Nagoya University.
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from train_wang2025_snellius import _resolve_train_wav_path
from eval_wang2025_knn_snellius import (
    _embed_train_bank,
    _load_eat_lora_from_checkpoint,
    embed_mel_batch,
    knn_min_cosine_distance_scores,
    load_mel_from_wav_path,
    wav_rows_train_normal,
)

try:
    from tse_snellius import load_tse_models
    _TSE_AVAILABLE = True
except Exception:
    load_tse_models = None
    _TSE_AVAILABLE = False


_EVAL_NAME_RE = re.compile(r"(section_\d+)_\d+\.wav$", re.IGNORECASE)


def list_eval_test_wavs(machine: str, base_path: str) -> pd.DataFrame:
    """List Evaluation test wavs for one machine: section_<NN>_<idx>.wav (no labels)."""
    test_dir = os.path.join(base_path, machine, "test")
    if not os.path.isdir(test_dir):
        return pd.DataFrame()

    rows = []
    for wav_name in sorted(os.listdir(test_dir)):
        if not wav_name.lower().endswith(".wav"):
            continue
        m = _EVAL_NAME_RE.search(wav_name)
        section = m.group(1) if m else "section_00"
        rows.append(
            {
                "file_name": f"{machine}/test/{wav_name}",
                "wav_name": wav_name,
                "section": section,
                "_machine": machine,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def embed_eval_queries(
    eat_lora,
    test_df: pd.DataFrame,
    base_path: str,
    device: torch.device,
    embed_batch_size: int,
    machine: str,
    tse_model=None,
):
    """Embed all eval test clips, keeping wav_name + section aligned with the embeddings."""
    q_emb = []
    out_names = []
    out_sections = []

    pending_mels = []
    pending_names = []
    pending_sections = []
    skipped = 0

    iterator = tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc=f"{machine} eval-test",
        leave=False,
    )

    for _, row in iterator:
        fp = _resolve_train_wav_path(base_path, row)
        if not os.path.isfile(fp):
            skipped += 1
            continue
        try:
            pending_mels.append(
                load_mel_from_wav_path(fp, tse_model=tse_model, tse_device=device)
            )
            pending_names.append(row["wav_name"])
            pending_sections.append(row["section"])
        except Exception as ex:
            skipped += 1
            print(f"[{machine}] eval wav fout {fp}: {ex}")
            continue

        if len(pending_mels) >= embed_batch_size:
            batch = torch.stack(pending_mels, dim=0)
            q_emb.append(embed_mel_batch(eat_lora, batch, device))
            out_names.extend(pending_names)
            out_sections.extend(pending_sections)
            pending_mels, pending_names, pending_sections = [], [], []

    if pending_mels:
        batch = torch.stack(pending_mels, dim=0)
        q_emb.append(embed_mel_batch(eat_lora, batch, device))
        out_names.extend(pending_names)
        out_sections.extend(pending_sections)

    if not q_emb:
        return None, [], [], skipped

    return np.concatenate(q_emb, axis=0), out_names, out_sections, skipped


def bank_self_scores(bank: np.ndarray) -> np.ndarray:
    """KNN anomaly score of every bank vector vs the rest of the bank (self excluded)."""
    sim = bank @ bank.T
    np.fill_diagonal(sim, -np.inf)
    max_sim = sim.max(axis=1)
    return (1.0 - max_sim).astype(np.float64)


def gamma_threshold(train_scores: np.ndarray, percentile: float = 0.9) -> float:
    """DCASE-baseline style threshold: gamma fit on train scores, take the p-th percentile."""
    scores = np.asarray(train_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("inf")

    # gamma support is (0, inf); shift away from 0 so the fit is well behaved.
    shift = 0.0
    smin = float(scores.min())
    if smin <= 0:
        shift = -smin + 1e-8
    shifted = scores + shift

    try:
        shape, loc, scale = stats.gamma.fit(shifted, floc=0)
        thr = float(stats.gamma.ppf(percentile, shape, loc=loc, scale=scale)) - shift
        if not np.isfinite(thr):
            raise ValueError("non-finite gamma threshold")
    except Exception:
        thr = float(np.quantile(scores, percentile))
    return thr


def write_score_csv(path: str, names, values, decision: bool = False):
    """Write a headerless DCASE CSV: filename,value (value = score or 0/1 decision)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for name, val in zip(names, values):
            if decision:
                writer.writerow([name, int(val)])
            else:
                writer.writerow([name, f"{float(val):.10f}"])


def plot_score_distribution(
    machine: str,
    scores: np.ndarray,
    threshold: float,
    out_png: str,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(
        threshold,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"threshold (gamma p90) = {threshold:.4f}",
    )
    n_anom = int(np.sum(scores > threshold))
    ax.set_title(f"{machine}: eval anomaly-score distribution ({n_anom}/{len(scores)} > thr)")
    ax.set_xlabel("KNN anomaly score (1 - max cosine similarity)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-root",
        default="data/additional",
        help="Map met <machine>/train/*.wav (Additional dataset) voor de KNN-bank.",
    )
    parser.add_argument(
        "--eval-data-root",
        default="data/evaluation",
        help="Map met <machine>/test/section_*.wav (Evaluation dataset) voor de queries.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="worstchan/EAT-base_epoch30_finetune_AS2M")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument(
        "--out-dir",
        required=True,
        help=(
            "Map waar de DCASE-CSV's komen, bijv. "
            "<evaluator>/teams/<team>/<system>. De evaluator vindt deze via teams/*/*."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        default="",
        help="Map voor anomaly-score histogrammen + summary csv. Leeg = geen plots.",
    )
    parser.add_argument("--decision-percentile", type=float, default=0.9)
    parser.add_argument(
        "--tse-checkpoint-dir",
        default="",
        help="Map met per-machine TSE checkpoints ({machine}.pt). Leeg = geen TSE.",
    )
    parser.add_argument(
        "--tse-mode",
        choices=["off", "test-only", "both"],
        default="off",
        help="off = geen TSE. test-only = TSE alleen op eval test. both = TSE op bank en test.",
    )
    args = parser.parse_args()

    train_root = os.path.normpath(args.train_data_root)
    eval_root = os.path.normpath(args.eval_data_root)
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train-data-root bestaat niet: {train_root}")
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"eval-data-root bestaat niet: {eval_root}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint bestaat niet: {args.checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)

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

    tse_models = {}
    if args.tse_mode != "off" and args.tse_checkpoint_dir:
        if not _TSE_AVAILABLE:
            print("tse_snellius niet importeerbaar, TSE wordt overgeslagen")
        elif not os.path.isdir(args.tse_checkpoint_dir):
            print(f"TSE checkpoint dir bestaat niet: {args.tse_checkpoint_dir}, TSE wordt overgeslagen")
        else:
            tse_models = load_tse_models(args.tse_checkpoint_dir, device)
            print(f"Loaded TSE checkpoints voor {len(tse_models)} machines (mode={args.tse_mode})")

    # Machines: alleen die in beide datasets voorkomen.
    train_machines = {
        d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))
    }
    eval_machines = sorted(
        d for d in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, d))
    )
    machine_dirs = [m for m in eval_machines if m in train_machines]
    if args.machines:
        allowed = set(args.machines)
        machine_dirs = [m for m in machine_dirs if m in allowed]
    if not machine_dirs:
        raise ValueError(
            "Geen machines die in zowel train- als eval-root bestaan. "
            f"train={sorted(train_machines)} eval={eval_machines}"
        )

    summary_rows = []

    for machine in machine_dirs:
        train_df = wav_rows_train_normal(machine, train_root)
        test_df = list_eval_test_wavs(machine, eval_root)

        if len(train_df) == 0:
            print(f"[{machine}] skip: geen train/*.wav (train+normal in naam) onder {train_root}")
            continue
        if len(test_df) == 0:
            print(f"[{machine}] skip: geen test/*.wav onder {eval_root}")
            continue

        tse_model_for_machine = tse_models.get(machine) if tse_models else None
        tse_for_bank = tse_model_for_machine if args.tse_mode == "both" else None
        tse_for_test = (
            tse_model_for_machine if args.tse_mode in ("test-only", "both") else None
        )

        bank_np, skipped_tr = _embed_train_bank(
            eat_lora=eat_lora,
            train_df=train_df,
            base_path=train_root,
            device=device,
            embed_batch_size=args.embed_batch_size,
            machine=machine,
            tse_model=tse_for_bank,
        )
        if bank_np is None:
            print(f"[{machine}] skip: geen train wav geladen (overgeslagen {skipped_tr})")
            continue

        q_np, names, sections, skipped_te = embed_eval_queries(
            eat_lora=eat_lora,
            test_df=test_df,
            base_path=eval_root,
            device=device,
            embed_batch_size=args.embed_batch_size,
            machine=machine,
            tse_model=tse_for_test,
        )
        if q_np is None:
            print(f"[{machine}] skip: geen eval wav geladen (overgeslagen {skipped_te})")
            continue

        y_score = knn_min_cosine_distance_scores(bank_np, q_np)

        # Drempel uit de train-verdeling (DCASE-baseline: gamma, 90e percentiel).
        thr = gamma_threshold(bank_self_scores(bank_np), percentile=args.decision_percentile)
        decisions = (y_score > thr).astype(int)

        names = np.asarray(names)
        sections = np.asarray(sections)

        for section in sorted(set(sections.tolist())):
            mask = sections == section
            sec_names = names[mask]
            sec_scores = y_score[mask]
            sec_dec = decisions[mask]

            anomaly_path = os.path.join(
                args.out_dir, f"anomaly_score_{machine}_{section}_test.csv"
            )
            decision_path = os.path.join(
                args.out_dir, f"decision_result_{machine}_{section}_test.csv"
            )
            write_score_csv(anomaly_path, sec_names, sec_scores, decision=False)
            write_score_csv(decision_path, sec_names, sec_dec, decision=True)
            print(
                f"[{machine}/{section}] wrote {len(sec_names)} scores -> {os.path.basename(anomaly_path)}, "
                f"{int(sec_dec.sum())} flagged anomalous"
            )

        if args.plots_dir:
            plot_score_distribution(
                machine=machine,
                scores=y_score,
                threshold=thr,
                out_png=os.path.join(args.plots_dir, f"anm_score_hist_{machine}.png"),
            )

        summary_rows.append(
            {
                "machine": machine,
                "n_bank": int(len(bank_np)),
                "n_test": int(len(y_score)),
                "skipped_train_wav": int(skipped_tr),
                "skipped_eval_wav": int(skipped_te),
                "threshold": float(thr),
                "n_flagged_anomaly": int(decisions.sum()),
                "mean_score": float(np.mean(y_score)),
                "min_score": float(np.min(y_score)),
                "max_score": float(np.max(y_score)),
            }
        )
        print(
            f"{machine}: bank={len(bank_np)} test={len(y_score)} thr={thr:.4f} "
            f"flagged={int(decisions.sum())} mean={np.mean(y_score):.4f}"
        )

    if not summary_rows:
        print("Geen machines verwerkt. Controleer train/eval roots en .wav bestanden.")
        return

    summary = pd.DataFrame(summary_rows).set_index("machine")
    print("\nInference summary per machine:")
    print(summary)

    if args.plots_dir:
        summary_csv = os.path.join(args.plots_dir, "eval_inference_summary.csv")
        summary.to_csv(summary_csv)
        print(f"Saved inference summary -> {summary_csv}")

    print(f"\nDCASE submission CSV's geschreven in: {args.out_dir}")
    print("Draai nu de officiele evaluator vanuit de dcase2025_task2_evaluator map.")


if __name__ == "__main__":
    main()
