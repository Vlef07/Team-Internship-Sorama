"""
this script runs knn evaluation on the official dcase 2025 task 2 evaluation dataset
train data comes from the additional dataset and test data uses anonymous filenames
step one build the reference bank from normal train wavs in the additional training dataset
step two load all test wavs from the evaluation dataset per machine
step three optionally enhance each wav with the trained tse model for that machine
step four convert wavs to mel spectrograms and eat embeddings the same way as during training
step five score each test clip with knn against the bank and write anomaly scores
step six compare scores to official ground truth csvs to compute auc and pauc per machine
step seven write dcase submission score and decision csvs plus a knn eval summary csv for plots
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn import metrics
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR
sys.path.insert(0, str(_CODE_DIR))

from eval_wang2025_knn_snellius import (  # noqa: E402
    _load_eat_lora_from_checkpoint,
    auc_domain_like_evaluator,
    auc_pauc_all,
    embed_mel_batch,
    knn_min_cosine_distance_scores,
    load_mel_from_wav_path,
)
from tse_snellius import load_tse_models  # noqa: E402

DEFAULT_MACHINES = [
    "AutoTrash",
    "BandSealer",
    "CoffeeGrinder",
    "HomeCamera",
    "Polisher",
    "ScrewFeeder",
    "ToyPet",
    "ToyRCCar",
]


def _resolve_wav_path(base_path: str, machine: str, split: str, file_name: str) -> str:
    name = str(file_name)
    if not name.lower().endswith(".wav"):
        name = name + ".wav"
    return os.path.join(base_path, machine, split, name)


def list_train_normal_wavs(machine: str, train_root: str) -> list[str]:
    train_dir = os.path.join(train_root, machine, "train")
    if not os.path.isdir(train_dir):
        return []
    out = []
    for wav_name in sorted(os.listdir(train_dir)):
        if not wav_name.lower().endswith(".wav"):
            continue
        low = wav_name.lower()
        if "anomaly" in low:
            continue
        if "train" not in low or "normal" not in low:
            continue
        out.append(wav_name)
    return out


def list_eval_test_wavs(machine: str, test_root: str) -> list[str]:
    test_dir = os.path.join(test_root, machine, "test")
    if not os.path.isdir(test_dir):
        return []
    return sorted(
        name for name in os.listdir(test_dir) if name.lower().endswith(".wav")
    )


@torch.no_grad()
def embed_wav_paths(
    eat_lora,
    wav_paths: list[str],
    device: torch.device,
    embed_batch_size: int,
    desc: str,
    tse_model=None,
):
    embeddings = []
    pending_mels = []
    for wav_path in tqdm(wav_paths, desc=desc, leave=False):
        pending_mels.append(
            load_mel_from_wav_path(wav_path, tse_model=tse_model, tse_device=device)
        )
        if len(pending_mels) >= embed_batch_size:
            batch = torch.stack(pending_mels, dim=0)
            embeddings.append(embed_mel_batch(eat_lora, batch, device))
            pending_mels = []
    if pending_mels:
        batch = torch.stack(pending_mels, dim=0)
        embeddings.append(embed_mel_batch(eat_lora, batch, device))
    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=0)


def threshold_from_train_bank(bank_np: np.ndarray) -> float:
    if len(bank_np) < 2:
        return float(np.median(knn_min_cosine_distance_scores(bank_np, bank_np)))
    sim = bank_np @ bank_np.T
    np.fill_diagonal(sim, -np.inf)
    max_sim = sim.max(axis=1)
    train_scores = 1.0 - max_sim
    return float(np.percentile(train_scores, 90))


def write_submission_csv(rows, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)


def metrics_from_ground_truth(
    score_csv: str,
    gt_label_csv: str,
    gt_domain_csv: str,
    max_fpr: float,
):
    gt_labels = pd.read_csv(gt_label_csv, header=None, names=["file_name", "value"])
    gt_domains = pd.read_csv(gt_domain_csv, header=None, names=["file_name", "value"])
    scores = pd.read_csv(score_csv, header=None, names=["file_name", "score"])
    score_map = dict(zip(scores["file_name"], scores["score"]))

    y_true = []
    y_score = []
    y_domain = []
    for _, row in gt_labels.iterrows():
        fname = row["file_name"]
        if fname not in score_map:
            continue
        y_true.append(int(row["value"]))
        y_score.append(float(score_map[fname]))
        dom_row = gt_domains.loc[gt_domains["file_name"] == fname, "value"]
        y_domain.append(int(dom_row.iloc[0]))

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=np.float64)
    y_domain = np.array(y_domain, dtype=int)

    auc_all, pauc = auc_pauc_all(y_true, y_score, max_fpr=max_fpr)
    auc_src = auc_domain_like_evaluator(y_true, y_score, y_domain, 0)
    auc_tgt = auc_domain_like_evaluator(y_true, y_score, y_domain, 1)
    return auc_all, pauc, auc_src, auc_tgt, len(y_true)


def run_official_evaluator(evaluator_root: str, team_name: str, system_name: str):
    script = os.path.join(evaluator_root, "dcase2025_task2_evaluator.py")
    if not os.path.isfile(script):
        print(f"Official evaluator not found, skip: {script}")
        return

    teams_root = os.path.join(evaluator_root, "teams")
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--teams_root_dir",
            teams_root,
            "--result_dir",
            os.path.join(evaluator_root, "teams_result"),
            "--additional_result_dir",
            os.path.join(evaluator_root, "teams_additional_result"),
            "--dir_depth",
            "2",
            "--out_all",
            "False",
        ],
        cwd=evaluator_root,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("dcase2025_task2_evaluator.py failed")

    result_csv = os.path.join(
        evaluator_root, "teams_result", f"{system_name}_result.csv"
    )
    if os.path.isfile(result_csv):
        print(f"Official evaluator result: {result_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--additional-train-root",
        required=True,
        help="Additional training dataset root (machine/train, machine/test).",
    )
    parser.add_argument(
        "--eval-test-root",
        required=True,
        help="Evaluation dataset root (machine/test with unlabeled wav names).",
    )
    parser.add_argument("--evaluator-root", default="", help="dcase2025_task2_evaluator root.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="worstchan/EAT-base_epoch30_finetune_AS2M")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--output-csv", default="results/knn_eval_dcase_eval.csv")
    parser.add_argument("--tse-checkpoint-dir", default="")
    parser.add_argument(
        "--tse-mode",
        choices=["off", "test-only", "both"],
        default="off",
    )
    parser.add_argument("--team-name", default="our_team")
    parser.add_argument("--system-name", default="eat_lora_tse_10pct_synth_15db_snellius")
    parser.add_argument("--section-id", default="section_00")
    parser.add_argument(
        "--run-official-evaluator",
        action="store_true",
        help="Run dcase2025_task2_evaluator.py after writing submission CSVs.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Only compute metrics CSV (requires existing score CSVs or re-run embed).",
    )
    args = parser.parse_args()

    # step one check paths and load eat lora weights onto gpu or cpu
    machines = args.machines if args.machines else list(DEFAULT_MACHINES)
    train_root = os.path.normpath(args.additional_train_root)
    test_root = os.path.normpath(args.eval_test_root)

    for label, path in [("additional-train-root", train_root), ("eval-test-root", test_root)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} bestaat niet: {path}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint bestaat niet: {args.checkpoint}")

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        if os.path.isdir(args.tse_checkpoint_dir):
            tse_models = load_tse_models(args.tse_checkpoint_dir, device)
            print(f"TSE geladen voor: {sorted(tse_models.keys())} (mode={args.tse_mode})")
        else:
            print(f"TSE map ontbreekt: {args.tse_checkpoint_dir}")

    submission_dir = None
    if args.evaluator_root and not args.skip_submission:
        submission_dir = os.path.join(
            args.evaluator_root,
            "teams",
            args.team_name,
            args.system_name,
        )
        os.makedirs(submission_dir, exist_ok=True)
        print(f"Submission map: {submission_dir}")

    rows_out = []

    for machine in machines:
        train_names = list_train_normal_wavs(machine, train_root)
        test_names = list_eval_test_wavs(machine, test_root)

        if not train_names:
            print(f"[{machine}] skip: geen train normal wavs in {train_root}")
            continue
        if not test_names:
            print(f"[{machine}] skip: geen test wavs in {test_root}")
            continue

        train_paths = [
            _resolve_wav_path(train_root, machine, "train", n) for n in train_names
        ]
        test_paths = [
            _resolve_wav_path(test_root, machine, "test", n) for n in test_names
        ]

        tse_model = tse_models.get(machine)
        tse_for_bank = tse_model if args.tse_mode == "both" else None
        tse_for_test = tse_model if args.tse_mode in ("test-only", "both") else None

        if args.tse_mode != "off" and tse_model is None:
            print(f"[{machine}] geen TSE checkpoint; raw waveforms")

        bank_np = embed_wav_paths(
            eat_lora,
            train_paths,
            device,
            args.embed_batch_size,
            f"{machine} train",
            tse_model=tse_for_bank,
        )
        if bank_np is None:
            print(f"[{machine}] skip: lege train bank")
            continue

        query_np = embed_wav_paths(
            eat_lora,
            test_paths,
            device,
            args.embed_batch_size,
            f"{machine} test",
            tse_model=tse_for_test,
        )
        if query_np is None:
            print(f"[{machine}] skip: lege test embeddings")
            continue

        scores = knn_min_cosine_distance_scores(bank_np, query_np)
        threshold = threshold_from_train_bank(bank_np)
        decisions = (scores > threshold).astype(int)

        score_csv = None
        if submission_dir:
            score_csv = os.path.join(
                submission_dir,
                f"anomaly_score_{machine}_{args.section_id}_test.csv",
            )
            decision_csv = os.path.join(
                submission_dir,
                f"decision_result_{machine}_{args.section_id}_test.csv",
            )
            write_submission_csv(
                [[n, float(s)] for n, s in zip(test_names, scores)], score_csv
            )
            write_submission_csv(
                [[n, int(d)] for n, d in zip(test_names, decisions)], decision_csv
            )

        auc_all = pauc = auc_src = auc_tgt = float("nan")
        n_test_gt = 0
        if args.evaluator_root:
            gt_label = os.path.join(
                args.evaluator_root,
                "ground_truth_data",
                f"ground_truth_{machine}_{args.section_id}_test.csv",
            )
            gt_domain = os.path.join(
                args.evaluator_root,
                "ground_truth_domain",
                f"ground_truth_{machine}_{args.section_id}_test.csv",
            )
            if score_csv and os.path.isfile(gt_label) and os.path.isfile(gt_domain):
                auc_all, pauc, auc_src, auc_tgt, n_test_gt = metrics_from_ground_truth(
                    score_csv, gt_label, gt_domain, args.max_fpr
                )
            else:
                print(f"[{machine}] ground truth of score csv ontbreekt voor metrics")

        rows_out.append(
            {
                "machine": machine,
                "n_bank": len(bank_np),
                "n_test": len(scores),
                "n_test_scored_gt": n_test_gt,
                "AUC_all": auc_all,
                "pAUC": pauc,
                "AUC_source": auc_src,
                "AUC_target": auc_tgt,
            }
        )
        print(
            f"{machine}: AUC_all={auc_all:.4f} pAUC={pauc:.4f} "
            f"AUC_src={auc_src:.4f} AUC_tgt={auc_tgt:.4f}  "
            f"bank={len(bank_np)} test={len(scores)}"
        )

    if not rows_out:
        raise RuntimeError("Geen machines geevalueerd. Controleer paden en wav-bestanden.")

    summary = pd.DataFrame(rows_out).set_index("machine")
    print("\nSamenvatting per machine:")
    print(summary)
    summary.to_csv(args.output_csv)
    print(f"KNN metrics csv: {args.output_csv}")

    valid = summary["AUC_all"].notna()
    if valid.any():
        flat = []
        for m in summary.loc[valid].index:
            r = summary.loc[m]
            flat.extend([r["AUC_source"], r["AUC_target"], r["pAUC"]])
        flat = np.maximum(np.array(flat, dtype=np.float64), sys.float_info.epsilon)
        print(
            "Harmonisch gemiddelde (AUC_src, AUC_tgt, pAUC per machine):",
            float(stats.hmean(flat)),
        )

    if args.run_official_evaluator and args.evaluator_root:
        run_official_evaluator(args.evaluator_root, args.team_name, args.system_name)


if __name__ == "__main__":
    main()
