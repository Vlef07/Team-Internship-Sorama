"""
Run BEATs + EAT models, collect evaluation metrics.
BEATs features are cached from overnight run; EAT needs extraction.
Results saved to results_all_models.json
"""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import torch
import torchaudio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR.parent / "unilm" / "beats"))

YEAR = "dcase2025t2"
MACHINES = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
TARGET_SR = 16000
PCA_COMPONENTS = 128
KNN_K = 5
KNN_THRESHOLD_PERCENTILE = 95
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# EAT mel spectrogram parameters (from Victor's notebook)
EAT_TARGET_LENGTH = 512
EAT_NORM_MEAN = -4.268
EAT_NORM_STD = 4.569


def get_data_path(machine):
    return BASE_DIR / "data" / YEAR / "dev_data" / "raw" / machine


def get_features_dir(model_name, machine):
    p = BASE_DIR / "features" / model_name / f"{YEAR}_features" / "dev_features" / "raw_features" / f"{machine}_features"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────── Audio loading ────────────────────────────

def load_audio(file_path, target_sr=TARGET_SR):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, target_sr).squeeze(0)
    if waveform.shape[0] < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[0]))
    return waveform


def wav_to_mel(file_path, target_length=EAT_TARGET_LENGTH):
    """Convert wav file to mel spectrogram for EAT (from Victor's UASD_EAT.ipynb)."""
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000,
        use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10
    )  # [T, 128]

    T = fbank.shape[0]
    if T < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length, :]

    fbank = (fbank - EAT_NORM_MEAN) / (EAT_NORM_STD * 2)
    return fbank.unsqueeze(0).unsqueeze(0)  # [1, 1, T, 128]


def _load_audio_worker(args):
    file_path, target_sr = args
    try:
        return (file_path.stem, load_audio(str(file_path), target_sr))
    except Exception:
        return (file_path.stem, None)


def _load_mel_worker(args):
    file_path, _ = args
    try:
        return (file_path.stem, wav_to_mel(str(file_path)))
    except Exception:
        return (file_path.stem, None)


# ──────────────────────────── Feature extraction ────────────────────────────

def extract_features_for_split(model_name, machine, split, extract_fn, loader_fn):
    """Extract features with partial caching."""
    features_dir = get_features_dir(model_name, machine)
    split_dir = features_dir / f"{split}_features"
    split_dir.mkdir(parents=True, exist_ok=True)

    data_path = get_data_path(machine) / split
    if not data_path.exists():
        print(f"  [WARN] {data_path} does not exist, skipping")
        return None

    wav_files = sorted(data_path.glob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No WAV files in {data_path}")
        return None

    already_done = {f.stem for f in split_dir.glob("*.npy")}
    todo_files = [f for f in wav_files if f.stem not in already_done]

    if not todo_files:
        print(f"  Skipping {split} - all {len(wav_files)} features cached")
        return np.array([np.load(split_dir / f"{f.stem}.npy") for f in wav_files])

    cached = len(already_done)
    total_todo = len(todo_files)
    print(f"  {split}: {total_todo} to process, {cached} cached, {len(wav_files)} total (batch={BATCH_SIZE})")

    processed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch_start in range(0, total_todo, BATCH_SIZE):
            batch_files = todo_files[batch_start:batch_start + BATCH_SIZE]
            load_args = [(f, TARGET_SR) for f in batch_files]
            results = list(executor.map(loader_fn, load_args))

            for name, data in results:
                if data is None:
                    continue
                try:
                    emb = extract_fn(data)
                    np.save(split_dir / f"{name}.npy", emb)
                    processed += 1
                except Exception as e:
                    print(f"    Error on {name}: {e}")

            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_todo - processed
            eta = remaining / rate if rate > 0 else 0
            if processed % 50 < BATCH_SIZE or processed >= total_todo:
                print(f"    {processed}/{total_todo} new ({rate:.1f}/sec, ETA: {eta:.0f}s)")

    print(f"  Done: {processed} new + {cached} cached = {processed + cached} total")
    all_features = []
    for f in wav_files:
        npy_path = split_dir / f"{f.stem}.npy"
        if npy_path.exists():
            all_features.append(np.load(npy_path))
    return np.array(all_features) if all_features else None


# ──────────────────────────── PCA + KNN ────────────────────────────

def apply_pca(model_name, machine, train_features, test_features):
    pca_dir = get_features_dir(model_name, machine) / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    n_comp = min(PCA_COMPONENTS, train_features.shape[1], train_features.shape[0])
    pca = PCA(n_components=n_comp)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)

    np.save(pca_dir / "train_features_pca.npy", train_pca)
    np.save(pca_dir / "test_features_pca.npy", test_pca)
    with open(pca_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    print(f"  PCA: {train_features.shape[1]}d -> {train_pca.shape[1]}d (var: {pca.explained_variance_ratio_.sum():.3f})")
    return train_pca, test_pca


def get_test_labels(machine):
    wav_files = sorted((get_data_path(machine) / "test").glob("*.wav"))
    labels = []
    for f in wav_files:
        name = f.stem.lower()
        if "anomaly" in name:
            labels.append(1)
        elif "normal" in name:
            labels.append(0)
        else:
            labels.append(-1)
    return np.array(labels)


def get_domain_info(machine):
    wav_files = sorted((get_data_path(machine) / "test").glob("*.wav"))
    domains = []
    for f in wav_files:
        name = f.stem.lower()
        if "source" in name:
            domains.append("source")
        elif "target" in name:
            domains.append("target")
        else:
            domains.append("unknown")
    return domains


def evaluate_knn(train_pca, test_pca, y_test, domains, machine_name):
    knn = NearestNeighbors(n_neighbors=KNN_K + 1)
    knn.fit(train_pca)

    train_distances, _ = knn.kneighbors(train_pca)
    threshold = np.percentile(train_distances[:, KNN_K], KNN_THRESHOLD_PERCENTILE)

    test_distances, _ = knn.kneighbors(test_pca)
    test_kth_dist = test_distances[:, KNN_K]

    predictions = (test_kth_dist > threshold).astype(int)
    y_scores = test_kth_dist

    valid = y_test >= 0
    y_valid, pred_valid, scores_valid = y_test[valid], predictions[valid], y_scores[valid]
    domains_valid = [d for d, v in zip(domains, valid) if v]

    precision, recall, f1, _ = precision_recall_fscore_support(y_valid, pred_valid, average='binary')
    roc_auc_all = roc_auc_score(y_valid, scores_valid)

    source_mask = np.array([d == "source" for d in domains_valid])
    target_mask = np.array([d == "target" for d in domains_valid])

    def safe_auc(mask):
        if mask.sum() > 1 and len(np.unique(y_valid[mask])) > 1:
            return roc_auc_score(y_valid[mask], scores_valid[mask])
        return None

    auc_source = safe_auc(source_mask)
    auc_target = safe_auc(target_mask)

    try:
        pauc = roc_auc_score(y_valid, scores_valid, max_fpr=0.1)
    except ValueError:
        pauc = None

    return {
        "machine": machine_name,
        "AUC_source": round(auc_source * 100, 2) if auc_source else None,
        "AUC_target": round(auc_target * 100, 2) if auc_target else None,
        "pAUC": round(pauc * 100, 2) if pauc else None,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "ROC_AUC_overall": round(roc_auc_all * 100, 2),
        "test_samples": int(valid.sum()),
    }


# ──────────────────────────── Model runners ────────────────────────────

def run_beats_frozen():
    from BEATs import BEATs, BEATsConfig

    print("\n" + "=" * 80)
    print("  MODEL: BEATs (frozen)")
    print("=" * 80)

    ckpt_path = BASE_DIR / "BEATs_iter3_plus_AS2M.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] BEATs checkpoint not found: {ckpt_path}")
        return None

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    print(f"  BEATs loaded on {DEVICE}")

    def extract_fn(waveform):
        waveform = waveform.to(DEVICE)
        with torch.no_grad():
            features, _ = model.extract_features(waveform.unsqueeze(0))
            return features.mean(dim=1).squeeze(0).cpu().numpy()

    results = []
    for machine in MACHINES:
        print(f"\n--- {machine} ---")
        if not get_data_path(machine).exists():
            print(f"  Data path not found, skipping")
            continue

        train_feat = extract_features_for_split("BEATs", machine, "train", extract_fn, _load_audio_worker)
        test_feat = extract_features_for_split("BEATs", machine, "test", extract_fn, _load_audio_worker)

        if train_feat is None or test_feat is None:
            continue

        train_pca, test_pca = apply_pca("BEATs", machine, train_feat, test_feat)
        y_test = get_test_labels(machine)
        domains = get_domain_info(machine)
        result = evaluate_knn(train_pca, test_pca, y_test, domains, machine)
        results.append(result)
        print(f"  AUC_src={result['AUC_source']}  AUC_tgt={result['AUC_target']}  pAUC={result['pAUC']}  "
              f"P={result['Precision']}  R={result['Recall']}  F1={result['F1']}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


def run_eat_frozen():
    print("\n" + "=" * 80)
    print("  MODEL: EAT (frozen)")
    print("=" * 80)

    try:
        import transformers
        from transformers import AutoModel

        _PATCH_FLAG = "_eat_compat_patched"
        if not getattr(transformers.PreTrainedModel, _PATCH_FLAG, False):
            _orig = getattr(transformers.PreTrainedModel, "_adjust_tied_keys_with_tied_pointers", None)
            if _orig is not None:
                def _eat_adjust(self, missing_keys):
                    v = getattr(self, "all_tied_weights_keys", None)
                    if not isinstance(v, dict):
                        self.all_tied_weights_keys = {}
                    return _orig(self, missing_keys)
                transformers.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _eat_adjust
                setattr(transformers.PreTrainedModel, _PATCH_FLAG, True)

        eat_model = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)
        eat_model.eval()
        eat_model.to(DEVICE)
        print(f"  EAT loaded on {DEVICE}")
    except Exception as e:
        print(f"[ERROR] Failed to load EAT model: {e}")
        import traceback; traceback.print_exc()
        return None

    def extract_fn(mel_tensor):
        """Extract EAT embedding from a mel spectrogram tensor [1, 1, T, 128]."""
        mel_tensor = mel_tensor.to(DEVICE)
        with torch.no_grad():
            out = eat_model.extract_features(mel_tensor)  # [1, T, 768]
            return out[:, 0, :].squeeze(0).cpu().numpy()   # CLS token -> [768]

    results = []
    for machine in MACHINES:
        print(f"\n--- {machine} ---")
        if not get_data_path(machine).exists():
            print(f"  Data path not found, skipping")
            continue

        train_feat = extract_features_for_split("EAT", machine, "train", extract_fn, _load_mel_worker)
        test_feat = extract_features_for_split("EAT", machine, "test", extract_fn, _load_mel_worker)

        if train_feat is None or test_feat is None:
            continue

        train_pca, test_pca = apply_pca("EAT", machine, train_feat, test_feat)
        y_test = get_test_labels(machine)
        domains = get_domain_info(machine)
        result = evaluate_knn(train_pca, test_pca, y_test, domains, machine)
        results.append(result)
        print(f"  AUC_src={result['AUC_source']}  AUC_tgt={result['AUC_target']}  pAUC={result['pAUC']}  "
              f"P={result['Precision']}  R={result['Recall']}  F1={result['F1']}")

    del eat_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ──────────────────────────── Summary + Main ────────────────────────────

def print_summary(all_results):
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY")
    print("=" * 100)

    for model_name, results in all_results.items():
        if results is None:
            print(f"\n{model_name}: FAILED")
            continue

        print(f"\n{model_name}:")
        print(f"  {'Machine':<12} {'AUC_src':>8} {'AUC_tgt':>8} {'pAUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 60)
        for r in results:
            print(f"  {r['machine']:<12} {str(r.get('AUC_source','--')):>8} {str(r.get('AUC_target','--')):>8} "
                  f"{str(r.get('pAUC','--')):>8} {r['Precision']:>8} {r['Recall']:>8} {r['F1']:>8}")

        vals = lambda key: [r[key] for r in results if r.get(key) is not None]
        if results:
            print("  " + "-" * 60)
            print(f"  {'AVERAGE':<12} {np.mean(vals('AUC_source')):>8.2f} {np.mean(vals('AUC_target')):>8.2f} "
                  f"{np.mean(vals('pAUC')):>8.2f} {np.mean(vals('Precision')):>8.2f} "
                  f"{np.mean(vals('Recall')):>8.2f} {np.mean(vals('F1')):>8.2f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Machines: {MACHINES}")
    start_time = time.time()

    all_results = {}

    beats_results = run_beats_frozen()
    all_results["BEATs (frozen)"] = beats_results

    eat_results = run_eat_frozen()
    all_results["EAT (frozen)"] = eat_results

    print_summary(all_results)

    output_file = BASE_DIR / "results_all_models.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
