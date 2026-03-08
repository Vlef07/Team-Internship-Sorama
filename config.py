# config.py
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"

# Years to process
YEARS = ["dcase2023t2", "dcase2024t2", "dcase2025t2"]

# Machine types per year (can differ)
MACHINES_PER_YEAR = {
    "dcase2023t2": ["bearing", "fan", "gearbox", "ToyCar", "ToyTrain", "valve"],
    "dcase2024t2": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
    "dcase2025t2": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
}

def get_raw_data_path(year, machine, dataset="dev"):
    """Get path to raw audio data"""
    return DATA_DIR / year / f"{dataset}_data" / "raw" / machine

def get_features_path(year, machine, dataset="dev", create=True):
    """Get path where features should be saved
    
    Returns the machine-level features directory containing:
    - train_features/ (BEATs embeddings from train data)
    - test_features/ (BEATs embeddings from test data)  
    - pca/ (PCA-transformed features and models)
    """
    # Match the actual directory structure: dcase2025t2_features/dev_features/raw_features/bearing_features/
    year_suffix = f"{year}_features"
    dataset_suffix = f"{dataset}_features"
    machine_suffix = f"{machine}_features"
    path = FEATURES_DIR / year_suffix / dataset_suffix / "raw_features" / machine_suffix
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_train_features_path(year, machine, dataset="dev", create=True):
    """Get path to train features subdirectory"""
    base_path = get_features_path(year, machine, dataset, create=create)
    train_path = base_path / "train_features"
    if create:
        train_path.mkdir(parents=True, exist_ok=True)
    return train_path

def get_test_features_path(year, machine, dataset="dev", create=True):
    """Get path to test features subdirectory"""
    base_path = get_features_path(year, machine, dataset, create=create)
    test_path = base_path / "test_features"
    if create:
        test_path.mkdir(parents=True, exist_ok=True)
    return test_path

def get_pca_path(year, machine, dataset="dev", create=True):
    """Get path to PCA results subdirectory"""
    base_path = get_features_path(year, machine, dataset, create=create)
    pca_path = base_path / "pca"
    if create:
        pca_path.mkdir(parents=True, exist_ok=True)
    return pca_path

def load_pca_results(year, machine, split="train", dataset="dev"):
    """
    Load PCA results for a specific machine and split.
    
    Args:
        year: Year (e.g., 'dcase2023t2')
        machine: Machine type (e.g., 'bearing')
        split: 'train' or 'test'
        dataset: Dataset type (default: 'dev')
        
    Returns:
        X_pca: PCA-transformed features (numpy array or None)
        variance_info: Dictionary with PCA metadata (dict or None)
    """
    import numpy as np
    
    pca_dir = get_pca_path(year, machine, dataset=dataset, create=False)
    
    if not pca_dir.exists():
        return None, None
    
    # Load PCA features
    pca_file = pca_dir / f"{split}_features_pca.npy"
    if not pca_file.exists():
        return None, None
    
    try:
        X_pca = np.load(pca_file)
    except Exception as e:
        print(f"Error loading PCA features: {e}")
        return None, None
    
    # Load variance info
    variance_file = pca_dir / "variance_info.npy"
    variance_info = None
    if variance_file.exists():
        try:
            variance_info = np.load(variance_file, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading variance info: {e}")
    
    return X_pca, variance_info

def get_all_machines(year):
    """Get list of machines for a specific year"""
    return MACHINES_PER_YEAR.get(year, [])

def discover_machines(year, dataset="dev"):
    """Auto-discover machines from the data directory for a specific year"""
    year_path = DATA_DIR / year / f"{dataset}_data" / "raw"
    if not year_path.exists():
        return []
    machines = [d.name for d in year_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    return sorted(machines)

def iter_all_data(years=None, dataset="dev"):
    """Iterator over all available (year, machine) pairs"""
    if years is None:
        years = YEARS
    
    for year in years:
        machines = discover_machines(year, dataset)
        for machine in machines:
            input_path = get_raw_data_path(year, machine, dataset)
            if input_path.exists():
                yield year, machine, input_path