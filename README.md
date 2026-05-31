# Team-Internship-Sorama

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download EAT model from https://github.com/cwx-worst-one/EAT

4. Download DCASE2025 data: https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

5. Download Sorama data (optional) - Available in multiple dataset versions:
   - `Data Sorama-20260425T105917Z-3-001`
   - `Data Sorama-20260425T105917Z-3-002`
   - `Data Sorama-20260425T105917Z-3-003`

## Data Structure

**DCASE2025:**
```
data/dcase2025t2/
└── {dataset_id}/
    ├── dev_{machine}/{machine}/supplemental/
    │   └── section_00_machine_source_0000_noAttribute.wav
    ├── dev_{machine}/{machine}/normal_files/
    ├── dev_{machine}/{machine}/anomalous_files/
    ├── test_{machine}/{machine}/test_files/
    └── ...
```

Example: `15097779/dev_bearing/bearing/supplemental/section_00_machine_source_0000_noAttribute.wav`

**Sorama (optional):**
```
Sorama_data/
├── Data Sorama-20260425T105917Z-3-001/
│   └── Data Sorama/
│       ├── Bearing/
│       │   └── Asset {1-4}/
│       │       ├── 0_normal/
│       │       │   └── *.wav
│       │       └── 1_anomaly/
│       │           └── *.wav
│       └── Pump/
│           └── Asset {1-4}/
│               ├── 0_normal/
│               │   └── *.wav
│               └── 1_anomaly/
│                   └── *.wav
├── Data Sorama-20260425T105917Z-3-002/
└── Data Sorama-20260425T105917Z-3-003/
```

Example: `Data Sorama-20260425T105917Z-3-003/Data Sorama/Bearing/Asset 1/0_normal/beamformed_api_response_1404_37s_R1B1_140cm_E.wav`

## Scripts

**Data Synthesis:**
- `synthesize_dcase_data.py` - Synthesize DCASE dataset
  ```bash
  python synthesize_dcase_data.py --input_dir <path> --output_dir <path>
  ```

- `synthesize_sorama_data.py` - Synthesize Sorama dataset
  ```bash
  python synthesize_sorama_data.py --input-roots <path1> [<path2> <path3>] --output-root <path> [--copies-per-source 10] [--dry-run]
  ```
  Example:
  ```bash
  python synthesize_sorama_data.py --input-roots "C:\Downloads\Sorama_data\Data Sorama-20260425T105917Z-3-001" --output-root "./sorama_synthetic" --copies-per-source 5
  ```

**Training:**
- `train_EAT_LoRa_snellius.py` - Train EAT model with LoRA adapters
  ```bash
  python train_EAT_LoRa_snellius.py --data_dir <path> --output_dir <path> [--epochs 10]
  ```

- `train_eat_auddsr_snellius.py` - Train EAT + AudDSR model
  ```bash
  python train_eat_auddsr_snellius.py --data_dir <path> --output_dir <path>
  ```

**Evaluation:**
- `eval_wang2025_knn_snellius.py` - Evaluate Wang2025 with KNN
  ```bash
  python eval_wang2025_knn_snellius.py --data_dir <path> --model_path <path>
  ```

**Notebooks:**
- `test_Wang2025_clean.ipynb` - **RECOMMENDED** - Clean Wang2025 model testing (label encoding, data loading, inference)
- `AudDSR_clean.ipynb` - **RECOMMENDED** - Clean AudDSR training pipeline (Stage 1: VQ-VAE, Stage 2: Detector)
- `test_Wang2025_Luc.ipynb` - Original Wang2025 notebook (extensive comments, for reference)
- `AudDSR.ipynb` - Original AudDSR notebook (large cells, for reference)