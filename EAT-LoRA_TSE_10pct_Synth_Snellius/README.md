# EAT-LoRA + TSE + 10% synth on the DCASE evaluation set

This folder contains a complete Snellius pipeline for DCASE 2025 Task 2. It builds synthetic training data, trains TSE and EAT-LoRA, runs kNN evaluation on the official evaluation dataset, and saves plots and CSV results.

Training uses about ten percent synthetic audio and ninety percent raw additional train audio. kNN evaluation uses the raw additional train set as the normal reference bank.

## What is inside this folder

The `code/` folder holds all Python scripts. The `scripts/` folder holds shell helpers to stage data, train models, and run evaluation. The file `run_full_pipeline.slurm` starts the full job on Snellius. Run `setup_snellius.sh` once to create the Python environment. See `PYTHON_README.md` for a short explanation of each Python file. See `SNELLIUS_DEPLOY.md` for upload steps and more detail.

## Datasets (not included in git)

Upload these three folders next to this project on Snellius:

1. `Additional dataset DCASE/`
2. `Evaluation dataset DCASE/`
3. `dcase2025_task2_evaluator/`

## Quick start on Snellius

```bash
cd ~/projects/EAT-LoRA_TSE_10pct_Synth_Snellius
bash setup_snellius.sh
bash scripts/preflight_data_paths.sh
sbatch run_full_pipeline.slurm
```

Run `setup_snellius.sh` only the first time. Use `SNELLIUS_DEPLOY.md` if you need the full upload guide from Windows.

## Outputs after a successful run

The job writes model weights to `checkpoints/` and metrics and figures to `results/`. These folders are not tracked in git.
