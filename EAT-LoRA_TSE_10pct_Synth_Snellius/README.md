# EAT-LoRA + TSE + 10% synth training on DCASE evaluation dataset

Self-contained Snellius pipeline: synthetic data, TSE training, EAT-LoRA training (10% synth / 90% raw mix), KNN evaluation on the official evaluation set, and plots.

## Folder layout

```
EAT-LoRA_TSE_10pct_Synth_Snellius/
├── code/                  Python scripts (11 files)
├── scripts/               Shell helpers + activate env
├── run_full_pipeline.slurm
├── setup_snellius.sh
├── requirements_snellius.txt
├── PYTHON_README.md       What each .py file does
└── SNELLIUS_DEPLOY.md     Upload and sbatch steps
```

## What you upload separately (not in git)

Place these next to this folder on Snellius:

- `Additional dataset DCASE/`
- `Evaluation dataset DCASE/`
- `dcase2025_task2_evaluator/`

## Quick start on Snellius

```bash
cd ~/projects/EAT-LoRA_TSE_10pct_Synth_Snellius
bash setup_snellius.sh          # once
bash scripts/preflight_data_paths.sh
sbatch run_full_pipeline.slurm
```

See SNELLIUS_DEPLOY.md for full upload instructions.

## Outputs (created by the job, not committed)

- `checkpoints/` — EAT-LoRA and TSE weights
- `results/` — KNN CSVs and PNG plots
