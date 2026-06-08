# How to run this pipeline on Snellius

This guide explains how to upload the project to Snellius and start the full training and evaluation job. You only need to do the upload and setup steps once. After that you can rerun the job with sbatch.

On Snellius your project folder should be something like

    ~/projects/EAT-LoRA_TSE_10pct_Synth_Snellius

The job also uses scratch storage for fast disk access. That path is created automatically and looks like

    $HOME/scratch/EAT-LoRA_TSE_10pct_Synth_Snellius/

Inside scratch you will see three subfolders after staging. The folder additional_train_raw holds the real additional train data. The folder additional_train_synth holds the synthetic copies used for training. The folder evaluation_test holds the official evaluation test wavs.

## Step one upload from your Windows pc

Open PowerShell and set your local path, your Snellius username, and the remote project path. Then create the folder on Snellius and copy the files.

```powershell
$LOCAL = "C:\path\to\Evaluation_BOTH_Trained_op_additional_dataset"
$REMOTE = "user@snellius.surf.nl"
$RPROJ = "/home/user/projects/EAT-LoRA_TSE_10pct_Synth_Snellius"

ssh $REMOTE "mkdir -p $RPROJ"

scp -r "$LOCAL\EAT-LoRA_TSE_10pct_Synth_Snellius" "${REMOTE}:/home/user/projects/"

scp -r "$LOCAL\Additional dataset DCASE" "${REMOTE}:${RPROJ}/"
scp -r "$LOCAL\Evaluation dataset DCASE" "${REMOTE}:${RPROJ}/"
scp -r "$LOCAL\dcase2025_task2_evaluator" "${REMOTE}:${RPROJ}/"
```

The first scp command uploads this pipeline folder with all Python and shell scripts. The next three commands upload the datasets and the official DCASE evaluator. Those three folders must sit inside the project folder on Snellius next to the code folder.

If you edited scripts on Windows, fix line endings once after upload so bash does not fail.

```powershell
ssh $REMOTE "find $RPROJ -type f \( -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +"
```

## Step two setup on Snellius

Log in to Snellius and go to the project folder. Run the Python environment setup once. It installs torch and the other packages into a venv in your home directory.

```bash
export EVAL_PROJECT_DIR=~/projects/EAT-LoRA_TSE_10pct_Synth_Snellius
cd "$EVAL_PROJECT_DIR"

find . -type f \( -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +

bash setup_snellius.sh
```

Before starting a long job, check that Snellius can find both datasets.

```bash
bash scripts/preflight_data_paths.sh
```

If evaluation data lives somewhere else you can point to it manually.

```bash
export EVAL_TEST_HOME=/path/to/Evaluation\ dataset\ DCASE
```

## Step three start the full job

From the project folder run

```bash
sbatch run_full_pipeline.slurm
```

Check whether the job is running with squeue. When it starts, log output goes to a file in your home folder named like dcase-full-retrain followed by the job id and .out.

```bash
squeue -u your_username
tail -f ~/dcase-full-retrain_<JOBID>.out
```

## What the job does

First it copies the additional train and evaluation test data to scratch if they are not there yet. Then it builds synthetic train wavs from the raw additional data with about 15 db noise by default. Then it trains TSE models and EAT LoRA using a mix of about ten percent synthetic batches and ninety percent raw batches. Then it runs kNN evaluation on the evaluation dataset. The normal train bank comes from raw additional data not from synth. TSE is applied on both bank and test waveforms during eval. Finally it writes csv results and png plots into the results folder.

Checkpoints go to checkpoints/eat_lora_additional_10pct_synth and checkpoints/tse_additional_10pct_synth inside the project folder.

## Optional settings before sbatch

You can change these if you want a different run. You do not need them for the default pipeline.

```bash
export SYNTH_SN_DB=15
export TRAIN_MIX_SYNTH_FRACTION=0.1
export SYNTH_REGENERATE=1
export TSE_STEPS=3000
```

SYNTH_SN_DB controls how loud the noise is when building synthetic data. TRAIN_MIX_SYNTH_FRACTION is the fraction of training steps that use the synth folder as primary data. SYNTH_REGENERATE forces the synth folder on scratch to be rebuilt. TSE_STEPS sets how many training steps each TSE model gets.

## Copy results back to your pc

After the job finishes you can download results and checkpoints with scp from PowerShell. Replace user and paths with your own values.

```powershell
$REMOTE = "user@snellius.surf.nl"
$RPROJ = "/home/user/projects/EAT-LoRA_TSE_10pct_Synth_Snellius"
$LOCAL = "C:\path\to\local\folder"

scp -r "${REMOTE}:${RPROJ}/results" "$LOCAL\"
scp -r "${REMOTE}:${RPROJ}/checkpoints" "$LOCAL\"
```
