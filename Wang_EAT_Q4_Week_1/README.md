# Team-Internship-Sorama

# Instructions: Using Jupytext with VS Code

## 1. Install Jupytext

Open a terminal in your project’s virtual environment and run:

```bash
pip install jupytext
```

Verify installation:

```bash
jupytext --version
```

## 2. Pair `.ipynb` with `.py`

* Convert notebook to script:

  ```bash
  jupytext --to py notebook.ipynb
  ```

* Convert script to notebook:

  ```bash
  jupytext --to notebook script.py
  ```

* For each notebook you want to track in Git:

    ```bash
    jupytext --set-formats ipynb,py:percent notebook.ipynb
    ```

    This creates a paired Python script `notebook.py` with `# %%` markers for each cell.

From now on:

* Edit/run the `.py` file in VS Code using the **Jupyter extension**.
* The `.ipynb` can still be opened normally if you prefer the notebook interface.

## 3. Keeping Files in Sync

When you make changes, you can sync both directions:

* When you push, update `.py` from `.ipynb`:

  ```bash
  jupytext --sync notebook.ipynb
  ```

* When you pull, update `.ipynb` from `.py`:

  ```bash
  jupytext --sync notebook.py
  ```

If you want **auto-sync**, we can set up a Git pre-commit hook so syncing happens before commits (see below).

## 4. Working in VS Code

* Install the **Jupyter extension** from the VS Code marketplace.
* Open the paired `.py` file in VS Code.
* You will see **Run Cell** (▶) buttons above `# %%` sections. These behave like Jupyter cells.
* Outputs appear in the **Interactive Window**, just like a notebook.

## 5. Commit and push

**Please push only .py files to the remote and keep notebooks on your local machine. If you want you can skip notebooks altogether and use .py files instead (as shown in step 4). I have added all files with .ipynb extension to .gitignore so they will not be tracked.**

# Setup instruction

1. Navigate to your project folder;

2. Create a `.venv` virtual environment;
   > ```bash
   > python -m venv .venv
   > ```

3. Activate your environment:

    - Linux / MacOS:
        > ```bash
        > source .venv/bin/activate
        > ```

    - Windows PowerShell:
        > ```bash
        > .\.venv\Scripts\Activate.ps1
        > ```

    - Windows Command Prompt:
        > ```bash
        > .\.venv\Scripts\activate.bat
        > ```

4. Install packages:

    > ```bash
    > pip install ipykernel numpy pandas scipy statsmodels scikit-learn scikit-fuzzy matplotlib seaborn
    > ```

# Snellius pipeline en notebook (zelfde volgorde)

De cluster pijplijn staat in `run_wang2025_snellius.slurm`. De uitvoer volgorde is hetzelfde idee als in `test_Wang2025_Luc.ipynb`: eerst TSE, dan EAT train, dan KNN, dan figuren. Alleen in het notebook staan stappen 1 tot en met 16 extra als uitleg (data, labels, EAT idee) vóór de echte pijplijn.

**Snellius projectmap:** `run_wang2025_snellius.slurm` en `run_wang2025_knn_eval.slurm` gebruiken nu **`~/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_stap_b`** (eigen checkpoints/results voor de huidige **Stap B**-default: EAT op ruwe mels, TSE in KNN `both`). De vorige map `eat_tse_knn_both` kun je op de cluster laten staan als archief.

**Op Snellius (login of dev node, bash):** nieuwe map + symlink naar je bestaande DCASE `raw` (pas `DATA_SRC` aan als jouw data elders staat).

```bash
BASE="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval"
NEW="$BASE/eat_tse_stap_b"
DATA_SRC="$BASE/Wang_EAT_Q4_Week_1/data/dcase2025t2/dev_data/raw"

mkdir -p "$NEW/data/dcase2025t2/dev_data"
ln -sfn "$DATA_SRC" "$NEW/data/dcase2025t2/dev_data/raw"
ls -la "$NEW/data/dcase2025t2/dev_data/raw"
mkdir -p "$NEW/logs" "$NEW/checkpoints/tse" "$NEW/results"
```

**Op je laptop (PowerShell):** upload alleen code (geen `data/`), bijvoorbeeld:

```powershell
$local = "C:\Users\lucth\Downloads\Sorama Internship\Wang_EAT_Q4_Week_1"
$stage = "$env:TEMP\eat_tse_stap_b_upload"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
robocopy $local $stage /E /XD data checkpoints results .git .ipynb_checkpoints __pycache__ .cursor /NFL /NDL /NJH /NJS /nc /ns /np
scp -r "$stage\*" "scur2597@snellius.surf.nl:~/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_stap_b/"
```

Daarna **weer op Snellius (bash):** controleer of `PROJECT_DIR` in de `.slurm` klopt, leeg **`checkpoints/eat_lora_system1`** voor een schone Stap B-run (of zet `auto-resume` uit), dan `cd "$NEW"` en `sbatch run_wang2025_snellius.slurm`.

| Slurm (STEP) | Script (kort) | Notebook (zelfde inhoud) |
|--------------|-----------------|-------------------------|
| STEP 1/4 | `train_tse_snellius.py` | Stappen 17 t/m 19, toggle `RUN_TSE` in het notebook is hetzelfde idee als `export RUN_TSE=0` of `1` vóór `sbatch` |
| STEP 2/4 | `train_wang2025_snellius.py` | Stap 20 (EAT LoRA training) |
| STEP 3/4 | `eval_wang2025_knn_snellius.py` | Stappen 21 t/m 22 (KNN, met of zonder TSE via `TSE_MODE` / `--tse-mode`) |
| STEP 4/4 | `plot_training_loss.py`, `plot_knn_*.py` | Zelfde plot idee: loss na training, KNN per machine, KNN over stappen |

**TSE uitzetten op de cluster:** vóór `sbatch` zet je `export RUN_TSE=0`. Dan sla je TSE training over en draait KNN eval alleen de baseline (geen TSE tak), net als in het notebook met `RUN_TSE = False`.

**Stap A vs Stap B (EAT en TSE hetzelfde “wereldje” houden):**

- **Stap A:** TSE ook **tijdens EAT-training** op de train-clips (`export EAT_TSE_IN_TRAINING=1` vóór `sbatch`). KNN met `TSE_MODE=both` blijft logisch als je overal enhanced audio gebruikt.
- **Stap B (slurm-default):** EAT traint op **ruwe** waveforms (`EAT_TSE_IN_TRAINING=0`, geen `--train-tse-checkpoint-dir`). TSE wordt nog wel getraind (als `RUN_TSE=1`) en alleen gebruikt in **KNN** met `TSE_MODE=both` (bank + test). LoRA leest dus ruwe mels; TSE is een vaste frontend alleen bij eval. Voor **nieuwe checkpoints** is dit vaak eenvoudiger te vergelijken met alleen-KNN-TSE. **Let op:** oude EAT-checkpoints die mét TSE in de train-loop zijn geleerd, passen niet bij Stap B; gebruik een lege `checkpoints/eat_lora_system1` (of zet `--auto-resume` tijdelijk uit) voor een schone Stap B-run.

**TSE modus in KNN:** in `run_wang2025_snellius.slurm` is `TSE_MODE` (standaard `both`) hetzelfde als `--tse-mode` in `eval_wang2025_knn_snellius.py`. `export TSE_MODE=test-only` is mogelijk maar geeft vaak bank/test mismatch.

**Plots na STEP 4:** `plot_knn_performance.py` schrijft één PNG met **aparte lijnen** per variant (baseline vs TSE). `plot_knn_per_machine.py` draait tweemaal: `knn_per_machine_baseline.png` en (als `RUN_TSE=1`) `knn_per_machine_tse.png`.

**Stappen vóór TSE in het notebook (1 tot en met 16):** uitleg en voorbereiding, geen exacte 1:1 regel in slurm, omdat op Snellius de data al op schijf staat en de job direct met TSE of EAT kan beginnen.

Zie de paper: Fujimura, T., Kuroyanagi, I., and Toda, T. (2025). The NU systems for DCASE 2025 Challenge Task 2. Technical report, Nagoya University.

# Officiele DCASE-ranking: train op Additional, test op Evaluation

Deze pijplijn hertraint **EAT-LoRA + TSE op de Additional dataset** en scoort daarna de
**Evaluation dataset**, zodat de officiele `dcase2025_task2_evaluator.py` een ranking-score
(AUC, pAUC, official score, precision/recall/F1) kan berekenen.

**Scripts (nieuw):**

| STEP (slurm) | Script | Wat |
|--------------|--------|-----|
| 1/5 | `train_tse_snellius.py` | TSE per machine, train-clips uit Additional dataset (uit met `RUN_TSE=0`) |
| 2/5 | `train_wang2025_snellius.py` | EAT-LoRA op Additional dataset (zelfde script, andere `--data-root`) |
| 3/5 | `infer_eval_dcase_snellius.py` | KNN-bank uit Additional/train, score Evaluation/test, schrijf DCASE-CSV's |
| 4/5 | `plot_training_loss.py` | loss-curves; score-histogrammen komen al uit STEP 3 |
| 5/5 | `dcase2025_task2_evaluator.py` | officiele scores + plots (`--out_all True`) |

`infer_eval_dcase_snellius.py` schrijft per machine/sectie twee headerloze CSV's in het
officiele submissieformaat naar `teams/<team>/<system>/`:

- `anomaly_score_<machine>_section_00_test.csv`  — `filename,score`
- `decision_result_<machine>_section_00_test.csv` — `filename,0|1`

De 0/1-drempel volgt de DCASE-baseline: een gamma-fit op de train-anomaliescores, 90e
percentiel (de beslissingen tellen **niet** mee voor de ranking, alleen voor precision/recall/F1).
De anomaliescore is de KNN min-cosine-afstand (1 − max cosine-similarity) tegen de bank.
TSE-modus via `--tse-mode` (`both` = bank + test enhanced, cluster-default).

## Op Snellius (one-shot via slurm)

```bash
BASE="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval"
NEW="$BASE/eat_tse_eval_dcase"
mkdir -p "$NEW/data" "$NEW/logs" "$NEW/checkpoints/tse" "$NEW/results"

# symlink je echte datamappen (pas de bron-paden aan):
ln -sfn /pad/naar/Additional_dataset "$NEW/data/additional"
ln -sfn /pad/naar/Evaluation_dataset "$NEW/data/evaluation"

# clone de officiele evaluator in het project:
git clone https://github.com/nttcslab/dcase2025_task2_evaluator.git "$NEW/dcase2025_task2_evaluator"

cd "$NEW"
sbatch run_eat_tse_eval_dcase.slurm
```

Toggles vóór `sbatch`: `export RUN_TSE=0` (geen TSE), `export TSE_MODE=off|test-only|both`,
`export EAT_TSE_IN_TRAINING=1` (Stap A), `export TEAM_NAME=... SYSTEM_NAME=...`.

## Resultaten

- `dcase2025_task2_evaluator/teams_result/<system>_result.csv` — AUC/pAUC/precision/recall/F1 per machine + **official score**.
- `dcase2025_task2_evaluator/teams_additional_result/` — aggregaten + anomaly-score plots (`--out_all True`).
- `results/eval_inference/anm_score_hist_<machine>.png` + `eval_inference_summary.csv` — score-verdeling en drempel per machine.

## Alleen de evaluator opnieuw draaien

Als de CSV's in `teams/<team>/<system>/` al bestaan:

```bash
cd dcase2025_task2_evaluator
python dcase2025_task2_evaluator.py --teams_root_dir ./teams --out_all True
# of: bash 03_evaluation_eval_data.sh
```
