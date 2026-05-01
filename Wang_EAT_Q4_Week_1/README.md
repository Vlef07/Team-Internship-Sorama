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

**Snellius projectmap:** de slurm-scripts verwachten `~/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_knn_both`. Maak die map op de login node, zet `data/.../raw` als **symlink** naar je bestaande DCASE-map (zie commando's hieronder), en vul de map met de **huidige** repo-inhoud vanaf je laptop (`scp`, `rsync`, of `git clone`), zodat `run_wang2025_snellius.slurm` en `setup_snellius.sh` de nieuwe paden bevatten.

Op de Snellius login node (**bash**):

```bash
BASE="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval"
NEW="$BASE/eat_tse_knn_both"
DATA_SRC="$BASE/Wang_EAT_Q4_Week_1/data/dcase2025t2/dev_data/raw"
mkdir -p "$NEW/data/dcase2025t2/dev_data"
ln -sfn "$DATA_SRC" "$NEW/data/dcase2025t2/dev_data/raw"
ls -la "$NEW/data/dcase2025t2/dev_data/raw"
```

Daarna: kopieer alle bestanden uit `Wang_EAT_Q4_Week_1` naar `$NEW`, **maar sla `data/`, `checkpoints/`, `results/` over** (data is al de symlink). Ga naar `$NEW` en start met `sbatch run_wang2025_snellius.slurm`.

| Slurm (STEP) | Script (kort) | Notebook (zelfde inhoud) |
|--------------|-----------------|-------------------------|
| STEP 1/4 | `train_tse_snellius.py` | Stappen 17 t/m 19, toggle `RUN_TSE` in het notebook is hetzelfde idee als `export RUN_TSE=0` of `1` vóór `sbatch` |
| STEP 2/4 | `train_wang2025_snellius.py` | Stap 20 (EAT LoRA training) |
| STEP 3/4 | `eval_wang2025_knn_snellius.py` | Stappen 21 t/m 22 (KNN, met of zonder TSE via `TSE_MODE` / `--tse-mode`) |
| STEP 4/4 | `plot_training_loss.py`, `plot_knn_*.py` | Zelfde plot idee: loss na training, KNN per machine, KNN over stappen |

**TSE uitzetten op de cluster:** vóór `sbatch` zet je `export RUN_TSE=0`. Dan sla je TSE training over en draait KNN eval alleen de baseline (geen TSE tak), net als in het notebook met `RUN_TSE = False`.

**TSE modus in KNN:** in `run_wang2025_snellius.slurm` is `TSE_MODE` (standaard `both`: TSE op KNN-trainbank én test, zelfde `TSE_DIR` als bij EAT-training) hetzelfde principe als `--tse-mode` in `eval_wang2025_knn_snellius.py` en de variabele `TSE_MODE` in het notebook. Je kunt `export TSE_MODE=test-only` zetten vóór `sbatch` voor een oude vergelijking; `both` voorkomt bank/test mismatch.

**Plots na STEP 4:** `plot_knn_performance.py` schrijft één PNG met **aparte lijnen** per variant (baseline vs TSE). `plot_knn_per_machine.py` draait tweemaal: `knn_per_machine_baseline.png` en (als `RUN_TSE=1`) `knn_per_machine_tse.png`.

**Stappen vóór TSE in het notebook (1 tot en met 16):** uitleg en voorbereiding, geen exacte 1:1 regel in slurm, omdat op Snellius de data al op schijf staat en de job direct met TSE of EAT kan beginnen.

Zie de paper: Fujimura, T., Kuroyanagi, I., and Toda, T. (2025). The NU systems for DCASE 2025 Challenge Task 2. Technical report, Nagoya University.
