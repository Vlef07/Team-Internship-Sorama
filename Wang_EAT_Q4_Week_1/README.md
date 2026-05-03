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

Dit is de setup waar we nu op draaien. Kort gezegd, EAT leert alleen nog maar op ruwe audio. TSE trainen we nog wel apart, maar tijdens het EAT-trainen zit TSE enhanced audio er niet in. Pas bij de kNN-test fase draait TSE wél op de golfvorm, en wel hetzelfde op de normale bank en op de test (both), zodat je geen scheve vergelijking krijgt.
Het hoofdscript is `run_wang2025_snellius.slurm`. Dat doet achter elkaar de volgende vier dingen.


1. TSE trainen per machine (`train_tse_snellius.py`). Dit leert de denoise-modellen, nog geen kNN evaluation.

2. EAT met LoRA trainen alleen op ruwe audio (train_wang2025_snellius.py`). Gewoon ruwe wav → mel.

3. kNN eval (`eval_wang2025_knn_snellius.py`). Per checkpoint, eerst zonder TSE (baseline), daarna met TSE op normal bank én test bank (TSE both). Zo kunnen we de baseline vs TSE (both) extension eerlijk vergelijken.

4. Plots uit de CSV’s (`plot_training_loss.py`, `plot_knn_per_machine.py`, `plot_knn_performance.py`).


Op de cluster verwachten de slurm-bestanden deze map: `~/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_stap_b`. Daar horen de code, `results/`, `checkpoints/` en een werkende `data/.../raw` bij (zelf kopiëren of symlinken naar je DCASE raw). Als `PROJECT_DIR` in de slurm ergens anders wijst, pas dat aan vóór je submit.

Wil je helemaal opnieuw trainen, maak `checkpoints/eat_lora_system1` leeg of zet tijdelijk auto-resume uit. Oude EAT-checkpoints die met TSE in de train-loop gemaakt zijn, horen niet bij deze workflow.
Alleen nog kNN draaien op bestaande checkpoints kan met `run_wang2025_knn_eval.slurm` (zelfde eval-script als stap 3).
Geen TSE nodig voor een run? Zet vóór sbatch `export RUN_TSE=0`. Dan sla je stap 1 over en valt de TSE-tak bij kNN weg.
Eenmalig een venv op Snellius: `setup_snellius.sh` en `requirements_snellius.txt`. TSE is gebaseerd op Fujimura et al. (2025), NU technical report DCASE 2025 Task 2.

In `test_Wang2025_Luc.ipynb` staat vooral conceptuele uitleg