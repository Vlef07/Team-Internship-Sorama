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

---

## 2. Pair `.ipynb` with `.py`


* Convert notebook → script:

  ```bash
  jupytext --to py notebook.ipynb
  ```

* Convert script → notebook:

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

---

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

---

## 4. Working in VS Code

* Install the **Jupyter extension** from the VS Code marketplace.
* Open the paired `.py` file in VS Code.
* You’ll see **Run Cell** (`▶`) buttons above `# %%` sections. These behave like Jupyter cells.
* Outputs appear in the **Interactive Window**, just like a notebook.

## 5. Commit and push

**Please push only .py files to the remote and keep notebooks on your local machine. If you want you can skip notebooks altogether and use .py files instead (as shown in step 4). I have added all files with .ipynb extension to .gitignore so they will not be tracked.**

---

## 6. Synthetic data generator

The repository includes [synthesize_dcase_data.py](synthesize_dcase_data.py) for generating DCASE-style synthetic audio from your own train WAV files, together with matching `attributes_00.csv` files.

### What your input folder should look like

Point `--input-root` at the folder that contains your machine folders. The script will search recursively, so both of these layouts are fine:

```text
your_data_root/
  bearing/
    train/*.wav
    attributes_00.csv
  fan/
    train/*.wav
```

or:

```text
your_data_root/
  dev_bearing/
    bearing/
      train/*.wav
      attributes_00.csv
  dev_fan/
    fan/
      train/*.wav
      attributes_00.csv
```

### Basic usage

Run this once to create a small test output for one machine (in this case the valve):

```powershell
python synthesize_dcase_data.py --input-root "C:\Path\To\Your\DataRoot" --machines valve --output-root data/dcase2025t2/dev_data/raw_synth_small
```

Run this to generate data for every machine folder found under your input root:

```powershell
python synthesize_dcase_data.py --input-root "C:\Path\To\Your\DataRoot" --output-root data/dcase2025t2/dev_data/raw_synth
```

You can also set the environment variable `DCASE_INPUT_ROOT` once and then run the script without retyping the path:

```powershell
$env:DCASE_INPUT_ROOT = "C:\Path\To\Your\DataRoot"
python synthesize_dcase_data.py --output-root data/dcase2025t2/dev_data/raw_synth
```

### Output structure

The generated data follows the same shape expected by the training and evaluation scripts:

```text
machine/
  train/*.wav
  test/*.wav
  attributes_00.csv
```

### Parameters you can tune

`--machines`
: Process only specific machine folders, for example `--machines valve bearing`. Leave it out to process everything found under the input root.

`--copies-per-source`
: Controls how many derived triplets are created from each source train clip. Higher values make more output data.

`--sample-rate`
: Resamples source audio to this rate before synthesis. The default is `16000`, which matches the training pipeline.

`--duration-sec`
: Target length in seconds for each generated clip. The script trims or repeats source audio to match this duration.

`--seed`
: Makes the synthesis reproducible. Use the same seed to regenerate the same outputs.

`--output-root`
: Where the generated dataset is written.

`--input-root` / `--template-root`
: The folder containing your source machine data. `--template-root` is kept for compatibility, but `--input-root` is the preferred name.

### Recommended starting point

If you are trying the script for the first time, start with one machine and a small output folder:

```powershell
python synthesize_dcase_data.py --input-root "C:\Path\To\Your\DataRoot" --machines valve --copies-per-source 1 --output-root data/dcase2025t2/dev_data/raw_synth_small
```

Once that works, remove `--machines` to generate data for all machine types.

---

## 7. Sorama synthetic data generator

The repository also includes [synthesize_sorama_data.py](synthesize_sorama_data.py) for generating synthetic Sorama-style data from existing Sorama WAV files.

### Supported input layouts

Pass one or more roots with `--input-roots`. Each root can be either:

```text
RootA/
  Pump/
  Bearing/
```

or:

```text
RootA/
  Data Sorama/
    Pump/
    Bearing/
```

Inside `Pump/` and `Bearing/`, the script preserves your existing folder layout, for example `Asset 1/0_normal` and `Asset 1/1_anomaly`.

### Dry-run first (recommended)

```powershell
python synthesize_sorama_data.py `
  --input-roots "C:\Path\To\DataRootA" "C:\Path\To\DataRootB" `
  --output-root "C:\Path\To\DataSorama_Synthetic" `
  --prefix-with-root-name `
  --copies-per-source 1 `
  --dry-run
```

### Full generation

```powershell
python synthesize_sorama_data.py `
  --input-roots "C:\Path\To\DataRootA" "C:\Path\To\DataRootB" "C:\Path\To\DataRootC" `
  --output-root "C:\Path\To\DataSorama_Synthetic" `
  --prefix-with-root-name `
  --copies-per-source 1
```

### Output behavior

- The directory structure under `Pump/` and `Bearing/` is preserved.
- For each source WAV, generated files are written with `_synXXX.wav` suffixes.
- Normal source clips (`0_normal`) generate normal-like synthetic variants.
- Anomalous source clips (`1_anomaly`) generate anomaly-like synthetic variants.

### Parameters you can tune

`--copies-per-source`
: Number of generated files per source file.

`--sample-rate`
: Target sample rate for output files. Use `0` to keep the original sample rate of each source file.

`--seed`
: Reproducibility seed.

`--prefix-with-root-name`
: If set, each input root is written to `output-root/<input_root_name>/...` to avoid collisions when combining multiple roots.

`--dry-run`
: Prints how many files would be generated without writing output.

### Notes

- Some datasets contain malformed WAV files; the script skips unreadable files with a warning.
- Always run `--dry-run` first when working with large roots.


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
