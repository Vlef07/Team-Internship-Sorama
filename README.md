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
