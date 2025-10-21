# GNN GUI (PyQt)

This app is a no-code front end to run your existing conversion (.mat -> .pt) and training scripts for graph neural networks, with optional SLURM submission.
It is mostly a project being built by myself for our lab's use and to explore agentic development (with Google's Jules agent @ URL: jules.google).
## 1st Run (macOS/Linux)

1. Create a venv and install requirements:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Set script paths in `config/default.yaml` (conversion and training). Optionally set SLURM defaults and environment activation.

3. Launch the GUI:

```bash
python src/main.py &
```

## 1st Run (Windows)

1. Create a venv and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Set script paths in `config/default.yaml` (conversion and training).

3. Launch the GUI:

```bash
python src/main.py
```

## Subsequent Runs (macOS/Linux)
1. 
```bash
source .venv/bin/activate
```
2. (optional) Set script paths in `config/default.yaml`

3. Launch the GUI:

```bash
python src/main.py &

```

## Subsequent Runs (Windows)
1.
```bash
.venv\Scripts\activate
```
2. (optional) Set script paths in `config\default.yaml`

3. Launch the GUI:

```bash
python src\main.py
```

## Running Tests

To run the test suite, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then, run pytest from the root of the repository:

```bash
python3 -m pytest
```

**Note:** Due to limitations in the sandboxed testing environment, tests that require a Qt Application (`pytest-qt`) may not run correctly. The current test suite focuses on core, non-GUI logic.

## Notes
- The GUI executes your scripts; it does not replace them. You can provide extra CLI args in the text fields. Tokens supported in args:
  - `{inputs}`: space-separated input files
  - `{output_dir}`: chosen output directory
  - `{dataset_dir}`: selected dataset directory (.pt files)
  - `{model}`: selected model key from the dropdown
- Job scripts are saved under `jobs/` and logs under `logs/`.
