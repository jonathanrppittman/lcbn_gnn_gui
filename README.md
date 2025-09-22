# GNN GUI (PyQt)

This app is a no-code front end to run your existing conversion (.mat -> .pt) and training scripts for graph neural networks, with optional SLURM submission.

## 1st Run

1. Create a venv and install requirements:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Set script paths in `config/default.yaml` (conversion and training). Optionally set SLURM defaults and environment activation.

3. Launch the GUI:

```bash
python src/main.py
```

## Notes
- The GUI executes your scripts; it does not replace them. You can provide extra CLI args in the text fields. Tokens supported in args:
  - `{inputs}`: space-separated input files
  - `{output_dir}`: chosen output directory
  - `{dataset_dir}`: selected dataset directory (.pt files)
  - `{model}`: selected model key from the dropdown
- Job scripts are saved under `jobs/` and logs under `logs/`.

## Subsequent Runs
1. 
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2. (optional) Set script paths in `config/default.yaml`

3. Launch the GUI:

```bash
python src/main.py

```