# Data directories

- `raw/`: bundled source files. Do not modify these files in place.
- `processed/`: generated finite feature matrices, targets, metadata, and cell-level masks identifying values missing in the original source data. Rebuild with `python run_experiment.py prepare --dataset <key> --force`.
- `amputated/`: generated model-input matrices with artificial missingness and their evaluation masks. Source-missing values remain missing in model input but are not included in the artificial evaluation mask. Regenerate with `python run_experiment.py amputate ... --force`.
- `legacy/diabetes/`: pre-generated artifacts retained from the original single-dataset experiment.

The processed and amputated directories are intentionally git-ignored because they are derived artifacts. Their schemas and generation commands are documented in the root README.
