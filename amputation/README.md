# Amputation moved into the reusable package

The old one-off `ampute_dataset.ipynb` expected an external `data_proc/` folder that was not included in the repository. Its hard-coded workflow has been replaced by the configurable generator at `src/pmae_tabular/data/amputation.py`.

Use `python run_experiment.py amputate --dataset <dataset-key> --mechanism MCAR|MAR|MNAR` or open `notebooks/01_Prepare_Datasets.ipynb`.

The saved boolean `mask` marks only artificially hidden cells used for evaluation. Values already missing in the raw source remain missing in the model input, but are excluded from the synthetic evaluation mask.
