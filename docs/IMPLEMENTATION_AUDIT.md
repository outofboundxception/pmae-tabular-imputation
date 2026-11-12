# Implementation audit and integration changes

This audit records the integration problems found in the uploaded repository and the changes made in this revision.

## Problems found in the original repository

1. **The code and notebooks did not form one reproducible workflow.** The main run notebook loaded the historical Diabetes pickle files using a `full` / `quasi` pattern. The separate `ampute_dataset.ipynb` expected a `data_proc/` directory that was not included and generated MCAR / MAR / MNAR artifacts under a different schema.
2. **The original run notebook did not initialize all model configuration fields.** `get_args_parser().parse_args([])` leaves some architecture fields as `None`; the notebook did not clearly apply the dataset profile before constructing the models.
3. **The original block implementation imported `timm.layers` at runtime.** The model used only a small subset of that library, and `timm` was not listed in the uploaded dependency file.
4. **The original PMAE trainer used `random` without importing it in its seed helper.** The notebook happened to seed through another imported helper, which hid the standalone issue.
5. **The original trainer did not safely handle all constant/all-missing features.** An all-missing column has no usable minimum/maximum; constant columns also need safe inverse scaling.
6. **The original ReMasker loss could treat originally missing cells, represented internally as zero, as training targets.** The loss is now gated by the original observed-value mask.
7. **The original ReMasker inference loop processed one row at a time.** Inference is now batched/vectorized.
8. **The original evaluator assumed particular numerical/categorical slices and did not provide a general missing-cell-only metric report.** The new evaluator handles numeric-only and mixed matrices and scores only the cells hidden by the artificial mask.
9. **There was no reusable dataset registry or generic loader.** Dataset-specific CSV and Excel layouts were not connected to the PMAE training API.

## What changed

- Moved the maintained implementation into `src/pmae_tabular/` with subpackages for data, training, legacy utilities, and evaluation.
- Kept root-level compatibility modules (`pMAE.py`, `fit_pMAE.py`, `utils.py`, and others) so existing import statements continue to work.
- Kept the PMAE network structure and proportional masking logic. Implemented the required `Mlp` and `DropPath` helpers locally, removing the `timm` runtime dependency.
- Added a registry and reader for 11 dataset variants: Online News, Adult, Credit Default, Online Shoppers, Letter Recognition, Bike Daily, Bike Hourly, Obesity, Wine Red, Wine White, and Diabetes.
- Added consistent feature ordering: numerical columns first, ordinal-encoded categorical columns last. Targets and identifiers are excluded from the imputation matrix and preserved separately where available.
- Added generic preprocessing that records fill counts, numerical fill values, categorical fill values, category mappings, feature metadata, and a cell-level mask for values missing in the source dataset.
- Made amputation source-missing-aware: the artificial mask never labels an originally unknown value as evaluation ground truth; source-missing values remain missing in model input and are excluded from metric calculations.
- Added a single MCAR / MAR / MNAR generator with fixed-seed outputs and a consistent boolean mask convention (`True` = artificially missing).
- Added an experiment runner that trains PMAE and/or ReMasker and saves metrics, both encoded and category-decoded imputed data, histories, model state dictionaries, and configuration metadata.
- Regenerates the amputation artifact whenever preprocessing is force-refreshed, preventing stale masks/data from being paired with a new complete matrix.
- Added clean-checkout tests that build their processed datasets in a temporary directory rather than relying on generated files being committed.
- Updated notebooks to call the same package modules as the CLI. The historical Diabetes data is kept under `data/legacy/diabetes/`.

## Validation performed in the build environment

The available validation environment used Python 3.13.5 and PyTorch 2.10.0+cpu, along with NumPy, Pandas, SciPy, scikit-learn and einops; it did not have `timm`, which is why removal of the timm import was verified directly. The project metadata intentionally targets Python 3.9–3.12 for compatibility with the pinned PyTorch 2.2.1 requirements. The suite was exercised in the available Python 3.13 runtime, but a full dependency installation in the target Python range was not performed in this environment.

- `python run_experiment.py prepare-all`: passed for all 11 dataset variants.
- MCAR, MAR and MNAR generation: passed smoke checks on all prepared matrices.
- Mixed-type dataset preprocessing: checked Adult, Credit Default, Online Shoppers, Bike Sharing and Obesity.
- PMAE and ReMasker one-epoch smoke training on synthetic numeric input: passed.
- End-to-end PMAE and ReMasker smoke runs on Wine Red (numeric) and Obesity (mixed): passed at two epochs.
- Compatibility imports from the original root module names: passed.
- `python -m compileall`: passed.
- `pytest -q`: **7 passed** (including preparation from all 11 raw dataset variants in a clean temporary output directory, source-missing-mask behavior, constant-feature handling, and legacy import shims).
- Short end-to-end smoke runs of both PMAE and ReMasker: passed on Wine Red (numeric-only), Obesity (mixed numeric/categorical), and Adult (mixed data with source-missing values); verified decoded category labels, encoded-code exports, and model checkpoints containing normalization metadata.

These are software integration checks, not full research experiments. The default 300-epoch runs across every dataset have not been executed. The historical Diabetes scores are therefore retained as historical reference values and are not claimed as results from this revision.

## Known modeling limitation

The original PMAE decoder is scalar-valued. Categorical columns are currently ordinal encoded and reconstructed as scalar values, then rounded/clipped for categorical accuracy. This makes the pipeline able to run on mixed datasets, but it does not make the model a native categorical imputer. A categorical embedding/decoder and categorical loss would be a separate architecture-level extension and should be evaluated independently.
