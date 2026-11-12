"""Generic, reproducible synthetic missingness generation (MCAR, MAR, MNAR)."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.special import expit


def _standardize_column(values: np.ndarray, observed: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if observed is None:
        observed = np.ones(values.shape[0], dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    mean = float(np.mean(values[observed])) if observed.any() else 0.0
    std = float(np.std(values[observed])) if observed.any() else 0.0
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    standardized = (values - mean) / std
    # Pre-existing unknown values are not used to define drivers; set their
    # standardized value to the mean (zero) for stable MAR weighting.
    standardized[~observed] = 0.0
    return standardized


def _weighted_rows(rng: np.random.Generator, weights: np.ndarray, count: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()
    return rng.choice(len(weights), size=count, replace=False, p=weights)


def _target_column_mask(
    rng: np.random.Generator,
    scores: np.ndarray,
    missing_rate: float,
    eligible_rows: np.ndarray,
) -> np.ndarray:
    eligible_rows = np.asarray(eligible_rows, dtype=int)
    n_eligible = len(eligible_rows)
    count = int(round(n_eligible * missing_rate))
    if n_eligible >= 2:
        count = min(max(count, 1), n_eligible - 1)
    else:
        count = 0  # leave a sole observed value available to train the model
    selected_local = (
        _weighted_rows(rng, np.asarray(scores)[eligible_rows], count)
        if count else np.array([], dtype=int)
    )
    mask = np.zeros(len(scores), dtype=bool)
    if count:
        mask[eligible_rows[selected_local]] = True
    return mask


def generate_missingness(
    X: np.ndarray,
    mechanism: str = "MCAR",
    column_fraction: float = 0.3,
    missing_rate: float = 0.5,
    seed: int = 42,
    eligible_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return incomplete matrix, artificial-missing mask and metadata.

    `column_fraction` selects the fraction of eligible feature columns that can
    receive artificial missingness. `missing_rate` is applied to each selected
    column's originally observed cells. `eligible_mask=True` means a value was
    present in the source dataset. Existing source NAs are never labeled as
    synthetic evaluation targets. MAR uses other observed features as drivers;
    MNAR uses each selected feature's own observed values as its driver.
    """
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("X must be a 2D matrix with at least one row and two columns.")
    if not np.isfinite(values).all():
        raise ValueError("Amputation expects a finite preprocessed matrix. Run preprocessing first.")
    if not 0.0 < column_fraction <= 1.0:
        raise ValueError("column_fraction must be in (0, 1].")
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must be in (0, 1).")
    mechanism = mechanism.upper()
    if mechanism not in {"MCAR", "MAR", "MNAR"}:
        raise ValueError("mechanism must be one of MCAR, MAR, MNAR.")

    if eligible_mask is None:
        eligible = np.ones(values.shape, dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != values.shape:
            raise ValueError("eligible_mask must have the same shape as X.")
    eligible_counts = eligible.sum(axis=0)
    candidate_columns = np.flatnonzero(eligible_counts >= 2)
    if candidate_columns.size == 0:
        raise ValueError("No feature has at least two originally observed values to amputate.")

    rng = np.random.default_rng(seed)
    n_rows, n_cols = values.shape
    n_selected = max(1, min(candidate_columns.size, int(math.ceil(column_fraction * n_cols))))
    selected_columns = np.sort(rng.choice(candidate_columns, size=n_selected, replace=False))
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    available_drivers = np.array(
        [j for j in candidate_columns if j not in set(selected_columns)], dtype=int
    )
    if mechanism == "MAR" and available_drivers.size == 0:
        raise ValueError(
            "MAR requires at least one originally observed feature column outside "
            "the selected missing columns. Reduce --column-fraction."
        )

    # Standardize potential MAR drivers using only source-observed values.
    standardized = np.column_stack([
        _standardize_column(values[:, j], eligible[:, j]) for j in range(n_cols)
    ])

    for col in selected_columns:
        eligible_rows = np.flatnonzero(eligible[:, col])
        if mechanism == "MCAR":
            weights = np.ones(n_rows, dtype=np.float64)
        elif mechanism == "MAR":
            n_drivers = min(5, len(available_drivers))
            drivers = rng.choice(available_drivers, size=n_drivers, replace=False)
            coefficients = rng.normal(0.0, 1.0, size=n_drivers)
            coefficients /= max(np.linalg.norm(coefficients), 1e-12)
            logits = standardized[:, drivers] @ coefficients
            weights = expit(logits)
        else:  # MNAR: self-masking based only on originally observed target values.
            direction = rng.choice([-1.0, 1.0])
            logits = direction * standardized[:, col]
            weights = expit(logits)

        mask[:, col] = _target_column_mask(rng, weights, missing_rate, eligible_rows)

    incomplete = values.copy()
    incomplete[mask] = np.nan
    actual_rate = float(mask.mean())
    eligible_cell_count = int(eligible.sum())
    metadata = {
        "mechanism": mechanism,
        "column_fraction_requested": float(column_fraction),
        "selected_columns": selected_columns.tolist(),
        "selected_column_count": int(n_selected),
        "missing_rate_per_selected_column_requested": float(missing_rate),
        "overall_artificial_missing_fraction": actual_rate,
        "artificial_missing_fraction_of_eligible_cells": (
            float(mask.sum() / eligible_cell_count) if eligible_cell_count else 0.0
        ),
        "artificial_missing_cells": int(mask.sum()),
        "preexisting_missing_cells": int((~eligible).sum()),
        "seed": int(seed),
        "shape": [int(n_rows), int(n_cols)],
        "mask_convention": "True means artificially missing; original source missing values are excluded from this mask.",
        "mar_definition": "Missingness probabilities depend on other, non-selected, originally observed features.",
        "mnar_definition": "Self-masking logistic mechanism; missingness depends on the observed value being hidden.",
    }
    return incomplete, mask, metadata


def amputation_path(amputated_root: Path, dataset: str, mechanism: str,
                     column_fraction: float, missing_rate: float, seed: int) -> Path:
    return (Path(amputated_root) / dataset / mechanism.upper()
            / f"columns_{column_fraction:.2f}" / f"rate_{missing_rate:.2f}" / f"seed_{seed}.npz")


def save_amputation(path: Path, incomplete: np.ndarray, mask: np.ndarray, metadata: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_incomplete=np.asarray(incomplete, dtype=np.float32),
                        mask=np.asarray(mask, dtype=np.bool_))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_amputation(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        incomplete = stored["X_incomplete"].astype(np.float32)
        mask = stored["mask"].astype(bool)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if incomplete.shape != mask.shape:
        raise ValueError(f"Inconsistent imputation input/mask shapes in {path}.")
    if np.any(mask & np.isfinite(incomplete)):
        raise ValueError(f"Artificial mask and incomplete data disagree in {path}.")
    if np.any((~mask) & np.isnan(incomplete)):
        # NaNs outside the synthetic mask are allowed for source-missing cells;
        # the original mask lives in processed metadata and remains unevaluated.
        if "preexisting_missing_cells" not in metadata:
            raise ValueError(f"Untracked NaNs outside the artificial mask in {path}.")
    return incomplete, mask, metadata
