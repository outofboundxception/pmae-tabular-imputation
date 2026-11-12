"""Deterministic preprocessing for the scalar-feature PMAE implementation."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .loading import load_raw_dataset
from .registry import DATASET_SPECS, canonical_dataset_name


def _mode_or_token(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else "__MISSING__"


def prepare_dataset(name: str, data_root: Path, processed_root: Path, force: bool = False):
    """Load, clean, encode, and persist a dataset in PMAE's expected column order.

    Numerical columns come first; ordinal-encoded categorical columns come last.
    The original target and identifier columns are deliberately not model inputs.
    """
    key = canonical_dataset_name(name)
    out_dir = Path(processed_root) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = out_dir / "processed.pkl"
    metadata_path = out_dir / "metadata.json"
    if pickle_path.exists() and metadata_path.exists() and not force:
        with pickle_path.open("rb") as stream:
            return pickle.load(stream)

    frame, target, source_meta = load_raw_dataset(key, Path(data_root))
    spec = DATASET_SPECS[key]
    configured_cats = set(spec.categorical_columns)
    categorical_cols = [
        col for col in frame.columns
        if col in configured_cats or pd.api.types.is_object_dtype(frame[col])
        or pd.api.types.is_string_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col])
    ]
    numerical_cols = [c for c in frame.columns if c not in categorical_cols]

    numeric_data = pd.DataFrame(index=frame.index)
    numeric_fill_values: Dict[str, float] = {}
    missing_by_column = {}
    dropped_all_missing_columns = []
    for col in numerical_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_count = int(values.isna().sum())
        missing_by_column[col] = values.isna().to_numpy(dtype=bool)
        if values.notna().sum() == 0:
            dropped_all_missing_columns.append(col)
            source_meta.setdefault("preexisting_missing_counts", {})[col] = missing_count
            continue
        median = float(values.median())
        if not np.isfinite(median):
            median = 0.0
        numeric_fill_values[col] = median
        numeric_data[col] = values.fillna(median).astype(np.float32)
        source_meta.setdefault("preexisting_missing_counts", {})[col] = missing_count

    categorical_data = pd.DataFrame(index=frame.index)
    category_mappings: Dict[str, list] = {}
    categorical_fill_values: Dict[str, str] = {}
    for col in categorical_cols:
        values = frame[col].astype("string").str.strip()
        missing_by_column[col] = values.isna().to_numpy(dtype=bool)
        if values.notna().sum() == 0:
            dropped_all_missing_columns.append(col)
            source_meta.setdefault("preexisting_missing_counts", {})[col] = int(values.isna().sum())
            continue
        fill_value = str(_mode_or_token(values.dropna()))
        categorical_fill_values[col] = fill_value
        values = values.fillna(fill_value).astype(str)
        categories = sorted(values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        categorical_data[col] = values.map(mapping).astype(np.float32)
        category_mappings[col] = categories
        source_meta.setdefault("preexisting_missing_counts", {})[col] = int(frame[col].isna().sum())

    # Drop unusable all-missing columns from the model matrix and its type
    # metadata together so numerical/categorical indices remain aligned.
    numerical_cols = [c for c in numerical_cols if c not in dropped_all_missing_columns]
    categorical_cols = [c for c in categorical_cols if c not in dropped_all_missing_columns]

    # Preserve feature type blocks: numerical first, categorical last.
    processed_frame = pd.concat([numeric_data, categorical_data], axis=1)
    if processed_frame.isna().any().any() or not np.isfinite(processed_frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Preprocessing '{key}' produced non-finite values.")
    if processed_frame.shape[1] < 2:
        raise ValueError(f"PMAE needs at least two feature columns; '{key}' has {processed_frame.shape[1]}.")

    X = processed_frame.to_numpy(dtype=np.float32)
    preexisting_missing_mask = np.column_stack(
        [missing_by_column[col] for col in processed_frame.columns]
    ).astype(bool)
    payload = {
        "dataset": key,
        "X": X,
        "preexisting_missing_mask": preexisting_missing_mask,
        "columns": processed_frame.columns.tolist(),
        "d_numerical": int(len(numerical_cols)),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "category_mappings": category_mappings,
        "categorical_sizes": {c: len(v) for c, v in category_mappings.items()},
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "dropped_all_missing_columns": dropped_all_missing_columns,
        "target": target,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "source_metadata": source_meta,
    }
    with pickle_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    serializable_meta = {
        k: v for k, v in payload.items()
        if k not in {"X", "target", "preexisting_missing_mask"}
    }
    serializable_meta["preexisting_missing_mask_shape"] = list(preexisting_missing_mask.shape)
    serializable_meta["preexisting_missing_cells"] = int(preexisting_missing_mask.sum())
    serializable_meta["target_columns"] = target.columns.tolist() if target is not None else []
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(serializable_meta, stream, indent=2, ensure_ascii=False, default=str)
    return payload


def load_processed_dataset(name: str, processed_root: Path):
    key = canonical_dataset_name(name)
    path = Path(processed_root) / key / "processed.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}. Run `prepare` first.")
    with path.open("rb") as stream:
        return pickle.load(stream)
