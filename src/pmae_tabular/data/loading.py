"""Raw dataset readers. Targets and identifiers are kept out of imputation inputs."""
from pathlib import Path
from typing import Optional
import pandas as pd

from .registry import DATASET_SPECS, canonical_dataset_name

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
LETTER_COLUMNS = ["letter"] + [f"x{i}" for i in range(1, 17)]


def _read_adult(folder: Path) -> pd.DataFrame:
    train = pd.read_csv(folder / "adult.data", header=None, names=ADULT_COLUMNS,
                        skipinitialspace=True, na_values=["?"], comment="|", encoding="latin-1")
    test = pd.read_csv(folder / "adult.test", header=None, names=ADULT_COLUMNS,
                       skipinitialspace=True, na_values=["?"], comment="|", skiprows=1,
                       encoding="latin-1")
    combined = pd.concat([train, test], ignore_index=True)
    combined["income"] = combined["income"].astype("string").str.strip().str.rstrip(".")
    return combined


def _read_credit(folder: Path, files: tuple) -> pd.DataFrame:
    xlsx = folder / files[0]
    if xlsx.exists() and xlsx.suffix.lower() == ".xlsx":
        frame = pd.read_excel(xlsx, header=1, engine="openpyxl")
    else:
        xls = folder / files[-1]
        try:
            frame = pd.read_excel(xls, header=1)
        except ImportError as exc:
            raise RuntimeError(
                "Reading the original .xls requires xlrd. The repository also includes an .xlsx conversion; "
                "make sure it is present under data/raw/default_credit_card_clients/."
            ) from exc
    return frame


def load_raw_dataset(name: str, data_root: Path):
    """Return (features_frame, target_frame_or_none, source_metadata)."""
    key = canonical_dataset_name(name)
    spec = DATASET_SPECS[key]
    data_root = Path(data_root)

    preloaded_target = None
    if spec.format == "sklearn":
        from sklearn.datasets import load_diabetes
        bunch = load_diabetes(as_frame=True)
        raw = bunch.data.copy()
        preloaded_target = pd.DataFrame({"target": bunch.target})
        source_files = ["sklearn.datasets.load_diabetes"]
    else:
        folder = data_root / spec.folder
        missing_files = [f for f in spec.files if not (folder / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Dataset '{key}' is missing files in {folder}: {missing_files}")

        if spec.format == "adult":
            raw = _read_adult(folder)
        elif spec.format == "credit":
            raw = _read_credit(folder, spec.files)
        elif spec.format == "letter":
            raw = pd.read_csv(folder / spec.files[0], header=None, names=LETTER_COLUMNS)
        elif spec.format == "semicolon":
            raw = pd.read_csv(folder / spec.files[0], sep=";")
        elif spec.format == "bike":
            raw = pd.read_csv(folder / spec.files[0])
        else:
            raw = pd.read_csv(folder / spec.files[0])
        source_files = list(spec.files)

    raw.columns = [str(c).strip() for c in raw.columns]
    # Normalize whitespace in string columns and common textual missing markers.
    for col in raw.select_dtypes(include=["object", "string"]).columns:
        raw[col] = raw[col].astype("string").str.strip()
        raw[col] = raw[col].replace({"?": pd.NA, "": pd.NA, "NA": pd.NA, "N/A": pd.NA})

    if preloaded_target is not None:
        target = preloaded_target.copy()
        targets = target.columns.tolist()
    else:
        missing_targets = [c for c in spec.target_columns if c not in raw.columns]
        missing_ids = [c for c in spec.id_columns if c not in raw.columns]
        missing_categorical = [c for c in spec.categorical_columns if c not in raw.columns]
        if missing_targets:
            raise ValueError(f"Dataset '{key}' is missing configured target columns: {missing_targets}")
        if missing_ids:
            raise ValueError(f"Dataset '{key}' is missing configured identifier columns: {missing_ids}")
        if missing_categorical:
            raise ValueError(f"Dataset '{key}' is missing configured categorical columns: {missing_categorical}")
        targets = list(spec.target_columns)
        target = raw[targets].copy() if targets else None
    ids = list(spec.id_columns)
    features = raw.drop(columns=targets + ids, errors="ignore").copy()

    if features.empty:
        raise ValueError(f"No feature columns remain for dataset '{key}'.")
    if features.columns.duplicated().any():
        duplicated = features.columns[features.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature names in '{key}': {duplicated}")

    metadata = {
        "dataset": key,
        "description": spec.description,
        "source_files": source_files,
        "n_rows": int(len(raw)),
        "n_raw_columns": int(raw.shape[1]),
        "excluded_target_columns": targets,
        "excluded_identifier_columns": ids,
        "configured_categorical_columns": [c for c in spec.categorical_columns if c in features.columns],
    }
    return features.reset_index(drop=True), target.reset_index(drop=True) if target is not None else None, metadata
