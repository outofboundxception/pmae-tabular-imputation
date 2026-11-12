from pathlib import Path

import numpy as np
from pmae_tabular.data.amputation import generate_missingness
from pmae_tabular.data.preprocessing import prepare_dataset
from pmae_tabular.data.registry import DATASET_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"


def test_all_registered_datasets_prepare_from_raw(tmp_path):
    """A clean checkout should be testable without committed processed files."""
    for name in DATASET_SPECS:
        payload = prepare_dataset(name, RAW_ROOT, tmp_path / "processed", force=True)
        X = payload["X"]
        assert X.ndim == 2
        assert X.shape[1] == len(payload["columns"])
        assert np.isfinite(X).all()
        assert 0 <= payload["d_numerical"] <= X.shape[1]
        assert X.shape[1] >= 2
        source_missing = payload["preexisting_missing_mask"]
        assert source_missing.shape == X.shape
        assert source_missing.dtype == bool
    adult = prepare_dataset("adult", RAW_ROOT, tmp_path / "adult-check", force=True)
    assert adult["preexisting_missing_mask"].sum() > 0


def test_preexisting_missing_values_are_not_synthetic_targets(tmp_path):
    adult = prepare_dataset("adult", RAW_ROOT, tmp_path / "processed", force=True)
    source_missing = adult["preexisting_missing_mask"]
    incomplete, artificial_mask, metadata = generate_missingness(
        adult["X"], "MCAR", column_fraction=0.4, missing_rate=0.25,
        seed=23, eligible_mask=~source_missing,
    )
    incomplete[source_missing] = np.nan
    assert not np.any(artificial_mask & source_missing)
    assert np.isnan(incomplete).sum() == artificial_mask.sum() + source_missing.sum()
    assert metadata["preexisting_missing_cells"] == int(source_missing.sum())


def test_amputation_mask_is_reproducible_and_consistent():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    incomplete_a, mask_a, meta_a = generate_missingness(X, "MAR", 0.4, 0.3, 19)
    incomplete_b, mask_b, meta_b = generate_missingness(X, "MAR", 0.4, 0.3, 19)
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(np.isnan(incomplete_a), mask_a)
    assert np.array_equal(incomplete_a, incomplete_b, equal_nan=True)
    assert meta_a == meta_b
    assert mask_a.any()
    assert not mask_a.all(axis=0).any()


def test_all_three_mechanisms_generate_missing_cells():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    for mechanism in ("MCAR", "MAR", "MNAR"):
        incomplete, mask, meta = generate_missingness(X, mechanism, 0.5, 0.4, 5)
        assert mask.sum() > 0
        assert np.isnan(incomplete).sum() == mask.sum()
        assert meta["mechanism"] == mechanism


def test_mar_requires_at_least_one_observed_driver_column():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 6)).astype(np.float32)
    try:
        generate_missingness(X, "MAR", column_fraction=1.0, missing_rate=0.3, seed=4)
    except ValueError as exc:
        assert "MAR requires at least one originally observed feature column outside" in str(exc)
    else:
        raise AssertionError("MAR should reject selecting every feature for missingness.")
