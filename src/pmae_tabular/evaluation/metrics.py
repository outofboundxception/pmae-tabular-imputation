"""Metrics for evaluating only the artificially hidden entries."""
from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance


def evaluate_imputation(X_true, X_pred, mask, d_numerical: int, categorical_sizes=None):
    truth = np.asarray(X_true, dtype=np.float64)
    pred = np.asarray(X_pred, dtype=np.float64)
    missing = np.asarray(mask, dtype=bool)
    if truth.shape != pred.shape or truth.shape != missing.shape:
        raise ValueError("X_true, X_pred and mask must have identical shapes.")
    if missing.sum() == 0:
        raise ValueError("The missingness mask contains no hidden cells to evaluate.")
    if not np.isfinite(pred).all():
        raise ValueError("Imputed output contains NaN or infinite values.")
    d = truth.shape[1]
    d_numerical = int(d_numerical)
    if not 0 <= d_numerical <= d:
        raise ValueError("d_numerical must be between 0 and the feature count.")
    categorical_sizes = categorical_sizes or {}

    metrics = {
        "n_samples": int(truth.shape[0]),
        "n_features": int(d),
        "n_missing": int(missing.sum()),
        "missing_fraction": float(missing.mean()),
    }
    norm_truth = truth.copy()
    norm_pred = pred.copy()
    for j in range(d):
        lo = float(np.min(truth[:, j]))
        hi = float(np.max(truth[:, j]))
        span = hi - lo
        if span < 1e-12:
            span = 1.0
        norm_truth[:, j] = (truth[:, j] - lo) / span
        norm_pred[:, j] = (pred[:, j] - lo) / span

    all_errors = norm_pred[missing] - norm_truth[missing]
    metrics["normalized_mae"] = float(np.mean(np.abs(all_errors)))
    metrics["normalized_rmse"] = float(np.sqrt(np.mean(all_errors ** 2)))

    num_mask = missing[:, :d_numerical]
    if d_numerical and num_mask.any():
        num_err = norm_pred[:, :d_numerical][num_mask] - norm_truth[:, :d_numerical][num_mask]
        metrics["numerical_mae_normalized"] = float(np.mean(np.abs(num_err)))
        metrics["numerical_rmse_normalized"] = float(np.sqrt(np.mean(num_err ** 2)))
        per_feature_r2 = []
        per_feature_wd = []
        for j in range(d_numerical):
            hidden = missing[:, j]
            if not hidden.any():
                continue
            y = truth[hidden, j]
            yhat = pred[hidden, j]
            denom = float(np.sum((y - np.mean(y)) ** 2))
            if denom > 1e-12 and len(y) > 1:
                per_feature_r2.append(1.0 - float(np.sum((y - yhat) ** 2)) / denom)
            per_feature_wd.append(wasserstein_distance(norm_pred[hidden, j], norm_truth[hidden, j]))
        metrics["numerical_r2_missing_only"] = float(np.mean(per_feature_r2)) if per_feature_r2 else None
        metrics["numerical_wasserstein"] = float(np.mean(per_feature_wd)) if per_feature_wd else None
    else:
        metrics["numerical_mae_normalized"] = None
        metrics["numerical_rmse_normalized"] = None
        metrics["numerical_r2_missing_only"] = None
        metrics["numerical_wasserstein"] = None

    cat_mask = missing[:, d_numerical:]
    if cat_mask.shape[1] and cat_mask.any():
        categorical_predictions = pred[:, d_numerical:].copy()
        cat_truth = truth[:, d_numerical:]
        cat_correct = np.zeros_like(cat_mask, dtype=bool)
        cat_errors = []
        for local_j in range(cat_mask.shape[1]):
            global_j = d_numerical + local_j
            rounded = np.rint(categorical_predictions[:, local_j])
            size = categorical_sizes.get(str(global_j))
            if size is None:
                size = categorical_sizes.get(global_j)
            if size:
                rounded = np.clip(rounded, 0, int(size) - 1)
            hidden = cat_mask[:, local_j]
            cat_correct[:, local_j] = rounded == cat_truth[:, local_j]
            if hidden.any():
                cat_errors.extend((rounded[hidden] - cat_truth[hidden, local_j]).tolist())
        metrics["categorical_accuracy"] = float(cat_correct[cat_mask].mean())
        metrics["categorical_code_mae"] = float(np.mean(np.abs(cat_errors))) if cat_errors else None
    else:
        metrics["categorical_accuracy"] = None
        metrics["categorical_code_mae"] = None

    # Aggregate featurewise Wasserstein distance on normalized numeric-coded columns.
    per_col_wd = []
    for j in range(d):
        hidden = missing[:, j]
        if hidden.any():
            per_col_wd.append(wasserstein_distance(norm_pred[hidden, j], norm_truth[hidden, j]))
    metrics["wasserstein_distance_normalized"] = float(np.mean(per_col_wd)) if per_col_wd else None
    return metrics
