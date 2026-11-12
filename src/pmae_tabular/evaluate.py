"""Backward-compatible evaluation API plus robust imputation metrics."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

from .evaluation.metrics import evaluate_imputation


def scale_data(X: pd.DataFrame):
    scaler = MinMaxScaler()
    columns = X.columns
    return pd.DataFrame(scaler.fit_transform(X), columns=columns), scaler


def MAE(X, X_true, mask):
    X, X_true, mask = np.asarray(X), np.asarray(X_true), np.asarray(mask, dtype=bool)
    return float(np.mean(np.abs(X[mask] - X_true[mask]))) if mask.any() else float("nan")


def RMSE(X, X_true, mask):
    X, X_true, mask = np.asarray(X), np.asarray(X_true), np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.mean((X[mask] - X_true[mask]) ** 2))) if mask.any() else float("nan")


def ws_score(X_fitted, X_true, miss_mask, miss_cols):
    scores = []
    for col in miss_cols:
        selected = np.asarray(miss_mask)[:, col].astype(bool)
        if selected.any():
            scores.append(wasserstein_distance(np.asarray(X_fitted)[selected, col],
                                               np.asarray(X_true)[selected, col]))
    return float(np.mean(scores)) if scores else float("nan")


def evaluator(X_init, out, mask, d, d_numerical, miss_cols=None, cat_exists=None,
              categorical_sizes=None):
    """Legacy signature retained for notebooks/scripts from the flat repository.

    `mask` follows the original convention (True/1 means artificially missing).
    """
    truth = np.asarray(X_init.detach().cpu() if hasattr(X_init, "detach") else X_init)
    prediction = np.asarray(out.detach().cpu() if hasattr(out, "detach") else out)
    missing = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask).astype(bool)
    scores = evaluate_imputation(truth, prediction, missing, d_numerical, categorical_sizes)
    numeric_r2 = scores.get("numerical_r2_missing_only")
    cat_accuracy = scores.get("categorical_accuracy")
    available = [v for v in (numeric_r2, cat_accuracy) if v is not None]
    imp_acc = float(np.mean(available)) if available else None
    metrics = {
        "imp_acc": imp_acc,
        "R2(numerical)": numeric_r2,
        "Acc(categorical)": cat_accuracy,
        "WD": scores.get("wasserstein_distance_normalized"),
        "RMSE": scores.get("normalized_rmse"),
        "RMSE(num)": scores.get("numerical_rmse_normalized"),
        "RMSE(cat)": scores.get("categorical_code_mae"),
    }
    metrics.update(scores)
    return metrics


__all__ = ["scale_data", "MAE", "RMSE", "ws_score", "evaluator", "evaluate_imputation"]
