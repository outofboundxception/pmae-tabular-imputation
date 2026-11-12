"""End-to-end dataset preparation, amputation, PMAE/Remasker training, and scoring."""
from __future__ import annotations
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .data import canonical_dataset_name, prepare_dataset, load_processed_dataset
from .data.amputation import generate_missingness, amputation_path, save_amputation, load_amputation
from .evaluation import evaluate_imputation
from .fit_pmae import ProportionalMasker, set_all_seeds
from .fit_remasker import ReMasker


def _namespace(epochs=300, batch_size=128, device="auto", embed_dim=32,
               depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0,
               learning_rate=None, weight_decay=0.05, warmup_epochs=40):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(
        batch_size=int(batch_size), accum_iter=1, min_lr=1e-5,
        norm_field_loss=False, weight_decay=float(weight_decay), lr=learning_rate,
        blr=1e-3, warmup_epochs=int(warmup_epochs), embed_dim=int(embed_dim),
        depth=int(depth), decoder_depth=int(decoder_depth), num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio), max_epochs=int(epochs), mask_ratio=0.5,
        encode_func="linear", device=device,
    )


def _result_directory(results_root, dataset, mechanism, column_fraction, missing_rate, seed):
    return (Path(results_root) / dataset / mechanism.upper()
            / f"columns_{column_fraction:.2f}" / f"rate_{missing_rate:.2f}" / f"seed_{seed}")


def prepare_one(dataset, repo_root: Path, force=False):
    root = Path(repo_root)
    return prepare_dataset(dataset, root / "data" / "raw", root / "data" / "processed", force=force)


def amputate_one(dataset, repo_root: Path, mechanism="MCAR", column_fraction=0.3,
                 missing_rate=0.5, seed=42, force=False):
    root = Path(repo_root)
    key = canonical_dataset_name(dataset)
    processed_path = root / "data" / "processed" / key / "processed.pkl"
    if not processed_path.exists():
        prepare_one(key, root)
    processed = load_processed_dataset(key, root / "data" / "processed")
    path = amputation_path(root / "data" / "amputated", key, mechanism,
                           column_fraction, missing_rate, seed)
    if path.exists() and not force:
        return path
    preexisting_mask = np.asarray(
        processed.get("preexisting_missing_mask", np.zeros_like(processed["X"], dtype=bool)),
        dtype=bool,
    )
    incomplete, mask, metadata = generate_missingness(
        processed["X"], mechanism=mechanism, column_fraction=column_fraction,
        missing_rate=missing_rate, seed=seed, eligible_mask=~preexisting_mask,
    )
    # Source-missing cells remain missing inputs to the model, but are not
    # included in the synthetic evaluation mask because their true values
    # are unknown. The preprocessed matrix stores median/mode placeholders
    # only so the data matrix remains finite and shape-stable.
    incomplete[preexisting_mask] = np.nan
    metadata.update({
        "dataset": key,
        "feature_columns": processed["columns"],
        "d_numerical": processed["d_numerical"],
        "categorical_columns": processed["categorical_columns"],
        "preexisting_missing_cells": int(preexisting_mask.sum()),
        "input_missing_cells_total": int(np.isnan(incomplete).sum()),
    })
    return save_amputation(path, incomplete, mask, metadata)


def run_experiment(dataset, repo_root: Path, mechanism="MCAR", column_fraction=0.3,
                   missing_rate=0.5, seed=42, epochs=300, batch_size=128,
                   device="auto", models=("pmae", "remasker"), force_prepare=False,
                   force_amputation=False, embed_dim=32, depth=4,
                   decoder_depth=2, num_heads=4, mlp_ratio=4.0):
    root = Path(repo_root)
    key = canonical_dataset_name(dataset)
    model_names = tuple(str(name).lower().strip() for name in models if str(name).strip())
    if not model_names:
        raise ValueError("Select at least one model: pmae and/or remasker.")
    unknown_models = set(model_names) - {"pmae", "remasker", "re_masker"}
    if unknown_models:
        raise ValueError(f"Unknown model(s): {sorted(unknown_models)}. Use pmae and/or remasker.")
    processed = prepare_one(key, root, force=force_prepare)
    # A forced preprocessing refresh changes the ground-truth matrix. Never
    # reuse an amputation artifact generated from an older processed matrix.
    amputation_file = amputate_one(
        key, root, mechanism, column_fraction, missing_rate, seed,
        force=(force_amputation or force_prepare),
    )
    incomplete, mask, amputation_meta = load_amputation(amputation_file)
    truth = np.asarray(processed["X"], dtype=np.float32)
    if incomplete.shape != truth.shape:
        raise ValueError(f"Amputated data shape {incomplete.shape} does not match complete data {truth.shape}.")

    out_dir = _result_directory(root / "results", key, mechanism, column_fraction, missing_rate, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = _namespace(epochs, batch_size, device, embed_dim, depth, decoder_depth,
                      num_heads, mlp_ratio)
    set_all_seeds(seed)
    training_config = {
        "epochs": int(epochs), "batch_size": int(batch_size), "device": args.device,
        "embed_dim": int(embed_dim), "depth": int(depth), "decoder_depth": int(decoder_depth),
        "num_heads": int(num_heads), "mlp_ratio": float(mlp_ratio),
        "learning_rate": float(args.lr if args.lr is not None else args.blr),
        "minimum_learning_rate": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "gradient_accumulation_steps": int(args.accum_iter),
        "encoding_function": str(args.encode_func),
        "norm_field_loss": bool(args.norm_field_loss),
        "pmae_block": "MLP-Mixer block_config=0",
        "remasker_block": "Transformer MAE",
        "pmae_proportional_masking": {"a": 0.05, "b": 0.5},
        "categorical_handling": "ordinal codes represented as scalar columns; outputs rounded/clipped and decoded; see README limitations",
    }
    experiment_config = {
        "format_version": 1,
        "dataset": key,
        "missingness": {
            "mechanism": mechanism.upper(),
            "column_fraction": float(column_fraction),
            "missing_rate_per_selected_column": float(missing_rate),
            "seed": int(seed),
        },
        "models": ["remasker" if name == "re_masker" else name for name in model_names],
        "training": training_config,
        "data": {
            "n_samples": int(truth.shape[0]),
            "n_features": int(truth.shape[1]),
            "feature_columns": list(processed["columns"]),
            "d_numerical": int(processed["d_numerical"]),
            "numerical_columns": list(processed["numerical_columns"]),
            "categorical_columns": list(processed["categorical_columns"]),
            "category_mappings": processed["category_mappings"],
            "target_columns": processed["source_metadata"].get("excluded_target_columns", []),
            "identifier_columns": processed["source_metadata"].get("excluded_identifier_columns", []),
        },
        "amputation_file": str(amputation_file.relative_to(root)),
    }
    metrics = {
        "dataset": key,
        "mechanism": mechanism.upper(),
        "column_fraction": float(column_fraction),
        "missing_rate_per_selected_column": float(missing_rate),
        "seed": int(seed),
        "n_samples": int(truth.shape[0]),
        "n_features": int(truth.shape[1]),
        "d_numerical": int(processed["d_numerical"]),
        "categorical_columns": processed["categorical_columns"],
        "amputation_file": str(amputation_file.relative_to(root)),
        "models": {},
    }
    categorical_sizes = {
        str(processed["columns"].index(col)): len(categories)
        for col, categories in processed["category_mappings"].items()
    }
    input_frame = pd.DataFrame(incomplete, columns=processed["columns"])

    start = time.time()
    for model_name in model_names:
        model_start = time.time()
        if model_name == "pmae":
            model = ProportionalMasker(args)
        elif model_name in {"remasker", "re_masker"}:
            model = ReMasker(args)
            model_name = "remasker"
        else:
            raise ValueError(f"Unknown model '{model_name}'. Use pmae and/or remasker.")
        set_all_seeds(seed)
        imputed_tensor = model.fit(input_frame)
        imputed = (imputed_tensor.detach().cpu().numpy()
                   if torch.is_tensor(imputed_tensor) else np.asarray(imputed_tensor))

        # The original scalar decoder predicts continuous values for ordinal
        # category codes. Convert those columns to valid category indices
        # before scoring or exporting the result.
        imputed_encoded = np.asarray(imputed, dtype=np.float32).copy()
        for categorical_column, categories in processed["category_mappings"].items():
            column_index = processed["columns"].index(categorical_column)
            if categories:
                imputed_encoded[:, column_index] = np.clip(
                    np.rint(imputed_encoded[:, column_index]), 0, len(categories) - 1
                )

        model_metrics = evaluate_imputation(
            truth, imputed_encoded, mask, processed["d_numerical"], categorical_sizes
        )
        model_metrics["training_seconds"] = float(time.time() - model_start)
        metrics["models"][model_name] = model_metrics

        # Save a numeric-code matrix that can be reloaded directly, plus a
        # human-readable CSV with original category labels restored.
        pd.DataFrame(imputed_encoded, columns=processed["columns"]).to_csv(
            out_dir / f"{model_name}_imputed_encoded.csv", index=False)
        decoded_frame = pd.DataFrame(imputed_encoded, columns=processed["columns"])
        for categorical_column, categories in processed["category_mappings"].items():
            column_index = processed["columns"].index(categorical_column)
            codes = imputed_encoded[:, column_index].astype(int)
            decoded_frame[categorical_column] = [categories[code] for code in codes]
        decoded_frame.to_csv(out_dir / f"{model_name}_imputed.csv", index=False)
        (out_dir / f"{model_name}_training_history.json").write_text(
            json.dumps(model.history, indent=2), encoding="utf-8")
        if model.model is not None:
            checkpoint = {
                "format_version": 1,
                "model_name": model_name,
                "model_state_dict": model.model.state_dict(),
                "normalization": model.norm_parameters,
                "feature_columns": list(processed["columns"]),
                "categorical_columns": list(processed["categorical_columns"]),
                "category_mappings": processed["category_mappings"],
                "d_numerical": int(processed["d_numerical"]),
                "training_configuration": training_config,
                "dataset": key,
            }
            torch.save(checkpoint, out_dir / f"{model_name}_model_checkpoint.pt")
        print(f"{model_name} metrics: {json.dumps(model_metrics, indent=2)}")

    metrics["configuration"] = training_config
    metrics["elapsed_seconds_total"] = float(time.time() - start)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return metrics
