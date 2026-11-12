"""Shared, device-safe training loop for the PMAE and ReMasker models."""
from __future__ import annotations
import math
import random
import time
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from ..utils import MAEDataset, adjust_learning_rate

EPS = 1e-8


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TabularMAETrainer:
    model_kind = "pmae"

    def __init__(self, args):
        self.args = args
        self.batch_size = int(getattr(args, "batch_size", 128))
        self.accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
        self.min_lr = float(getattr(args, "min_lr", 1e-5))
        self.norm_field_loss = bool(getattr(args, "norm_field_loss", False))
        self.weight_decay = float(getattr(args, "weight_decay", 0.05))
        self.lr = getattr(args, "lr", None)
        self.blr = float(getattr(args, "blr", 1e-3))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 40))
        self.embed_dim = int(getattr(args, "embed_dim", 32) or 32)
        self.depth = int(getattr(args, "depth", 4) or 4)
        self.decoder_depth = int(getattr(args, "decoder_depth", 2) or 2)
        self.num_heads = int(getattr(args, "num_heads", 4) or 4)
        self.mlp_ratio = float(getattr(args, "mlp_ratio", 4.0) or 4.0)
        self.max_epochs = int(getattr(args, "max_epochs", 300))
        self.mask_ratio = float(getattr(args, "mask_ratio", 0.5) or 0.5)
        self.encode_func = getattr(args, "encode_func", "linear") or "linear"
        self.device = getattr(args, "device", "cpu") or "cpu"
        self.model = None
        self.norm_parameters = None
        self.history = []
        self.block_mlp = None
        self.new_imp = True
        self.old_loss = False
        self.a = 0.05
        self.b = 0.5

    def _resolve_device(self):
        wanted = str(self.device)
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA was requested but is unavailable; using CPU.")
            wanted = "cpu"
        return torch.device(wanted)

    @staticmethod
    def _as_array(X_raw):
        if isinstance(X_raw, pd.DataFrame):
            columns = list(X_raw.columns)
            values = X_raw.to_numpy(dtype=np.float32, copy=True)
        elif torch.is_tensor(X_raw):
            columns = None
            values = X_raw.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            columns = None
            values = np.asarray(X_raw, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
            raise ValueError("Input must be a non-empty 2D matrix with at least two features.")
        values[~np.isfinite(values)] = np.nan
        return values, columns

    def _make_model(self, n_features):
        from .. import pmae, model_mae
        norm_layer = partial(nn.LayerNorm, eps=EPS)
        common = dict(
            rec_len=n_features,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func,
        )
        if self.model_kind == "pmae":
            return pmae.MaskedAutoencoder(**common, block_mlp=self.block_mlp, old_loss=self.old_loss)
        return model_mae.MaskedAutoencoder(**common)

    def _model_forward(self, samples, masks, ratio=None):
        if self.model_kind == "pmae":
            return self.model(samples, masks, mask_ratio=ratio)
        return self.model(samples.unsqueeze(1), masks, mask_ratio=0.5)

    def _model_predict(self, X, M):
        if self.model_kind == "pmae":
            _, prediction, *_ = self.model(X, M)
        else:
            _, prediction, *_ = self.model(X.unsqueeze(1), M, mask_ratio=0.5)
        return prediction.squeeze(-1)

    def fit(self, X_raw):
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        values, self.columns = self._as_array(X_raw)
        missing_mask = ~np.isfinite(values)
        if missing_mask.all(axis=0).any():
            bad = np.where(missing_mask.all(axis=0))[0].tolist()
            raise ValueError(f"All values are missing in feature indices {bad}; cannot train PMAE.")

        device = self._resolve_device()
        n_rows, n_features = values.shape
        min_values = np.nanmin(values, axis=0).astype(np.float32)
        max_values = np.nanmax(values, axis=0).astype(np.float32)
        ranges = max_values - min_values
        safe_ranges = np.where(ranges > EPS, ranges, 1.0).astype(np.float32)
        self.norm_parameters = {"min": min_values, "max": max_values, "range": safe_ranges}

        normalized = (values - min_values) / safe_ranges
        normalized[missing_mask] = np.nan
        observed = torch.from_numpy((~missing_mask).astype(np.float32)).to(device)
        normalized_tensor = torch.from_numpy(np.nan_to_num(normalized, nan=0.0).astype(np.float32)).to(device)

        self.model = self._make_model(n_features).to(device)
        learning_rate = self.lr if self.lr is not None else self.blr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                      betas=(0.9, 0.95), weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        dataset = MAEDataset(normalized_tensor, observed)
        loader = DataLoader(dataset, sampler=RandomSampler(dataset),
                            batch_size=max(1, min(self.batch_size, n_rows)),
                            pin_memory=(device.type == "cuda"))
        self.history = []
        started = time.time()

        for epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            batches = len(loader)
            for step, (samples, masks) in enumerate(loader):
                progress = epoch + step / max(1, batches)
                adjust_learning_rate(optimizer, progress, learning_rate, self.min_lr,
                                     max(self.max_epochs, 1), min(self.warmup_epochs, max(self.max_epochs - 1, 0)))
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                ratio = None
                if self.model_kind == "pmae":
                    observed_rate = masks.float().mean(dim=0)
                    ratio = self.a * torch.log((observed_rate + 1e-6) / (1.0 - observed_rate + 1e-6)) + self.b
                    ratio = torch.where(observed_rate >= 0.9999, torch.ones_like(ratio), ratio.clamp(0.0, 1.0))

                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    output = self._model_forward(samples, masks, ratio)
                    loss = output[0]

                if loss is None or not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                losses.append(float(loss.detach().cpu()))
                if scaler is not None:
                    scaler.scale(loss / self.accum_iter).backward()
                else:
                    (loss / self.accum_iter).backward()
                should_step = ((step + 1) % self.accum_iter == 0) or (step + 1 == batches)
                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch_loss = float(np.mean(losses)) if losses else float("nan")
            self.history.append(epoch_loss)
            if (epoch % 20 == 0) or (epoch + 1 == self.max_epochs):
                print(f"{self.model_kind.upper()} epoch {epoch + 1}/{self.max_epochs}: loss={epoch_loss:.6f}")
            if not math.isfinite(epoch_loss):
                raise RuntimeError(f"{self.model_kind} training produced no finite losses in epoch {epoch + 1}.")

        print(f"{self.model_kind.upper()} training completed in {time.time() - started:.1f}s on {device}.")
        self.model.eval()
        return self.transform(values)

    def transform(self, X_raw):
        if self.model is None or self.norm_parameters is None:
            raise RuntimeError("Call fit() before transform().")
        values, _ = self._as_array(X_raw)
        if values.shape[1] != len(self.norm_parameters["min"]):
            raise ValueError("Transform input feature count differs from the fitted data.")
        missing = ~np.isfinite(values)
        min_values = self.norm_parameters["min"]
        ranges = self.norm_parameters["range"]
        normalized = (values - min_values) / ranges
        normalized[missing] = np.nan
        X = torch.from_numpy(np.nan_to_num(normalized, nan=0.0).astype(np.float32)).to(self._resolve_device())
        M = torch.from_numpy((~missing).astype(np.float32)).to(self._resolve_device())
        self.model.eval()
        with torch.no_grad():
            imputed_scaled = self._model_predict(X, M).detach().cpu().numpy()
        imputed = imputed_scaled * ranges + min_values
        # Constant columns have no meaningful scale; their only valid
        # imputation is the constant value observed during fitting.
        constant_columns = self.norm_parameters["max"] - min_values <= EPS
        if np.any(constant_columns):
            imputed[:, constant_columns] = min_values[constant_columns]
        result = values.copy()
        result[missing] = imputed[missing]
        if not np.isfinite(result).all():
            raise RuntimeError("The imputed result contains NaN or infinite values.")
        return torch.from_numpy(result.astype(np.float32))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X).cpu().numpy()
