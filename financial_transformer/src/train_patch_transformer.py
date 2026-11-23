"""Train the project PatchTST-style model with a strict chronological split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from patch_transformer import PatchTransformer
from run_tabular_baselines import FEATURES, make_features, split_masks


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_windows(frame: pd.DataFrame, context: int, mean: np.ndarray, std: np.ndarray, pca: PCA | None) -> tuple[np.ndarray, pd.DataFrame]:
    windows, targets, identifiers = [], [], []
    for ticker, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        values = ((group[FEATURES].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
        if pca is not None:
            values = pca.transform(values).astype(np.float32)
        for row in range(context - 1, len(group)):
            windows.append(values[row - context + 1 : row + 1])
            targets.append(group.loc[row, ["target", "target_date"]].tolist())
            identifiers.append((group.loc[row, "date"], ticker))
    metadata = pd.DataFrame(identifiers, columns=["date", "ticker"])
    metadata[["target", "target_date"]] = pd.DataFrame(targets, index=metadata.index)
    return np.asarray(windows), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    parser.add_argument("--output", type=Path, default=Path("financial_transformer/results/patch_transformer_predictions.csv"))
    parser.add_argument("--context", type=int, default=60, choices=[60, 120])
    parser.add_argument("--patch", type=int, default=5, choices=[5, 10])
    parser.add_argument("--pca-variance", type=float, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    started = time.monotonic()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.pca_variance is not None and not 0 < args.pca_variance < 1:
        parser.error("--pca-variance must be strictly between 0 and 1")
    model_name = "pca_transformer" if args.pca_variance is not None else "patch_transformer"
    if args.context % args.patch:
        raise ValueError("Context must be divisible by patch length")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    compute = device()
    if args.require_gpu and compute.type == "cpu":
        raise RuntimeError("No CUDA or MPS accelerator is available")

    raw = pd.read_csv(args.data, parse_dates=["date"])
    featured = make_features(raw)
    train_rows = featured[split_masks(featured)[0]]
    mean = train_rows[FEATURES].mean().to_numpy(dtype=np.float32)
    std = train_rows[FEATURES].std().replace(0, 1).to_numpy(dtype=np.float32)
    pca = None
    if args.pca_variance is not None:
        standardized = (train_rows[FEATURES].to_numpy(dtype=np.float32) - mean) / std
        pca = PCA(n_components=args.pca_variance, svd_solver="full").fit(standardized)
    windows, metadata = build_windows(featured, args.context, mean, std, pca)
    train_mask, valid_mask, test_mask = split_masks(metadata)
    train_x, valid_x, test_x = (windows[mask.to_numpy()] for mask in (train_mask, valid_mask, test_mask))
    train_y = metadata.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    valid_y = metadata.loc[valid_mask, "target"].to_numpy(dtype=np.float32)
    if min(len(train_x), len(valid_x), len(test_x)) == 0:
        raise ValueError("Empty sequence partition; check date coverage and context length")

    model = PatchTransformer(train_x.shape[-1], context_length=args.context, patch_length=args.patch).to(compute)
    # Express the same MSE objective in percentage points for numerical conditioning.
    train_y *= 100
    valid_y *= 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=128, shuffle=True)
    best_state, best_valid, stale = None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(compute)), batch_y.to(compute))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_errors = [
                (model(batch_x.to(compute)) - batch_y.to(compute)).square().sum().item()
                for batch_x, batch_y in zip(torch.from_numpy(valid_x).split(128), torch.from_numpy(valid_y).split(128))
            ]
            valid_loss = sum(validation_errors) / len(valid_y)
        print(f"{model_name} seed={args.seed} epoch={epoch} validation_mse={valid_loss / 10000:.8f}", flush=True)
        if valid_loss < best_valid:
            best_epoch = epoch
            best_valid, stale = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = np.concatenate([
            model(batch.to(compute)).cpu().numpy()
            for batch in torch.from_numpy(test_x).split(128)
        ]) / 100
    output = metadata.loc[test_mask, ["date", "ticker", "target"]].copy()
    output["model"] = model_name
    output["prediction"] = predicted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps({
        "model": model_name, "seed": args.seed, "device": compute.type,
        "torch": str(torch.__version__), "pandas": pd.__version__, "numpy": np.__version__,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "context": args.context, "patch": args.patch,
        "features": FEATURES, "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "pca": None if pca is None else {
            "variance_threshold": args.pca_variance,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(), "center": pca.mean_.tolist(),
        },
        "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.0001,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch, "epochs_run": epoch, "max_epochs": args.epochs,
        "validation_mse": best_valid / 10000, "seconds": round(time.monotonic() - started, 2),
    }, indent=2) + "\n")
    print(f"device={compute.type}; wrote {len(output)} predictions")


if __name__ == "__main__":
    main()
