"""Train and backtest matched five-minute AAPL forecasts from minute bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backtest import maximum_drawdown
from patch_transformer import PatchTransformer
from train_patch_transformer import device

CONTEXT = 60
HORIZON = 5
SEEDS = (7, 17, 27)
FEATURES = ["ret_1", "ret_5", "ret_20", "body", "range", "volatility", "tick_change"]


def load_samples(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    bars = pd.read_csv(path)
    bars["timestamp"] = pd.to_datetime(bars["datetime"], utc=True).dt.tz_convert("America/New_York")
    bars = bars.sort_values("timestamp")
    values = bars[["open", "high", "low", "close", "volume"]]
    if bars["timestamp"].duplicated().any() or not np.isfinite(values).all().all():
        raise ValueError("Duplicate timestamps or non-finite OHLCV")
    if (values.iloc[:, :4] <= 0).any().any() or (bars["volume"] < 0).any():
        raise ValueError("Invalid price or tick volume")
    if (bars["high"] < values[["open", "close", "low"]].max(axis=1)).any() or (
        bars["low"] > values[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC")
    clock = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    bars = bars.loc[clock.between(570, 959)].copy()
    bars["session"] = bars["timestamp"].dt.strftime("%Y-%m-%d")
    # Restart feature calculations at each session and every missing minute.
    segment = (bars["timestamp"].diff().ne(pd.Timedelta(minutes=1)) |
               bars["session"].ne(bars["session"].shift())).cumsum()
    windows, metadata = [], []
    for _, block in bars.groupby(segment):
        if len(block) < 20 + CONTEXT + HORIZON:
            continue
        close = np.log(block["close"])
        features = pd.DataFrame({
            "ret_1": close.diff(),
            "ret_5": close.diff(5),
            "ret_20": close.diff(20),
            "body": close - np.log(block["open"]),
            "range": np.log(block["high"] / block["low"]),
            "volatility": close.diff().rolling(20).std(),
            "tick_change": np.log1p(block["volume"]).diff(),
        })
        # Cut off the first 20 minutes, whose rolling features lack history.
        starts = np.arange(20, len(block) - CONTEXT - HORIZON + 1)
        ends = starts + CONTEXT - 1
        windows.append(features[FEATURES].to_numpy()[starts[:, None] + np.arange(CONTEXT)])
        signal = block["timestamp"].iloc[ends].reset_index(drop=True) + pd.Timedelta(minutes=1)
        entry = block["open"].to_numpy()[ends + 2]
        base = block["close"].to_numpy()[ends]
        actual = block["close"].to_numpy()[ends + HORIZON]
        metadata.append(pd.DataFrame({
            "signal_time": signal, "session": signal.dt.strftime("%Y-%m-%d"),
            "entry_time": signal + pd.Timedelta(minutes=1),
            "target_time": signal + pd.Timedelta(minutes=HORIZON),
            "price": base, "actual_price": actual, "entry": entry,
            "target": np.log(actual / base),
        }))
    if not windows:
        raise ValueError("No complete minute windows")
    return np.concatenate(windows).astype(np.float32), pd.concat(metadata, ignore_index=True)


def fit_predict(x: np.ndarray, y: np.ndarray, masks: list[np.ndarray], seed: int,
                compute: torch.device, epochs: int) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    train, validation, test = (torch.from_numpy(x[mask]) for mask in masks)
    y_train, y_valid = (torch.from_numpy(y[mask]) for mask in masks[:2])
    loader = DataLoader(TensorDataset(train, y_train), shuffle=True, batch_size=256)
    model = PatchTransformer(x.shape[-1], embedding_dim=32, encoder_layers=2).to(compute)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = nn.MSELoss()
    best_loss, best_epoch, weights = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for rows, labels in loader:
            optimizer.zero_grad()
            error = loss(model(rows.to(compute)), labels.to(compute))
            error.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model(batch.to(compute)).cpu() for batch in validation.split(256)])
        validation_loss = loss(predictions, y_valid).item()
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 3:
            break
    if weights is None:
        raise ValueError("Training produced no finite validation score")
    model.load_state_dict(weights)
    with torch.inference_mode():
        prediction = torch.cat([model(batch.to(compute)).cpu() for batch in test.split(256)]).numpy() / 10000
    return prediction, {"seed": seed, "best_epoch": best_epoch, "epochs": epoch, "validation_rmse_bps": best_loss**0.5}


def evaluate(test: pd.DataFrame, prediction: np.ndarray, cost_bps: float) -> tuple[dict, pd.DataFrame]:
    errors = test["price"] * np.exp(prediction) - test["actual_price"]
    # A clock-based five-minute cadence makes holding periods non-overlapping.
    eligible = test["signal_time"].dt.minute.mod(HORIZON).eq(0).to_numpy()
    signals = test.loc[eligible].copy()
    gross = signals["actual_price"] / signals["entry"] - 1
    cost = cost_bps / 10000
    # Use only the forecast and the completed signal bar to choose positions.
    # Round-trip costs are used as a fixed entry threshold.
    forecast = np.expm1(prediction[eligible])
    position = np.where(np.abs(forecast) > cost, np.sign(forecast), 0)
    signals["position"] = position
    signals["gross_return"] = position * gross
    signals["net_return"] = signals["gross_return"] - np.abs(position) * cost
    net = signals["net_return"]
    return {
        "price_mae_usd": errors.abs().mean(),
        "price_rmse_usd": np.sqrt((errors**2).mean()),
        "return_mae_bps": np.abs(prediction - test["target"]).mean() * 10000,
        "direction_accuracy": (np.sign(prediction) == np.sign(test["target"])).mean(),
        "opportunities": len(signals), "trades": np.count_nonzero(position),
        "gross_cumulative": (1 + signals["gross_return"]).prod() - 1,
        "net_cumulative": (1 + net).prod() - 1, "max_drawdown": maximum_drawdown(net),
    }, signals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    compute = device()
    if args.require_gpu and compute.type == "cpu":
        raise RuntimeError("GPU requested but neither CUDA nor MPS is available")

    windows, metadata = load_samples(args.data)
    sessions = metadata["session"].unique()
    if len(sessions) < 20:
        raise ValueError("At least 20 sessions with eligible windows are required")
    split = np.split(sessions, [int(len(sessions) * .60), int(len(sessions) * .80)])
    masks = [metadata["session"].isin(group).to_numpy() for group in split]

    # Fit transformations once per unique training minute appearing in an input.
    train_ends = metadata.loc[masks[0], "signal_time"].array.as_unit("ns").asi8
    minute_ns = pd.Timedelta(minutes=1).value
    input_times = train_ends[:, None] - minute_ns * np.arange(CONTEXT, 0, -1)
    _, first = np.unique(input_times.ravel(), return_index=True)
    train_features = windows[masks[0]].reshape(-1, len(FEATURES))[first]
    mean, std = train_features.mean(axis=0), train_features.std(axis=0)
    std[std == 0] = 1
    standardized = (windows - mean) / std
    pca = PCA(n_components=.90, svd_solver="full").fit((train_features - mean) / std)
    compressed = pca.transform(standardized.reshape(-1, len(FEATURES)))
    compressed = compressed.reshape(len(windows), CONTEXT, pca.n_components_)

    target_bps = metadata["target"].to_numpy(dtype=np.float32) * 10000
    test = metadata.loc[masks[2]].reset_index(drop=True)
    forecasts = {"unchanged_price": np.zeros(len(test)),
                 "historical_mean": np.full(len(test), target_bps[masks[0]].mean() / 10000)}
    details = []
    for name, inputs in (("full_transformer", standardized), ("pca_transformer", compressed)):
        runs = []
        for seed in SEEDS:
            prediction, training = fit_predict(inputs, target_bps, masks, seed, compute, args.epochs)
            runs.append(prediction)
            seed_metrics, _ = evaluate(test, prediction, 5)
            details.append({"model": name, **training, **seed_metrics})
            print(name, details[-1], flush=True)
        forecasts[name] = np.mean(runs, axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics, trades = [], []
    for name, prediction in forecasts.items():
        test[name + "_price"] = test["price"] * np.exp(prediction)
        for cost in (0, 2, 5, 10):
            scores, ledger = evaluate(test, prediction, cost)
            metrics.append({"model": name, "round_trip_bps": cost, **scores})
            if cost == 5:
                trades.append(ledger.assign(model=name))
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(details).to_csv(args.output / "training_metrics.csv", index=False)
    test.to_csv(args.output / "predictions.csv", index=False)
    pd.concat(trades).to_csv(args.output / "backtest_5bps.csv", index=False)

    # Paired bootstrap by session keeps overlapping five-minute forecast errors together.
    error = (test["pca_transformer_price"] - test["actual_price"]).abs()
    error -= (test["full_transformer_price"] - test["actual_price"]).abs()
    daily = error.groupby(test["session"]).agg(["sum", "count"])
    rng = np.random.default_rng(7)
    selections = rng.integers(len(daily), size=(5000, len(daily)))
    draws = daily["sum"].to_numpy()[selections].sum(axis=1) / daily["count"].to_numpy()[selections].sum(axis=1)
    report = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "device": str(compute), "torch": str(torch.__version__), "epochs_cap": args.epochs,
        "context_minutes": CONTEXT, "horizon_minutes": HORIZON, "seeds": SEEDS,
        "features": FEATURES, "train_mean": mean.tolist(), "train_std": std.tolist(),
        "pca_components": pca.components_.tolist(), "pca_center": pca.mean_.tolist(),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "splits": [{"first": group[0], "last": group[-1], "sessions": len(group), "samples": int(mask.sum())}
                   for group, mask in zip(split, masks)],
        "pca_minus_full_mae_usd_95ci": np.quantile(draws, [.025, .975]).tolist(),
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
