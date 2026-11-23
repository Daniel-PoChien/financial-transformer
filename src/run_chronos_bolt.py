"""Evaluate frozen Chronos-Bolt-small as a zero-shot foundation-model baseline."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline

from run_tabular_baselines import make_features, split_masks


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("Chronos benchmark requires CUDA or MPS; run it in the project GPU notebook.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    parser.add_argument("--output", type=Path, default=Path("financial_transformer/results/chronos_bolt_predictions.csv"))
    parser.add_argument("--context", type=int, default=60)
    parser.add_argument("--checkpoint", default="amazon/chronos-bolt-small")
    args = parser.parse_args()
    started = time.monotonic()
    compute = device()
    pipeline = ChronosBoltPipeline.from_pretrained(args.checkpoint, device_map=compute, dtype=torch.float32)
    featured = make_features(pd.read_csv(args.data, parse_dates=["date"]))
    frames = []
    for _, group in featured.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        rows = np.flatnonzero(split_masks(group)[2].to_numpy())
        if len(rows) == 0 or rows.min() < args.context - 1:
            raise ValueError("Insufficient historical context for the holdout")
        observed = np.log(group["close"] / group["open"]).to_numpy(dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(observed, args.context)
        contexts = torch.from_numpy(windows[rows - args.context + 1])
        forecasts = []
        for batch in contexts.split(64):
            quantiles, _ = pipeline.predict_quantiles(batch, prediction_length=1, quantile_levels=[0.5])
            forecasts.append(quantiles[:, 0, 0].cpu().numpy())
        frame = group.loc[rows, ["date", "ticker", "target"]].copy()
        frame["model"] = "chronos_bolt_small"
        frame["prediction"] = np.concatenate(forecasts)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "device": compute, "context": args.context,
        "chronos_forecasting": version("chronos-forecasting"), "torch": str(torch.__version__),
        "seconds": round(time.monotonic() - started, 2),
    }, indent=2) + "\n")
    print(f"device={compute}; wrote {len(output)} predictions")


if __name__ == "__main__":
    main()
