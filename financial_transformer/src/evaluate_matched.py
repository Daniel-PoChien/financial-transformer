"""Score a common holdout and simulate identical trades for each model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import backtest, summarize
from run_tabular_baselines import make_features, score, split_masks

KEY = ["date", "ticker"]


def load_predictions(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if frame.empty or frame["model"].isna().any():
        raise ValueError(f"No predictions in {path}")
    for model, group in frame.groupby("model"):
        aligned = group.set_index(KEY).sort_index()
        if not aligned.index.equals(expected.index):
            raise ValueError(f"{model} in {path} does not cover the exact holdout panel")
        np.testing.assert_allclose(aligned["target"], expected["target"], atol=1e-9, rtol=1e-6)
        if not np.isfinite(aligned["prediction"]).all():
            raise ValueError(f"Non-finite predictions for {model} in {path}")
    return frame


def paired_mae_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.assign(error=(frame["target"] - frame["prediction"]).abs())
    daily = daily.groupby(["date", "model"])["error"].mean().unstack()
    block = 20
    rng = np.random.default_rng(7)
    starts = rng.integers(0, len(daily) - block + 1, size=(5000, int(np.ceil(len(daily) / block))))
    sample = (starts[:, :, None] + np.arange(block)).reshape(5000, -1)[:, :len(daily)]
    rows = []
    for reference in daily.columns.drop("pca_transformer"):
        diff = (daily["pca_transformer"] - daily[reference]).to_numpy()
        lower, upper = np.quantile(diff[sample].mean(axis=1), [0.025, 0.975])
        rows.append({"reference": reference, "mae_difference": diff.mean(), "ci_low": lower, "ci_high": upper})
    return pd.DataFrame(rows)


def forecast_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups):
        metrics = score(group["target"], group["prediction"].to_numpy())
        if "ticker" in groups:
            actual, predicted = group["target"].rank(), group["prediction"].rank()
            metrics["spearman"] = actual.corr(predicted) if actual.nunique() > 1 and predicted.nunique() > 1 else np.nan
        rows.append({**dict(zip(groups, keys)), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("financial_transformer/results/matched"))
    parser.add_argument("--prices", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    args = parser.parse_args()
    prices = pd.read_csv(args.prices, parse_dates=["date"])
    if prices.duplicated(KEY).any():
        raise ValueError("Duplicate price observations")
    if not np.isfinite(prices[["open", "high", "low", "close", "adjusted_close", "volume"]]).all().all():
        raise ValueError("Non-finite OHLCV values")
    featured = make_features(prices)
    expected = featured.loc[split_masks(featured)[2]].set_index(KEY).sort_index()

    frames = [load_predictions(args.results / "tabular_predictions.csv", expected)]
    seed_metrics = []
    for model in ("patch_transformer", "pca_transformer"):
        runs = []
        for seed in (7, 17, 27):
            run = load_predictions(args.results / f"{model}_{seed}.csv", expected)
            if set(run["model"]) != {model}:
                raise ValueError(f"Unexpected model in {model}_{seed}.csv")
            runs.append(run)
            trade = summarize(backtest(run, prices, 10)).loc[model].to_dict()
            seed_metrics.append({"model": model, "seed": seed, **score(run["target"], run["prediction"].to_numpy()), **trade})
        ensemble = pd.concat(runs).groupby(KEY + ["model"], as_index=False)[["target", "prediction"]].mean()
        frames.append(ensemble)

    chronos = args.results / "chronos_bolt_predictions.csv"
    if chronos.exists():
        frames.append(load_predictions(chronos, expected))
    matched = pd.concat(frames, ignore_index=True)
    if matched.duplicated(KEY + ["model"]).any():
        raise ValueError("Duplicate matched predictions")
    matched.to_csv(args.results / "matched_predictions.csv", index=False)
    metrics = forecast_metrics(matched, ["model"])
    metrics.to_csv(args.results / "forecast_metrics.csv", index=False)
    by_asset = forecast_metrics(matched.assign(year=matched["date"].dt.year), ["model", "year", "ticker"])
    by_asset.to_csv(args.results / "forecast_by_asset_year.csv", index=False)
    paired_mae_intervals(matched).to_csv(args.results / "paired_mae_intervals.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(args.results / "seed_metrics.csv", index=False)

    returns = backtest(matched, prices, 10, include_benchmarks=True, require_full_coverage=True)
    returns = returns.merge(expected.reset_index()[["date", "target_date"]].drop_duplicates(),
                            on="date", validate="many_to_one")
    returns.to_csv(args.results / "backtest_returns.csv", index=False)
    summary = summarize(returns)
    if not summary["days"].eq(expected.index.get_level_values("date").nunique()).all():
        raise ValueError("Backtest did not cover every holdout date")
    summary.to_csv(args.results / "backtest_metrics.csv")
    print(metrics.to_string(index=False))
    print(summary.round(5).to_string())


if __name__ == "__main__":
    main()
