"""Backtest cross-sectional long-short predictions with transparent assumptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax().clip(lower=1.0) - 1).min())


def backtest(predictions: pd.DataFrame, prices: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["next_open"] = prices.groupby("ticker")["open"].shift(-1)
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices["next_intraday_return"] = prices["next_close"] / prices["next_open"] - 1
    merged = predictions.merge(prices[["date", "ticker", "next_intraday_return"]], on=["date", "ticker"], how="inner", validate="many_to_one").dropna(subset=["next_intraday_return"])
    records: list[dict[str, object]] = []
    for (model, date), day in merged.groupby(["model", "date"], sort=True):
        if len(day) < 6:
            continue
        day = day.sort_values(["prediction", "ticker"])
        short = day.head(3)
        long = day.tail(3)
        gross = 0.5 * long["next_intraday_return"].mean() - 0.5 * short["next_intraday_return"].mean()
        # Positions are rebuilt every day: total one-way turnover is 2.0 gross notional.
        turnover = 2.0
        cost = turnover * cost_bps / 10_000
        records.append({"date": date, "model": model, "gross_return": gross, "turnover": turnover, "net_return": gross - cost})
    return pd.DataFrame(records)


def summarize(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in returns.groupby("model"):
        net = group["net_return"]
        annual_return = (1 + net).prod() ** (252 / len(net)) - 1
        annual_volatility = net.std(ddof=1) * np.sqrt(252)
        rows.append({
            "model": model,
            "days": len(net),
            "annualized_return": annual_return,
            "annualized_volatility": annual_volatility,
            "sharpe": net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) else np.nan,
            "max_drawdown": maximum_drawdown(net),
            "mean_daily_turnover": group["turnover"].mean(),
            "cumulative_net_return": (1 + net).prod() - 1,
        })
    return pd.DataFrame(rows).set_index("model").sort_index()


def add_matched_benchmarks(returns: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    """Add cash and a daily equal-weight long-only reference on identical open-close returns."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["next_open"] = prices.groupby("ticker")["open"].shift(-1)
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices["next_intraday_return"] = prices["next_close"] / prices["next_open"] - 1
    dates = returns["date"].drop_duplicates()
    reference = prices[prices["date"].isin(dates)].groupby("date")["next_intraday_return"].mean().dropna()
    equal_weight = pd.DataFrame({"date": reference.index, "model": "equal_weight_long_only", "gross_return": reference.to_numpy(), "turnover": 2.0, "net_return": reference.to_numpy() - 2 * cost_bps / 10_000})
    cash = pd.DataFrame({"date": reference.index, "model": "cash", "gross_return": 0.0, "turnover": 0.0, "net_return": 0.0})
    return pd.concat([returns, equal_weight, cash], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("financial_transformer/results/backtest_returns.csv"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    prices = pd.read_csv(args.prices, parse_dates=["date"])
    returns = add_matched_benchmarks(backtest(predictions, prices, args.cost_bps), prices, args.cost_bps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(args.output, index=False)
    metrics = summarize(returns)
    metrics.to_csv(args.output.with_name("backtest_metrics.csv"))
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
