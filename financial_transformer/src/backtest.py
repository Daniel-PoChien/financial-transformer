"""Backtest cross-sectional long-short predictions with transparent assumptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax().clip(lower=1.0) - 1).min())


def _next_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    stocks = prices.loc[prices["ticker"].ne("SPY")].sort_values(["ticker", "date"])
    following = stocks.groupby("ticker")[["open", "close"]].shift(-1)
    return stocks[["date", "ticker"]].assign(next_intraday_return=following["close"] / following["open"] - 1)


def backtest(
    predictions: pd.DataFrame, prices: pd.DataFrame, cost_bps: float, *,
    include_benchmarks: bool = False, require_full_coverage: bool = False,
) -> pd.DataFrame:
    """Rank stock forecasts and optionally include the matched cash and long-only portfolios."""
    stock_returns = _next_stock_returns(prices)
    is_cash = predictions["model"].eq("zero_return")
    ranked = predictions.loc[~is_cash & predictions["ticker"].ne("SPY")]
    if require_full_coverage:
        coverage = ranked.groupby(["model", "date"]).size()
        if not coverage.eq(stock_returns["ticker"].nunique()).all():
            raise ValueError("Trading requires complete daily coverage of the stock universe")
    merged = ranked.merge(stock_returns, on=["date", "ticker"], validate="many_to_one")
    merged = merged.dropna(subset=["next_intraday_return"])
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
    returns = pd.DataFrame.from_records(records, columns=["date", "model", "gross_return", "turnover", "net_return"])
    if include_benchmarks:
        cash_dates = pd.Index(predictions.loc[is_cash, "date"].unique())
        dates = cash_dates.union(returns["date"].unique(), sort=False)
        return _with_benchmarks(returns, stock_returns, dates, cost_bps)
    return returns


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
    dates = pd.Index(returns["date"].unique())
    return _with_benchmarks(returns, _next_stock_returns(prices), dates, cost_bps)


def _with_benchmarks(
    returns: pd.DataFrame, stock_returns: pd.DataFrame, dates: pd.Index, cost_bps: float,
) -> pd.DataFrame:
    reference = stock_returns.loc[stock_returns["date"].isin(dates)].groupby("date")["next_intraday_return"].mean().dropna()
    equal_weight = pd.DataFrame({"date": reference.index, "model": "equal_weight_long_only", "gross_return": reference.to_numpy(), "turnover": 2.0, "net_return": reference.to_numpy() - 2 * cost_bps / 10_000})
    cash = pd.DataFrame({"date": reference.index, "model": "cash", "gross_return": 0.0, "turnover": 0.0, "net_return": 0.0})
    benchmarks = [equal_weight, cash]
    if not returns.empty:
        benchmarks.insert(0, returns)
    return pd.concat(benchmarks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("financial_transformer/results/backtest_returns.csv"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    prices = pd.read_csv(args.prices, parse_dates=["date"])
    returns = backtest(predictions, prices, args.cost_bps, include_benchmarks=True)
    if returns.empty:
        parser.error("No eligible portfolios on dates with next-session stock prices")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(args.output, index=False)
    metrics = summarize(returns)
    metrics.to_csv(args.output.with_name("backtest_metrics.csv"))
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
