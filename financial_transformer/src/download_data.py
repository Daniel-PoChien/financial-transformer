"""Download the fixed daily OHLCV research dataset with reproducible settings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "WMT", "CAT", "DIS", "KO", "SPY"]
START = "2015-01-01"
END = "2026-01-01"  # Exclusive: retains all observations through 2025.


def main() -> None:
    output = Path(__file__).parents[1] / "data" / "ohlcv.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        tickers=TICKERS,
        start=START,
        end=END,
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    rows: list[pd.DataFrame] = []
    for ticker in TICKERS:
        frame = raw[ticker].copy().reset_index()
        frame["ticker"] = ticker
        frame = frame.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        rows.append(frame[["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]])
    data = pd.concat(rows, ignore_index=True).dropna().sort_values(["ticker", "date"])
    if data.groupby("ticker").size().min() < 2_000:
        raise RuntimeError("At least one ticker has insufficient downloaded history")
    data.to_csv(output, index=False)
    print(f"Wrote {len(data):,} rows to {output}")


if __name__ == "__main__":
    main()
