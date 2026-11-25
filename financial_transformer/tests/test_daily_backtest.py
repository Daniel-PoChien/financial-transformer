import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
TICKERS = ("A", "B", "C", "D", "E", "F", "SPY")


class DailyBacktestTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prices = self.root / "prices.csv"

        # Supply 20 sessions for features and enough holdout signals for a 20-day bootstrap.
        warmup = pd.bdate_range(end="2023-12-29", periods=20)
        holdout = pd.bdate_range("2024-01-02", periods=23)
        self.signals = holdout[:-1]
        self.executions = holdout[1:]
        dates = warmup.append(holdout)
        prices, targets = [], []
        for ticker in TICKERS:
            moves = pd.Series(0.0, index=dates)
            if ticker == "SPY":
                moves.loc[:] = 0.90
            else:
                moves.iloc[-1] = -0.06 if ticker in ("A", "B", "C") else 0.06
            close = 100 * (1 + moves.to_numpy())
            prices.append(pd.DataFrame({
                "date": dates, "ticker": ticker, "open": 100, "close": close,
                "adjusted_close": close, "high": np.maximum(100, close),
                "low": np.minimum(100, close), "volume": 1000,
            }))
            targets.append(pd.DataFrame({
                "date": self.signals, "ticker": ticker, "target": np.log1p(moves.loc[self.executions].to_numpy()),
            }))
        pd.concat(prices).to_csv(self.prices, index=False)
        self.targets = pd.concat(targets, ignore_index=True)

    def run_cli(self, script, *arguments):
        result = subprocess.run([sys.executable, SRC / script, *arguments], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_standalone_matches_evaluator_and_hand_calculated_trades(self):
        matched = self.root / "matched"
        matched.mkdir()
        # Flat scores for an ordinary model exercise ticker ordering on ties.
        tabular = [self.targets.assign(model=name, prediction=0.0) for name in ("zero_return", "ridge")]
        pd.concat(tabular).to_csv(matched / "tabular_predictions.csv", index=False)
        rank = {ticker: i for i, ticker in enumerate(TICKERS)}
        for model in ("patch_transformer", "pca_transformer"):
            predictions = self.targets.assign(model=model, prediction=self.targets["ticker"].map(rank))
            for seed in (7, 17, 27):
                predictions.to_csv(matched / f"{model}_{seed}.csv", index=False)

        self.run_cli("evaluate_matched.py", "--prices", self.prices, "--results", matched)
        standalone = self.root / "standalone" / "returns.csv"
        self.run_cli("backtest.py", "--prices", self.prices, "--predictions", matched / "matched_predictions.csv",
                     "--output", standalone)

        keys = ["date", "model"]
        fields = ["gross_return", "turnover", "net_return"]
        daily = pd.read_csv(matched / "backtest_returns.csv", parse_dates=["date", "target_date"])
        self.assertListEqual(list(daily.columns), keys + fields + ["target_date"])
        execution_by_signal = pd.Series(self.executions, index=self.signals)
        pd.testing.assert_series_equal(daily["target_date"], daily["date"].map(execution_by_signal), check_names=False)
        expected = daily.set_index(keys)[fields].sort_index()

        traded = ["ridge", "patch_transformer", "pca_transformer"]
        self.assertEqual(set(daily["model"]), set(traded + ["equal_weight_long_only", "cash"]))
        for model in traded + ["equal_weight_long_only", "cash"]:
            with self.subTest(model=model):
                returns = expected.xs(model, level="model")
                self.assertListEqual(returns.index.tolist(), self.signals.tolist())
                # Only the final execution moves the stocks. The long basket gains 6% and the short basket falls 6%.
                gross = np.zeros(len(returns))
                if model in traded:
                    gross[-1] = 0.06
                turnover = 0.0 if model == "cash" else 2.0
                np.testing.assert_allclose(returns["gross_return"], gross, atol=1e-12)
                np.testing.assert_allclose(returns["turnover"], turnover)
                np.testing.assert_allclose(returns["net_return"], gross - turnover * 0.001, atol=1e-12)

        actual = pd.read_csv(standalone, parse_dates=["date"]).set_index(keys)[fields].sort_index()
        pd.testing.assert_frame_equal(actual, expected)
        matched_summary = pd.read_csv(matched / "backtest_metrics.csv").set_index("model").sort_index()
        standalone_summary = pd.read_csv(standalone.with_name("backtest_metrics.csv")).set_index("model").sort_index()
        pd.testing.assert_frame_equal(standalone_summary, matched_summary)
        forecasts = pd.read_csv(matched / "matched_predictions.csv")
        self.assertEqual(len(forecasts), len(self.signals) * 7 * 4)
        self.assertIn("SPY", set(forecasts["ticker"]))
        self.assertIn("zero_return", set(forecasts["model"]))

    def test_zero_only_input_keeps_cash_and_stock_benchmark_on_signal_dates(self):
        predictions = self.root / "zero.csv"
        signals = self.signals[-2:]
        self.targets[self.targets["date"].isin(signals)].assign(model="zero_return", prediction=0.0).to_csv(
            predictions, index=False
        )
        output = self.root / "zero_returns.csv"
        self.run_cli("backtest.py", "--prices", self.prices, "--predictions", predictions,
                     "--output", output, "--cost-bps", "3")
        daily = pd.read_csv(output, parse_dates=["date"]).set_index(["model", "date"]).sort_index()
        expected_keys = pd.MultiIndex.from_product([["cash", "equal_weight_long_only"], signals],
                                                   names=["model", "date"])
        self.assertListEqual(daily.index.tolist(), expected_keys.tolist())
        np.testing.assert_allclose(daily.loc["cash", ["gross_return", "turnover", "net_return"]], 0.0)
        benchmark = daily.loc["equal_weight_long_only"]
        np.testing.assert_allclose(benchmark["gross_return"], 0.0, atol=1e-12)
        np.testing.assert_allclose(benchmark["turnover"], 2.0)
        np.testing.assert_allclose(benchmark["net_return"], -0.0006, atol=1e-12)
        summary = pd.read_csv(output.with_name("backtest_metrics.csv")).set_index("model")
        self.assertEqual(summary.loc["cash", "days"], len(signals))
        self.assertEqual(summary.loc["cash", "cumulative_net_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
