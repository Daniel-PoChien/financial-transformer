import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_tabular_baselines import FEATURES, make_features, split_masks
from train_patch_transformer import build_windows


class DailyTimeSplitTest(unittest.TestCase):
    def test_cross_year_label_and_first_holdout_window(self):
        last_validation = pd.Timestamp("2023-12-29")
        first_test = pd.Timestamp("2024-01-02")
        next_session = pd.Timestamp("2024-01-03")
        dates = pd.bdate_range(end=last_validation, periods=90).append(
            pd.DatetimeIndex([first_test, next_session, pd.Timestamp("2024-01-04")])
        )
        rows = []
        for day, date in enumerate(dates):
            for ticker, base, rate in [("AAA", 100, 0.0005), ("BBB", 200, -0.0003)]:
                opening = base + day
                closing = opening * (1 + rate * (day + 1))
                rows.append((date, ticker, opening, closing, closing, 1000 + day))
        raw = pd.DataFrame(rows, columns=["date", "ticker", "open", "close", "adjusted_close", "volume"])

        context = 60
        windows, metadata = build_windows(
            make_features(raw), context, np.zeros(len(FEATURES)), np.ones(len(FEATURES)), None
        )
        train, valid, test = split_masks(metadata)
        for ticker in ("AAA", "BBB"):
            with self.subTest(ticker=ticker):
                asset = metadata["ticker"].eq(ticker)
                boundary = asset & metadata["date"].eq(last_validation)
                self.assertEqual(metadata.loc[boundary, "target_date"].tolist(), [first_test])
                self.assertFalse((train | valid | test)[boundary].any())

                holdout = metadata.loc[asset & test].sort_values("date")
                first = holdout.iloc[0]
                self.assertEqual(first["date"], first_test)
                self.assertEqual(first["target_date"], next_session)
                prices = raw.loc[raw["ticker"].eq(ticker)].set_index("date")
                execution = prices.loc[next_session]
                self.assertAlmostEqual(first["target"], np.log(execution["close"] / execution["open"]))

                # Derive all 60 expected returns from this ticker's raw prices through the signal close.
                history = np.log(prices.loc[:first_test, "adjusted_close"]).diff().tail(context)
                np.testing.assert_allclose(
                    windows[first.name, :, FEATURES.index("return_1")], history, rtol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
