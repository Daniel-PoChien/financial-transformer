"""Train chronological tabular baselines and write 2024--2025 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["return_1", "return_5", "return_20", "ma_5_ratio", "ma_20_ratio", "volatility_20", "volume_change"]


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in data.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy()
        price = group["adjusted_close"]
        log_return = np.log(price / price.shift(1))
        group["return_1"] = log_return
        group["return_5"] = np.log(price / price.shift(5))
        group["return_20"] = np.log(price / price.shift(20))
        group["ma_5_ratio"] = price / price.rolling(5).mean() - 1
        group["ma_20_ratio"] = price / price.rolling(20).mean() - 1
        group["volatility_20"] = log_return.rolling(20).std()
        group["volume_change"] = np.log(group["volume"].replace(0, np.nan) / group["volume"].shift(1))
        # Forecast made after day t's close and evaluated on tradable next-day open-to-close return.
        group["target_date"] = group["date"].shift(-1)
        group["target"] = np.log(group["close"].shift(-1) / group["open"].shift(-1))
        frames.append(group)
    return pd.concat(frames, ignore_index=True).dropna(subset=FEATURES + ["target"])


TRAIN_END = pd.Timestamp("2021-12-31")
VALID_END = pd.Timestamp("2023-12-31")
TEST_END = pd.Timestamp("2025-12-31")


def split_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    date, target_date = frame["date"], frame["target_date"]
    return (
        date.le(TRAIN_END) & target_date.le(TRAIN_END),
        date.gt(TRAIN_END) & target_date.le(VALID_END),
        date.gt(VALID_END) & target_date.le(TEST_END),
    )


def score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "directional_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
    }


def predictions_for_model(name: str, model: object, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if name == "historical_mean":
        means = train.groupby("ticker")["target"].mean()
        predicted = test["ticker"].map(means).fillna(train["target"].mean()).to_numpy()
    elif name == "zero_return":
        predicted = np.zeros(len(test))
    else:
        model.fit(train[FEATURES + ["ticker"]], train["target"])
        predicted = model.predict(test[FEATURES + ["ticker"]])
    output = test[["date", "ticker", "target"]].copy()
    output["model"] = name
    output["prediction"] = predicted
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("financial_transformer/data/ohlcv.csv"))
    parser.add_argument("--output", type=Path, default=Path("financial_transformer/results/tabular_predictions.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.data, parse_dates=["date"])
    featured = make_features(data)
    train_mask, _, test_mask = split_masks(featured)
    train, test = featured[train_mask], featured[test_mask]
    if train.empty or test.empty:
        raise ValueError("Expected non-empty train and test partitions")

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ridge = Pipeline([
        ("features", ColumnTransformer([("numeric", numeric, FEATURES), ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"])])),
        ("model", Ridge(alpha=10.0)),
    ])
    boosted = Pipeline([
        ("features", ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES), ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ticker"])])),
        ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=7)),
    ])
    models = {"zero_return": None, "historical_mean": None, "ridge": ridge, "hist_gradient_boosting": boosted}
    predictions = [predictions_for_model(name, model, train, test) for name, model in models.items()]
    result = pd.concat(predictions, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    scores = result.groupby("model").apply(lambda frame: pd.Series(score(frame["target"], frame["prediction"])), include_groups=False)
    scores.to_csv(args.output.with_name("tabular_metrics.csv"))
    print(scores.round(6).to_string())


if __name__ == "__main__":
    main()
