# Transformer stock-return benchmark

The primary model is a patch Transformer whose inputs are principal components of seven daily market features. The full-feature Transformer uses the same encoder and training settings and measures the effect of PCA. The project also runs zero return, per-ticker historical mean, ridge regression, histogram gradient boosting, and a frozen Chronos-Bolt-small benchmark. Chronos uses the median forecast from a univariate history of the same open-to-close return it predicts.

## Data and compute

The verified Yahoo Finance snapshot has 30,426 rows, 2,766 sessions and 11 tickers, covering 2015-01-02 to 2025-12-31. The panel has no duplicate ticker/dates, missing cells, or invalid OHLC relationships. The snapshot matches the SHA-256 in [the data manifest](results/data_manifest.json). The fixed universe comprises AAPL, CAT, DIS, JNJ, JPM, KO, MSFT, PG, WMT, XOM, and SPY.

The GPU runs use PyTorch 2.14.1 with the Apple M4 Pro's 20-core Metal GPU through MPS. The sandbox can hide MPS from PyTorch. Run from a local terminal that exposes Metal, or from a CUDA environment. The training command fails if GPU access was requested and PyTorch can only see CPU.

The raw snapshot is excluded from Git. If needed, acquire it using `src/download_data.py` and record its hash with `src/make_manifest.py`. Vendor revisions can yield a different hash for a fresh download.

## Matched holdout

- Forecast next-session open-to-close log return using information through the signal day's close.
- Fit models and transformations on 2015–2021. Use 2022–2023 for neural early stopping. Evaluate fixed models in 2024–2025 with no test-period refits. Drop labels crossing split boundaries and allow prior observations to warm up the first held-out context.
- Standardize seven features using training means and standard deviations. Fit PCA on the standardized training rows with a preselected 90% explained-variance threshold. Both Transformers use 60 trailing sessions, five-day patches, and seeds 7, 17, and 27. Average the three predictions for each model. Tabular baselines use the current row of engineered features with ticker identity for ridge and boosting.
- Use exactly the same forecast observations for MAE, RMSE, and three-class directional accuracy. Report per-asset and year metrics, seed-level stability, and pointwise 95% paired MAE intervals using 5,000 resamples of 20-session blocks jointly across assets.
- Exclude SPY from portfolio formation. Rank the ten stocks at the signal close and enter the next open with 50% gross long exposure across the top three and 50% gross short across the bottom three. Close all positions that afternoon. Apply 10 bps per side on total daily turnover of 200% of capital. Apply the same round-trip cost to the daily equal-weight long-only benchmark. Represent the zero signal with cash. The simulation uses recorded open and close prices and excludes borrow fees and additional slippage.

## Reproduce

Run from the repository root in a Python 3.11 environment:

```bash
python -m pip install -r financial_transformer/requirements.txt
python financial_transformer/src/verify_environment.py
out=financial_transformer/results/matched
python financial_transformer/src/run_tabular_baselines.py --output "$out/tabular_predictions.csv"
for seed in 7 17 27; do
  python financial_transformer/src/train_patch_transformer.py --require-gpu \
    --seed "$seed" --output "$out/patch_transformer_${seed}.csv"
  python financial_transformer/src/train_patch_transformer.py --require-gpu \
    --seed "$seed" --pca-variance 0.90 --output "$out/pca_transformer_${seed}.csv"
done
python financial_transformer/src/run_chronos_bolt.py --output "$out/chronos_bolt_predictions.csv"
python financial_transformer/src/evaluate_matched.py
```

The evaluator checks each model's observation keys and targets against the full holdout. It writes forecast metrics, portfolio metrics, daily returns, and paired error intervals under `results/matched`. Neural run JSON files record device, preprocessing, seed, and configuration.

## Benchmark results

The evaluator matched 5,511 forecasts per model across 501 signal dates and 11 assets. The PCA fit retained four components explaining 90.86% of training variance. Each trading strategy used the same 501 execution sessions from January 3, 2024 through December 31, 2025.

| Model | MAE (bps) | RMSE (bps) | Directional accuracy | Cumulative net return |
| --- | ---: | ---: | ---: | ---: |
| PCA Transformer | 82.29 | 116.98 | 49.05% | -65.25% |
| Full-feature Transformer | 82.42 | 117.04 | 48.63% | -68.33% |
| Historical mean | 81.98 | 116.92 | 51.79% | -62.99% |
| Zero return / cash | 82.04 | 116.95 | 0.31% | 0.00% |
| Ridge | 82.12 | 116.83 | 50.90% | -64.68% |
| Histogram gradient boosting | 82.20 | 117.15 | 51.52% | -63.76% |
| Chronos-Bolt-small | 84.67 | 121.39 | 49.66% | -62.78% |
| Equal-weight long-only | | | | -48.85% |

PCA lowered MAE by 0.13 bps relative to the full-feature Transformer. The paired 95% interval for PCA minus full-feature MAE was -0.35 to +0.08 bps. The historical mean had the lowest forecast MAE. The PCA strategy averaged -0.94 bps gross per session before its 20 bps round-trip cost. Among the three paired training seeds, PCA improved MAE and net backtest return in two runs each.

## Five-minute AAPL pilot

This separate experiment uses the [GetData AAPL one-minute sample](https://github.com/getdata-finance/aapl-1m-ohlcv-stocks-historical-data). Save the snapshot as `data/aapl_1m.csv` and verify its SHA-256 against `results/intraday/run.json`. The source updates its public sample. Its volume column is a count of price updates. Almost all contiguous bar opens in this snapshot match the prior bar's close to CSV precision, so the opening-price execution proxy warrants particular caution.

Run from the repository root:

```bash
python financial_transformer/src/run_intraday.py \
  --data financial_transformer/data/aapl_1m.csv \
  --output financial_transformer/results/intraday --require-gpu
```

The run splits whole market sessions chronologically (60% training, 20% validation, 20% test), uses only contiguous regular-session bars, and predicts the log close-to-close return five minutes after each signal. Multiplying the last observed close by the exponential of the forecast converts it to a price. The matched benchmark compares three-seed ensembles for the full-feature and PCA models against unchanged price and the training historical mean.

The simulated strategy evaluates signals every five minutes, enters at the bar open one minute after the signal and exits at the close observed five minutes after the signal. It trades long or short only when the predicted move exceeds the specified round-trip cost. The CSV reports 0, 2, 5 and 10 bps scenarios. The zero-cost scenario uses the sign of the forecast, so its trade count can differ from the cost scenarios. Fills use historical bar values as proxies for execution prices.
