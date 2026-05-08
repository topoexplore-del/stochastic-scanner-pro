# Stochastic Scanner Pro

Independent stochastic projection system for financial markets. Runs Monte Carlo simulations with GARCH(1,1) time-varying volatility and t-student fat-tailed distributions to estimate calibrated probabilities for price movements across multiple horizons.

**This is NOT a "quantum" or "AI" system.** It is standard financial mathematics (Monte Carlo since the 1970s, GARCH since the 1980s) implemented honestly with sample-size checks, regime detection, and built-in calibration backtesting.

## What it does

For each ticker in the universe:
- Fetches 2 years of daily price history from Yahoo Finance
- Fits GARCH(1,1) to log returns via Maximum Likelihood Estimation
- Fits t-student distribution (falls back to normal if sample is too small)
- Detects market regime (STRONG_TREND, TREND, COMPRESSION, RANGE, HIGH_VOL)
- Runs 10,000 Monte Carlo paths with time-varying volatility
- Computes:
  - Prob UP at 5 horizons (1w, 2w, 1m, 3m, 6m)
  - Prob TP+3%, +5%, +10%
  - Prob drop -5%
  - Value at Risk (VaR) 95%
  - Conditional VaR (CVaR) 95%
  - 68% and 95% confidence intervals
  - Skewness and kurtosis of projected distribution

Results render in a web dashboard with per-ticker cards showing all metrics.

## What it is NOT

- Not a predictor of the future
- Not a "quantum" anything — it is classical statistics
- Not a source of financial advice
- Not a replacement for your own analysis

## Files

```
stochastic-scanner-pro/
├── index.html                        # Web dashboard
├── requirements.txt
├── README.md
├── .github/workflows/refresh.yml     # Daily auto-rebuild
├── scripts/
│   ├── stochastic_projector.py       # Core SPE engine
│   ├── build_data.py                 # Pipeline: fetch → SPE → snapshot.json
│   ├── calibration_backtest.py       # Real-data walk-forward validation
│   └── calibration_test_offline.py   # Synthetic GARCH validation (offline)
└── data/
    └── snapshot.json                 # Generated output consumed by index.html
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build snapshot (takes ~3-5 minutes for default universe of ~100 tickers)
python scripts/build_data.py --out-dir data

# 3. Preview locally
python -m http.server 8000
# Open http://localhost:8000
```

## Validate calibration (IMPORTANT)

Before trusting the probabilities, run the calibration tests:

**Offline (no internet, synthetic GARCH data):**
```bash
python scripts/calibration_test_offline.py
```

**Real data:**
```bash
python scripts/calibration_backtest.py --tickers SPY,QQQ,AAPL,MSFT --years 5
```

The output tells you:
- **ECE (Expected Calibration Error)**: lower is better. < 0.03 excellent, > 0.10 unreliable.
- **Brier score**: lower is better. Random = 0.25. Perfect = 0.
- **Reliability diagram**: shows whether the 60% predictions actually happen 60% of the time.

Rule: if ECE > 0.10 for your ticker universe, do NOT treat the probabilities as trustworthy.

## Custom ticker list

```bash
python scripts/build_data.py --tickers AAPL,MSFT,NVDA,TSLA --out-dir data
```

Or edit `TICKER_GROUPS` in `scripts/build_data.py` to customize the default universe.

## Deploy to GitHub Pages

1. Push this repo to GitHub.
2. Settings → Pages → Deploy from `main` branch.
3. The included GitHub Action refreshes `data/snapshot.json` daily at 21:30 UTC (after US market close).
4. Manual refresh: Actions → Refresh Stochastic Snapshot → Run workflow.

## How to read the cards

Each ticker card shows:

- **Prob UP**: % of simulated paths that ended above current price at horizon. This is the model's estimate. A 65% estimate does NOT mean "the price will definitely go up". It means "in a well-calibrated model, outcomes with this estimate should happen ~65% of the time, on average".
- **Expected target**: mean of 10,000 simulated terminal prices. Not a point prediction — the center of a distribution.
- **68% CI**: range where 68% of simulations landed. Probability that actual price falls here is ~68% (if model is calibrated).
- **VaR 95%**: 5th percentile of simulated prices. "In the worst 5% of cases, price ≥ this."
- **CVaR 95%**: average of the worst 5%. Always worse than VaR. Used for position sizing.
- **Regime badge**: current market state (helps contextualize probabilities).
- **★ reliable / ⚠ limited data**: star means ≥60 days of data → t-student fit is valid.
- **GARCH / EWMA badge**: which volatility model was used. GARCH is preferred.

## Warning signs in cards

- **"⚠ limited data"** — don't trust the probabilities
- **Regime = HIGH_VOL** — position sizes should be reduced
- **Prob UP > 75%** — SUSPICIOUS on liquid tickers. Probably pathological data (recent split, earnings event distorting return distribution). Double-check before acting.
- **VaR and CVaR are very close** — distribution has thin tails, model may be underestimating risk.

## License

No warranty expressed or implied. Do not risk money you cannot afford to lose. This software is provided as-is for research purposes. It is not investment advice, a trading system, or a guarantee of any outcome.
