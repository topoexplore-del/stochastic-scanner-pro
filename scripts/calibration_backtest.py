"""
COSMOS SCANNER PRO — Calibration Backtester
============================================
Tests whether the Stochastic Projection Engine's estimated probabilities
match realized outcomes on historical data.

Methodology:
  1. Walk through real historical data of N tickers.
  2. At each rebalance date, run SPE to compute prob_up(21d).
  3. Record the model's prediction AND the actual realized outcome
     21 trading days later.
  4. Bin predictions in buckets (e.g., [0.4, 0.5), [0.5, 0.6), ...).
  5. For each bucket, compute:
        avg predicted probability (what the model said)
        empirical frequency      (what actually happened)
     A well-calibrated model has avg_pred ≈ empirical in every bucket.
  6. Report:
        - Reliability diagram (text-based)
        - Brier score (lower = better; 0 = perfect, 0.25 = worthless)
        - Directional accuracy
        - Expected Calibration Error (ECE)

CRITICAL: This is true out-of-sample testing — the model only sees
data up to time t when making the prediction. The outcome at t+21 is
withheld from the model.

Usage:
    python calibration_backtest.py
    python calibration_backtest.py --tickers SPY,QQQ,AAPL --years 5
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from stochastic_projector import StochasticProjector

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM",
                   "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
                   "JPM", "V", "WMT", "XOM", "JNJ"]


def _rsi(closes, period=14):
    d = np.diff(closes)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    if len(up) < period:
        return pd.Series([50.0] * len(closes))
    roll_up = pd.Series(up).rolling(period).mean()
    roll_dn = pd.Series(dn).rolling(period).mean()
    rs = roll_up / (roll_dn + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    # prepend to match original length
    rsi = pd.concat([pd.Series([50.0]), rsi]).reset_index(drop=True)
    return rsi


def _atr(high, low, close, period=14):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr1 = h - l
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum.reduce([tr1[1:], tr2, tr3])])
    return pd.Series(tr).rolling(period).mean().fillna(method="bfill")


def walk_forward_backtest(ticker, years=5, horizon=21, rebalance_every=5,
                          min_train_days=252):
    """
    For `ticker`:
      - Download `years` of daily data.
      - Starting at day `min_train_days`, every `rebalance_every` days,
        make a projection and record prediction + realized outcome.
    Returns list of (predicted_prob_up, realized_up, confidence_tag).
    """
    if not HAS_YF:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    hist = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if hist is None or len(hist) < min_train_days + horizon + 10:
        return []

    hist = hist.dropna(subset=["Close"])
    n = len(hist)
    results = []

    # Walk forward
    for t in range(min_train_days, n - horizon, rebalance_every):
        train = hist.iloc[:t].copy()
        current_price = float(train["Close"].iloc[-1])
        future_price = float(hist["Close"].iloc[t + horizon - 1])
        went_up = 1 if future_price > current_price else 0

        # Approximate indicators from training slice
        closes = train["Close"].values
        highs = train["High"].values
        lows = train["Low"].values

        atr_series = _atr(highs, lows, closes)
        rsi_series = _rsi(closes)

        # Simple ADX proxy: avg absolute price change / atr
        if len(closes) >= 30 and atr_series.iloc[-1] > 0:
            abs_change = np.mean(np.abs(np.diff(closes[-30:])))
            adx_proxy = min(50, (abs_change / atr_series.iloc[-1]) * 30)
        else:
            adx_proxy = 20.0

        ind = {
            "adx": float(adx_proxy),
            "rsi": float(rsi_series.iloc[-1]),
            "atr": atr_series,
            "rel_vol": 1.0,
        }

        try:
            sp = StochasticProjector(train, ind)
            mc = sp.monte_carlo(horizon=horizon, n_sims=3000)  # 3k for speed
            pred_prob_up = mc["prob_up"] / 100.0  # to 0-1
            results.append({
                "date": str(train.index[-1].date()),
                "pred": pred_prob_up,
                "realized": went_up,
                "current": current_price,
                "future": future_price,
                "return_pct": (future_price / current_price - 1) * 100,
                "reliable": mc["reliable"],
                "vol_method": mc["vol_method"],
            })
        except Exception as e:
            print(f"  {ticker} skip t={t}: {e}")
            continue

    return results


def calibration_report(all_results, out_csv=None):
    """
    Print a calibration report and optionally save to CSV.
    """
    preds = np.array([r["pred"] for r in all_results])
    reals = np.array([r["realized"] for r in all_results])
    n = len(preds)
    if n == 0:
        print("No results — nothing to analyze.")
        return

    print("\n" + "=" * 70)
    print(f"CALIBRATION REPORT — {n} predictions")
    print("=" * 70)

    # Binning
    bins = np.array([0.0, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01])
    print(f"\n{'Bin':<14} {'N':<6} {'Avg Pred':<12} {'Empirical':<12} "
          f"{'Error':<10} {'Label'}")
    print("-" * 70)

    ece = 0.0  # Expected Calibration Error
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (preds >= lo) & (preds < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = preds[mask].mean()
        empirical = reals[mask].mean()
        err = abs(avg_pred - empirical)
        weight = n_bin / n
        ece += weight * err

        if err < 0.03:
            label = "★★★ excellent"
        elif err < 0.07:
            label = "★★  good"
        elif err < 0.12:
            label = "★   acceptable"
        else:
            label = "    poor"

        print(f"[{lo:.2f},{hi:.2f}) {n_bin:<6} {avg_pred:<12.3f} "
              f"{empirical:<12.3f} {err:<10.3f} {label}")

    # Aggregate metrics
    brier = np.mean((preds - reals) ** 2)
    directional_acc = np.mean((preds > 0.5) == (reals > 0.5))

    print("\n" + "-" * 70)
    print(f"Aggregate metrics:")
    print(f"  Mean predicted prob_up      : {preds.mean():.3f}")
    print(f"  Empirical up rate           : {reals.mean():.3f}")
    print(f"  Expected Calibration Error  : {ece:.4f}  (0 = perfect)")
    print(f"  Brier score                 : {brier:.4f}  "
          f"(0 = perfect, 0.25 = random)")
    print(f"  Directional accuracy        : {directional_acc:.3f}  "
          f"(0.5 = random)")

    # Interpretation
    print("\nInterpretation:")
    if ece < 0.03:
        print("  ★★★ Excellent calibration. Probabilities can be trusted.")
    elif ece < 0.06:
        print("  ★★  Good calibration. Probabilities are reasonably reliable.")
    elif ece < 0.10:
        print("  ★   Acceptable calibration. Some bias but usable.")
    else:
        print("      Poor calibration. Recalibrate before using probabilities.")

    if brier < 0.22:
        print("  Model beats random chance meaningfully.")
    else:
        print("  Model barely beats random — prob estimates have low info.")

    # Save CSV
    if out_csv:
        df = pd.DataFrame(all_results)
        df.to_csv(out_csv, index=False)
        print(f"\nFull log: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="SPE calibration backtester")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                        help="Comma-separated tickers")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history (default 5)")
    parser.add_argument("--horizon", type=int, default=21,
                        help="Forecast horizon in trading days (default 21)")
    parser.add_argument("--rebalance", type=int, default=5,
                        help="Rebalance every N days (default 5)")
    parser.add_argument("--out-csv", default=None, help="Save raw log CSV")
    args = parser.parse_args()

    if not HAS_YF:
        print("ERROR: yfinance required. Install with: pip install yfinance")
        return

    tickers = [t.strip() for t in args.tickers.split(",")]
    all_results = []

    print(f"Running calibration backtest:")
    print(f"  Tickers  : {len(tickers)}")
    print(f"  Horizon  : {args.horizon} trading days")
    print(f"  Rebalance: every {args.rebalance} days")
    print(f"  Period   : last {args.years} years")
    print()

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker} ... ", end="", flush=True)
        results = walk_forward_backtest(
            ticker, years=args.years,
            horizon=args.horizon,
            rebalance_every=args.rebalance
        )
        if results:
            preds_here = np.array([r["pred"] for r in results])
            reals_here = np.array([r["realized"] for r in results])
            print(f"{len(results)} predictions, "
                  f"mean_pred={preds_here.mean():.2f}, "
                  f"empirical={reals_here.mean():.2f}")
            for r in results:
                r["ticker"] = ticker
            all_results.extend(results)
        else:
            print("no data")

    calibration_report(all_results, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
