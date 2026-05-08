"""
Generate a demo snapshot.json with synthetic GARCH data.
Used when yfinance is not available (e.g., for initial deploy
or sandbox environments).
"""
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_projector import StochasticProjector


def generate_price_series(n_days=500, seed=42, drift=0.0005, initial=100):
    """Simulate a realistic GARCH(1,1) + t-student price series."""
    np.random.seed(seed)
    omega, alpha, beta = 5e-6, 0.08, 0.90
    sigma2 = np.zeros(n_days)
    returns = np.zeros(n_days)
    sigma2[0] = omega / (1 - alpha - beta)
    shocks = np.random.standard_t(df=6, size=n_days) / np.sqrt(6 / 4)
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = drift + np.sqrt(sigma2[t]) * shocks[t]
    return initial * np.exp(np.cumsum(returns))


def make_ticker(ticker, seed, drift=0.0005, initial=100):
    """Generate a full ticker dict as if it came from yfinance."""
    prices = generate_price_series(500, seed=seed, drift=drift, initial=initial)
    # Use RangeIndex — avoids business-day calendar mismatch
    hist = pd.DataFrame({
        "Open": prices * 0.998,
        "High": prices * 1.012,
        "Low":  prices * 0.988,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, 500),
    })

    # Build indicators inline
    h, l, c = hist["High"].values, hist["Low"].values, hist["Close"].values
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = pd.Series(tr, index=hist.index).rolling(14).mean().bfill()

    returns = np.diff(c)
    gains = np.where(returns > 0, returns, 0.0)
    losses = np.where(returns < 0, -returns, 0.0)
    avg_gain = pd.Series(gains).rolling(14).mean().iloc[-1]
    avg_loss = pd.Series(losses).rolling(14).mean().iloc[-1]
    rsi = 100 - 100 / (1 + avg_gain / max(avg_loss, 1e-9))
    adx = min(50, (np.mean(np.abs(np.diff(c[-30:]))) / atr.iloc[-1]) * 30) if atr.iloc[-1] > 0 else 20
    rel_vol = hist["Volume"].iloc[-1] / hist["Volume"].iloc[-50:].mean()

    ind = {"atr": atr, "rsi": float(rsi), "adx": float(adx), "rel_vol": float(rel_vol)}

    # Run SPE
    sp = StochasticProjector(hist, ind)
    ens = sp.ensemble_projection(horizon=21)
    multi = sp.multi_horizon_projection()
    mc = ens["monte_carlo"]
    regime = ens["regime"]

    return {
        "ticker": ticker,
        "close": round(float(c[-1]), 4),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rsi": round(float(rsi), 2),
        "adx": round(float(adx), 2),
        "rel_vol": round(float(rel_vol), 2),
        "atr": round(float(atr.iloc[-1]), 4),
        "target": ens["target"],
        "upside_pct": ens["upside_pct"],
        "confidence": ens["confidence"],
        "prob_up": ens["prob_up"],
        "prob_tp3": ens["prob_tp3"],
        "prob_tp5": ens["prob_tp5"],
        "prob_tp10": ens["prob_tp10"],
        "prob_drop_5": ens["prob_drop_5"],
        "var_95": mc["var_95"],
        "cvar_95": mc["cvar_95"],
        "ci_68_low": mc["ci_68_low"],
        "ci_68_high": mc["ci_68_high"],
        "ci_95_low": mc["ci_95_low"],
        "ci_95_high": mc["ci_95_high"],
        "volatility_annualized": mc["volatility_annualized"],
        "skewness": mc["skewness"],
        "kurtosis": mc["kurtosis"],
        "method": mc["method"],
        "vol_method": mc["vol_method"],
        "reliable": mc["reliable"],
        "regime": regime["regime"],
        "regime_desc": regime["description"],
        "atr_ratio": regime["atr_ratio"],
        "horizons": {
            k: {
                "days": v["days"],
                "expected": v["expected"],
                "upside_pct": v["upside_pct"],
                "prob_up": v["prob_up"],
                "ci_68_low": v["ci_68_low"],
                "ci_68_high": v["ci_68_high"],
            }
            for k, v in multi.items()
        },
        "history_60d": [round(float(p), 4) for p in c[-60:]],
        "process_ms": 0,
    }


def main():
    # Demo universe — synthetic tickers with varied characteristics
    groups = {
        "US Large Cap": [
            ("AAPL",  1, 0.0006, 180),
            ("MSFT",  2, 0.0007, 420),
            ("GOOGL", 3, 0.0005, 165),
            ("AMZN",  4, 0.0004, 195),
            ("META",  5, 0.0008, 510),
            ("NVDA",  6, 0.0010, 820),
            ("TSLA",  7, 0.0003, 240),
        ],
        "ETFs": [
            ("SPY",   100, 0.0004, 560),
            ("QQQ",   101, 0.0005, 480),
            ("DIA",   102, 0.0003, 420),
            ("IWM",   103, 0.0002, 220),
            ("GLD",   104, 0.0002, 255),
        ],
        "Finance": [
            ("JPM",   200, 0.0004, 230),
            ("BAC",   201, 0.0003,  46),
            ("GS",    202, 0.0005, 505),
            ("V",     203, 0.0005, 295),
        ],
    }

    results = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0-demo",
        "groups": {},
        "n_tickers": 0,
        "n_reliable": 0,
        "n_failed": 0,
        "note": "DEMO snapshot generated from synthetic GARCH data (Yahoo not available).",
    }

    for group_name, tickers in groups.items():
        print(f"═══ {group_name} ═══")
        group_res = []
        for ticker, seed, drift, initial in tickers:
            print(f"  {ticker}...", end=" ", flush=True)
            try:
                r = make_ticker(ticker, seed, drift, initial)
                group_res.append(r)
                results["n_tickers"] += 1
                if r["reliable"]:
                    results["n_reliable"] += 1
                print(f"prob_up={r['prob_up']:.0f}% regime={r['regime']}")
            except Exception as e:
                print(f"FAILED: {e}")
                results["n_failed"] += 1
        results["groups"][group_name] = group_res

    out = os.path.join(os.path.dirname(__file__), "..", "data", "snapshot.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote demo snapshot: {out}")
    print(f"Tickers: {results['n_tickers']} | Reliable: {results['n_reliable']}")
    print(f"Size: {os.path.getsize(out)/1024:.1f} KB")


if __name__ == "__main__":
    main()
