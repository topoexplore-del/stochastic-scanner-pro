"""
Stochastic Scanner Pro — Data Builder
======================================
Generates snapshot.json with Monte Carlo projections for a universe of tickers.

Each ticker output includes:
  - 2 years of daily price history (for SPE model)
  - Basic technical indicators (ATR, ADX proxy, RSI) used by SPE regime detection
  - Full SPE output: expected, prob_up, CI, VaR, CVaR, multi-horizon, regime

NO fundamental data. NO composite scores. NO market analysis layers.
This project is purely about stochastic projections.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_projector import StochasticProjector

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─────────────────────────────────────────────────────────────
# Ticker universe — organized by group for the web UI
# ─────────────────────────────────────────────────────────────
TICKER_GROUPS = {
    "US Large Cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "BRK-B", "V", "JPM", "JNJ", "WMT", "PG", "MA", "HD",
        "UNH", "XOM", "BAC", "KO", "PFE", "CVX", "ABBV", "CSCO",
        "PEP", "AVGO", "LLY", "TMO", "COST", "MRK", "ADBE", "VRT", "POWL", "ETN", "ANET", "MPWR", "PWR", "CAT", "FCX",
        "NVDA","PLTR","AVGO","AMD","LMT","NOC","CEG","SMCI",
        "GE", "ROK", "URI", "DE", "ORCL", "CRM", "ADBE", "NOW",
        "PANW", "CRWD", "SNOW", "NET", "DDOG", "INTU", "CDNS", "SNPS", "FTNT", "ZS",
        "WDAY", "TEAM", "HUBS", "DOCU", "VEEV", "ANSS", "CPAY", "IT", "KEYS", "TYL",
        "EPAM", "PAYC", "MANH", "MPWR", "NXPI", "MCHP", "SWKS", "QRVO", "ZBRA", "TER",
        "TRMB", "GDDY", "GEN", "CTSH", "WIT", "ACN", "IBM", "CSCO", "HPQ", "HPE", "DELL",
        "JPM", "V", "MA", "GS", "MS", "BLK", "SCHW", "AXP", "C", "BAC", "WFC", "USB",
        "PNC", "TFC", "COF", "ICE", "CME", "SPGI", "MCO", "MSCI", "FIS", "FISV", "GPN",
        "AIG", "MET", "PRU", "AFL", "ALL", "TRV", "CB", "AON", "MMC", "AJG", "CINF", "BRO",
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ISRG", "VRTX", "REGN",
        "AMGN", "GILD", "MDT", "SYK", "BSX", "EW", "ZBH", "BAX", "BDX", "HOLX", "DXCM",
        "IDXX", "MTD", "A", "WAT", "IQV", "CRL", "TECH", "ALGN", "PODD", "INCY",
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "HES",
        "OXY", "DVN", "HAL", "FANG", "CTRA", "APA", "TRGP", "WMB", "OKE", "KMI",
        "RTX", "GD", "BA", "LHX", "NOC", "LMT", "HII", "TXT", "HWM", "TDG", "AXON",
        "HON", "MMM", "CMI", "PH", "ITW", "TT", "EMR", "GE", "ETN", "ROK", "AME",
        "DOV", "FTV", "XYL", "NDSN", "ROP", "IEX", "GWW", "FAST", "WSO", "AOS",
        "IR", "CARR", "OTIS", "JCI", "GNRC", "HUBB", "RBC", "SNA", "WCC",
        "COST", "WMT", "HD", "LOW", "NKE", "SBUX", "MCD", "TJX", "ROST", "DG", "DLTR",
        "BKNG", "ABNB", "MAR", "HLT", "RCL", "CCL", "LVS", "WYNN", "MGM",
        "F", "GM", "APTV", "BWA", "LEA", "RL", "TPR", "GRMN", "POOL", "BBY", "TSCO",
        "ORLY", "AZO", "AAP", "KMX", "LULU", "DECK", "ON", "ULTA", "EL", "CPRI",
        "PG", "KO", "PEP", "PM", "MO", "STZ", "BF.B", "MNST", "KDP", "CLX",
        "CL", "KMB", "CHD", "SJM", "HSY", "MKC", "GIS", "CAG", "K", "HRL", "TSN", "MDLZ",
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "DLR", "VICI", "WELL",
        "AVB", "EQR", "MAA", "ESS", "UDR", "ARE", "BXP", "SLG", "VNO",
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
        "AEE","CMS","CNP","PNW","NI","EVRG","ATO","PEG",
        "LIN", "APD", "SHW", "ECL", "NUE", "STLD", "CF", "MOS", "ALB", "FMC",
        "IFF", "CE", "PPG", "VMC", "MLM", "NEM", "FCX", "AA",
        "GOOG", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO",
        "MTCH", "ZG", "PINS", "SNAP", "ROKU", "SPOT", "WBD", "PARA", "LYV", "IACI",
    ],
    "US Tech": [
        "ORCL", "CRM", "AMD", "INTC", "QCOM", "TXN", "NOW",
        "UBER", "SHOP", "PLTR", "COIN", "SNOW", "DDOG", "ZS",
        "CRWD", "NET", "PANW", "ABNB", "SPOT",
    ],
    "US Finance": [
        "GS", "MS", "C", "WFC", "AXP", "BLK", "SCHW", "PYPL",
        "SQ", "PGR", "AIG", "MET", "TRV",
    ],
    "ETFs & Indexes": [
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "EEM", "EFA",
        "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLV", "XLI",
    ],
    "International": [
        "BABA", "TSM", "NVO", "TM", "SAP", "SONY", "SHEL",
        "BP", "ASML", "UL", "HSBC",
    ],
    "LATAM": [
        "VALE", "PBR", "ITUB", "BBD", "MELI", "GGAL", "BIOX",
        "EC", "CIB", "BSAC",
    ],
}


def fetch_ticker(ticker, period="2y"):
    """Fetch 2 years of daily data for a ticker. Returns DataFrame or None."""
    if not HAS_YF:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or len(hist) < 250:  # need at least 1 year
            return None
        return hist
    except Exception as e:
        print(f"  [{ticker}] fetch error: {e}")
        return None


def compute_indicators(hist):
    """
    Compute the minimal set of indicators SPE needs for regime detection:
      - ATR (14)
      - RSI (14) as last value
      - ADX proxy as last value
      - Relative volume vs 50-period average
    """
    h, l, c = hist["High"].values, hist["Low"].values, hist["Close"].values
    v = hist["Volume"].values

    # ATR — True Range rolling mean
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr_series = pd.Series(tr, index=hist.index).rolling(14).mean().bfill()

    # RSI
    returns = np.diff(c)
    gains = np.where(returns > 0, returns, 0.0)
    losses = np.where(returns < 0, -returns, 0.0)
    avg_gain = pd.Series(gains).rolling(14).mean().iloc[-1] if len(gains) >= 14 else 0
    avg_loss = pd.Series(losses).rolling(14).mean().iloc[-1] if len(losses) >= 14 else 1e-9
    rsi = 100 - 100 / (1 + avg_gain / max(avg_loss, 1e-9))

    # ADX proxy — abs change / atr over last 30 days × 30
    if len(c) >= 30 and atr_series.iloc[-1] > 0:
        abs_change = np.mean(np.abs(np.diff(c[-30:])))
        adx = min(50, (abs_change / atr_series.iloc[-1]) * 30)
    else:
        adx = 20.0

    # Relative volume: last / mean of last 50
    if len(v) >= 50:
        rel_vol = v[-1] / np.mean(v[-50:])
    else:
        rel_vol = 1.0

    return {
        "atr": atr_series,
        "rsi": float(rsi),
        "adx": float(adx),
        "rel_vol": float(rel_vol),
    }


def process_ticker(ticker):
    """
    Full pipeline for one ticker:
      fetch → indicators → SPE projection → packaged dict.
    """
    t0 = time.time()
    hist = fetch_ticker(ticker)
    if hist is None:
        return None

    try:
        ind = compute_indicators(hist)
    except Exception as e:
        print(f"  [{ticker}] indicator error: {e}")
        return None

    try:
        sp = StochasticProjector(hist, ind)
        ens = sp.ensemble_projection(horizon=21)
        multi = sp.multi_horizon_projection()
    except Exception as e:
        print(f"  [{ticker}] SPE error: {e}")
        return None

    mc = ens["monte_carlo"]
    regime = ens["regime"]

    result = {
        "ticker": ticker,
        "close": round(float(hist["Close"].iloc[-1]), 4),
        "fetched_at": datetime.now(timezone.utc).isoformat(),

        # Technicals used by SPE
        "rsi": round(ind["rsi"], 2),
        "adx": round(ind["adx"], 2),
        "rel_vol": round(ind["rel_vol"], 2),
        "atr": round(float(ind["atr"].iloc[-1]), 4),

        # Headline SPE results (1 month default horizon)
        "target": ens["target"],
        "upside_pct": ens["upside_pct"],
        "confidence": ens["confidence"],
        "prob_up": ens["prob_up"],
        "prob_tp3": ens["prob_tp3"],
        "prob_tp5": ens["prob_tp5"],
        "prob_tp10": ens["prob_tp10"],
        "prob_drop_5": ens["prob_drop_5"],

        # Risk metrics
        "var_95": mc["var_95"],
        "cvar_95": mc["cvar_95"],
        "ci_68_low": mc["ci_68_low"],
        "ci_68_high": mc["ci_68_high"],
        "ci_95_low": mc["ci_95_low"],
        "ci_95_high": mc["ci_95_high"],

        # Diagnostics
        "volatility_annualized": mc["volatility_annualized"],
        "skewness": mc["skewness"],
        "kurtosis": mc["kurtosis"],
        "method": mc["method"],
        "vol_method": mc["vol_method"],
        "reliable": mc["reliable"],

        # Regime
        "regime": regime["regime"],
        "regime_desc": regime["description"],
        "atr_ratio": regime["atr_ratio"],

        # Multi-horizon projections
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

        # Price history for mini-chart (last 60 days, closes only)
        "history_60d": [round(float(p), 4) for p in hist["Close"].values[-60:]],

        # Timing
        "process_ms": int((time.time() - t0) * 1000),
    }
    return result


def build_snapshot(tickers=None, groups=None, out_dir="data", verbose=True):
    """
    Main entry point. Fetches all tickers, runs SPE, saves snapshot.json.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine target tickers
    if groups is None:
        groups = TICKER_GROUPS

    if tickers is not None:
        # Filter to just the given list — put them all under one group
        groups = {"Custom": list(tickers)}

    total = sum(len(v) for v in groups.values())
    print(f"Building snapshot for {total} tickers across {len(groups)} groups\n")

    results = {"built_at": datetime.now(timezone.utc).isoformat(),
               "version": "1.0",
               "groups": {},
               "n_tickers": 0,
               "n_reliable": 0,
               "n_failed": 0}

    idx = 0
    for group_name, tickers_in_group in groups.items():
        if verbose:
            print(f"═══ {group_name} ═══")
        group_results = []
        for ticker in tickers_in_group:
            idx += 1
            if verbose:
                print(f"  [{idx}/{total}] {ticker}...", end=" ", flush=True)
            r = process_ticker(ticker)
            if r is None:
                results["n_failed"] += 1
                if verbose:
                    print("FAILED")
                continue
            group_results.append(r)
            results["n_tickers"] += 1
            if r["reliable"]:
                results["n_reliable"] += 1
            if verbose:
                flag = "★" if r["reliable"] else "⚠"
                print(f"{flag} prob_up={r['prob_up']:.0f}% "
                      f"upside={r['upside_pct']:+.1f}% "
                      f"regime={r['regime']} ({r['process_ms']}ms)")

        results["groups"][group_name] = group_results

    # Save snapshot
    output_file = out_path / "snapshot.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'═'*60}")
    print(f"SUMMARY")
    print(f"  Tickers processed : {results['n_tickers']} / {total}")
    print(f"  Reliable (≥60d)   : {results['n_reliable']}")
    print(f"  Failed            : {results['n_failed']}")
    print(f"  Output            : {output_file}")
    print(f"  File size         : {os.path.getsize(output_file)/1024:.1f} KB")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stochastic Scanner Pro — Build snapshot.json from Yahoo data")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers (overrides groups)")
    parser.add_argument("--out-dir", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-ticker output")
    args = parser.parse_args()

    if not HAS_YF:
        print("ERROR: yfinance not installed.")
        print("Install with: pip install yfinance")
        sys.exit(1)

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    build_snapshot(tickers=tickers, out_dir=args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
