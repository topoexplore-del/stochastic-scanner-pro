"""
Calibration self-test using GARCH-simulated realistic market data.
This runs offline (no network) and demonstrates that the SPE's
probability estimates are well-calibrated against realized outcomes.

The synthetic data includes volatility clustering and fat tails,
similar to real equity markets.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_projector import StochasticProjector


def generate_garch_series(n_days=800, omega=5e-6, alpha=0.08, beta=0.90,
                          drift=0.0004, seed=None):
    """
    Generate a realistic price series with GARCH(1,1) volatility and
    t-student innovations (fat tails). Close to how real markets behave.
    """
    if seed is not None:
        np.random.seed(seed)

    sigma2 = np.zeros(n_days)
    returns = np.zeros(n_days)
    # Start from unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)

    # Sample shocks from t-student (df=6: realistic for equities)
    shocks = np.random.standard_t(df=6, size=n_days) / np.sqrt(6 / (6 - 2))

    for t in range(1, n_days):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = drift + np.sqrt(sigma2[t]) * shocks[t]

    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def _simple_atr(highs, lows, closes, period=14):
    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).rolling(period).mean().bfill()


def walk_forward(prices, horizon=21, min_train=252, rebalance=5, n_sims=3000):
    """Walk-forward calibration on a single series."""
    n = len(prices)
    results = []

    # Build fake OHLC from close (add small hi/lo noise)
    highs = prices * 1.008
    lows = prices * 0.992
    closes = prices.copy()

    for t in range(min_train, n - horizon, rebalance):
        train_closes = closes[:t]
        train_highs = highs[:t]
        train_lows = lows[:t]

        train_df = pd.DataFrame({
            "Close": train_closes,
            "High": train_highs,
            "Low": train_lows,
            "Volume": np.ones(t) * 1e6,
        })

        atr = _simple_atr(train_highs, train_lows, train_closes)
        ind = {"adx": 20.0, "rsi": 50.0, "atr": atr, "rel_vol": 1.0}

        try:
            sp = StochasticProjector(train_df, ind)
            mc = sp.monte_carlo(horizon=horizon, n_sims=n_sims)
            pred = mc["prob_up"] / 100.0
            future_price = closes[t + horizon - 1]
            realized = 1 if future_price > closes[t - 1] else 0

            results.append({
                "pred": pred, "realized": realized,
                "vol_method": mc["vol_method"],
                "reliable": mc["reliable"],
            })
        except Exception as e:
            print(f"  t={t} skip: {e}")

    return results


def run_calibration_test(n_series=20, n_days=1250, horizon=21,
                         rebalance=10, seed=42):
    """Generate multiple GARCH series and run calibration on each."""
    all_results = []
    print(f"Generating {n_series} synthetic market series "
          f"({n_days} days each, GARCH(1,1) + t-student)...")

    for i in range(n_series):
        # Vary parameters slightly per series for realism
        sub_seed = seed + i
        np.random.seed(sub_seed)
        alpha = np.random.uniform(0.05, 0.12)
        beta = np.random.uniform(0.83, 0.92)
        drift = np.random.uniform(-0.0002, 0.0008)
        if alpha + beta >= 0.99:
            beta = 0.98 - alpha
        omega = np.random.uniform(2e-6, 8e-6)

        prices = generate_garch_series(
            n_days=n_days, omega=omega, alpha=alpha, beta=beta,
            drift=drift, seed=sub_seed,
        )
        results = walk_forward(
            prices, horizon=horizon, min_train=252,
            rebalance=rebalance, n_sims=2000,
        )
        print(f"  Series {i+1:2d}/{n_series}: {len(results)} predictions "
              f"(drift={drift*252*100:+.1f}%/yr, "
              f"alpha={alpha:.2f}, beta={beta:.2f})")
        all_results.extend(results)

    return all_results


def calibration_report(all_results):
    preds = np.array([r["pred"] for r in all_results])
    reals = np.array([r["realized"] for r in all_results])
    n = len(preds)
    if n == 0:
        print("No results!")
        return

    print("\n" + "=" * 72)
    print(f"CALIBRATION REPORT — {n} out-of-sample predictions")
    print("=" * 72)

    bins = np.array([0.0, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01])
    print(f"\n{'Bin':<14} {'N':<6} {'Avg Pred':<12} {'Empirical':<12} "
          f"{'Error':<10} {'Label'}")
    print("-" * 72)

    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (preds >= lo) & (preds < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = preds[mask].mean()
        empirical = reals[mask].mean()
        err = abs(avg_pred - empirical)
        ece += (n_bin / n) * err

        if err < 0.03:   label = "excellent"
        elif err < 0.07: label = "good"
        elif err < 0.12: label = "acceptable"
        else:            label = "poor"

        print(f"[{lo:.2f},{hi:.2f})  {n_bin:<6} {avg_pred:<12.3f} "
              f"{empirical:<12.3f} {err:<10.3f} {label}")

    brier = np.mean((preds - reals) ** 2)
    directional_acc = np.mean((preds > 0.5) == (reals > 0.5))

    print("\n" + "-" * 72)
    print("Aggregate metrics:")
    print(f"  Mean predicted prob_up      : {preds.mean():.3f}")
    print(f"  Empirical up rate           : {reals.mean():.3f}")
    print(f"  Expected Calibration Error  : {ece:.4f}  (0 = perfect)")
    print(f"  Brier score                 : {brier:.4f}  "
          f"(random=0.25, perfect=0)")
    print(f"  Directional accuracy        : {directional_acc:.3f}  "
          f"(random=0.5)")

    print("\nInterpretation:")
    if ece < 0.03:
        print("  >>> Excellent calibration. Probabilities can be trusted.")
    elif ece < 0.06:
        print("  >>  Good calibration. Reasonably reliable probabilities.")
    elif ece < 0.10:
        print("  >   Acceptable. Some bias but usable for decision-making.")
    else:
        print("      Poor. Recalibrate before using probabilities.")

    # Count vol methods used
    vol_methods = [r["vol_method"] for r in all_results]
    garch_pct = sum(1 for m in vol_methods if m == "garch(1,1)") / n * 100
    ewma_pct = sum(1 for m in vol_methods if m == "ewma") / n * 100
    print(f"\nVolatility forecasts used:")
    print(f"  GARCH(1,1): {garch_pct:.1f}%")
    print(f"  EWMA     : {ewma_pct:.1f}%")


if __name__ == "__main__":
    import time
    t0 = time.time()
    results = run_calibration_test(
        n_series=8, n_days=800, horizon=21, rebalance=20,
    )
    elapsed = time.time() - t0
    calibration_report(results)
    print(f"\nTotal runtime: {elapsed:.1f}s for {len(results)} predictions")
