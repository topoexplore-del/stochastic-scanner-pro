"""
COSMOS SCANNER PRO — Stochastic Projection Engine (SPE) v18.0
================================================================
Monte Carlo projections with honest statistical calibration.

DESIGN PRINCIPLES:
  - No marketing hyperbole ("quantum", "extreme accuracy").
  - Uses rolling-window volatility (EWMA) not naive historical.
  - t-distribution for fat tails, but checks sample size.
  - Detects market regimes (trend/range/compression/high-vol).
  - Produces intervals of confidence, VaR, CVaR — industry-standard.
  - Every probability is clearly labeled as ESTIMATED, not guaranteed.

WHAT IT IS NOT:
  - It is not "quantum" in the physics sense.
  - It does not predict the future with high accuracy.
  - It is a standard Monte Carlo pricing model like those used in
    finance since the 1970s, improved with modern best practices.

WHAT IT IS:
  - A principled stochastic projection system giving calibrated
    probability estimates for price movements over multiple horizons.
  - Risk metrics (VaR, CVaR) for position sizing.
  - Regime detection to adapt projections to current market state.

FIXES vs PRIOR VERSION:
  1. EWMA volatility (not naive full-history volatility)
  2. Sample size checks before t-distribution fitting
  3. Double-counting eliminated (MC uses only price; score uses only
     indicators; they are combined transparently, not summed blindly)
  4. No yfinance .info spam — fundamentals passed in externally
  5. Clear separation: MC gives price estimates, regime gives weights,
     combined score explains HOW it was computed.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class StochasticProjector:
    """
    Monte Carlo price projector with regime-aware adjustments.

    Inputs:
        hist           : pd.DataFrame with 'Close', 'High', 'Low', 'Volume' columns
        indicators     : dict with precomputed adx, rsi, atr (as pd.Series or last value)
        fundamentals   : dict with fund_score, is_etf (optional context)

    Usage:
        sp = StochasticProjector(hist, ind, fund)
        proj = sp.ensemble_projection()
        # proj has: target, prob_up, ci_68, ci_95, var_95, cvar_95, regime, ...
    """

    # Class constants (documented thresholds, not magic numbers)
    MIN_SAMPLES_FOR_T_FIT = 60          # below this, use normal dist
    EWMA_DECAY = 0.94                   # industry-standard (RiskMetrics)
    DEFAULT_HORIZON_DAYS = 21           # ~1 month trading days
    N_SIMULATIONS_DEFAULT = 10_000
    TAIL_RISK_ALPHA = 0.05              # 5% for VaR/CVaR

    def __init__(self, hist, indicators=None, fundamentals=None):
        self.hist = hist
        self.ind = indicators or {}
        self.fund = fundamentals or {}

        self.close = hist["Close"].values
        # Log returns — standard practice
        self.returns = np.diff(np.log(self.close))
        self.returns = self.returns[np.isfinite(self.returns)]

    # ──────────────────────────────────────────────────────────────
    # FIX 1: EWMA volatility — weight recent observations more
    # ──────────────────────────────────────────────────────────────
    def _ewma_volatility(self, decay=None):
        """
        Exponentially Weighted Moving Average volatility.
        Reacts faster to regime changes than simple historical vol.
        This is the RiskMetrics standard (JP Morgan 1996).
        """
        if decay is None:
            decay = self.EWMA_DECAY
        if len(self.returns) < 2:
            return 0.02  # fallback: 2% daily vol (typical equity)

        n = len(self.returns)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted mean and variance
        mean = np.sum(weights * self.returns)
        var = np.sum(weights * (self.returns - mean) ** 2)
        return np.sqrt(var)

    # ──────────────────────────────────────────────────────────────
    # GARCH(1,1) — Generalized Autoregressive Conditional Heteroskedasticity
    # ──────────────────────────────────────────────────────────────
    # GARCH models volatility clustering: after a large move, more large
    # moves are expected; after calm, calm persists.
    # Formula: σ²(t) = ω + α·r²(t-1) + β·σ²(t-1)
    # where ω, α, β are fitted parameters; α+β < 1 for stationarity.
    # ──────────────────────────────────────────────────────────────
    def _fit_garch_11(self):
        """
        Fit GARCH(1,1) to returns using maximum likelihood estimation.
        Simplified, single-start optimization for speed.

        Returns (omega, alpha, beta, last_sigma) or None if fit fails.
        """
        r = self.returns
        if len(r) < 100:
            return None

        r_demeaned = r - np.mean(r)
        r2 = r_demeaned ** 2
        unconditional_var = np.var(r)
        if unconditional_var <= 0:
            return None

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            n = len(r_demeaned)
            # Vectorized filter: still Python-loop but numpy-only body
            sigma2 = np.empty(n)
            sigma2[0] = unconditional_var
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    return 1e10
            if not np.all(np.isfinite(sigma2)):
                return 1e10
            # Gaussian log-likelihood
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r2 / sigma2)
            return -ll

        from scipy.optimize import minimize
        x0 = [unconditional_var * 0.1, 0.1, 0.85]
        bounds = [(1e-10, unconditional_var * 2), (1e-4, 0.5), (1e-4, 0.99)]

        try:
            result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": 50})
            if not result.success:
                return None
            omega, alpha, beta = result.x

            # Final forward pass to get last sigma
            n = len(r_demeaned)
            sigma2 = np.empty(n)
            sigma2[0] = unconditional_var
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1]
            last_sigma = np.sqrt(max(sigma2[-1], 1e-10))
            return (float(omega), float(alpha), float(beta), float(last_sigma))
        except Exception:
            return None

    def _garch_forecast(self, horizon):
        """
        Forecast conditional volatility for next `horizon` periods
        using fitted GARCH(1,1).

        Returns array of forecasted sigmas (length=horizon), or None if fit fails.
        """
        fit = self._fit_garch_11()
        if fit is None:
            return None

        omega, alpha, beta, last_sigma = fit
        last_var = last_sigma ** 2
        persistence = alpha + beta

        if persistence >= 1.0:
            # Degenerate case: vol doesn't decay
            return np.full(horizon, last_sigma)

        long_run_var = omega / (1 - persistence)
        sigmas = np.zeros(horizon)
        var_t = last_var
        for t in range(horizon):
            # E[σ²(t+k)] = σ²_LR + (persistence)^k · (σ²(t) − σ²_LR)
            var_t = long_run_var + persistence * (var_t - long_run_var)
            sigmas[t] = np.sqrt(max(var_t, 1e-10))

        return sigmas

    # ──────────────────────────────────────────────────────────────
    # FIX 2: Sample-size-aware t-distribution fit
    # ──────────────────────────────────────────────────────────────
    def _fit_return_distribution(self):
        """
        Fit distribution to returns. Use t-student if enough samples,
        else fall back to normal (safer with few samples).
        Returns: (distribution_type, params)
        """
        n = len(self.returns)
        if n < self.MIN_SAMPLES_FOR_T_FIT:
            # With fewer samples, t-student fit is unreliable.
            # Use normal — less optimistic for probabilities.
            mu = np.mean(self.returns)
            sigma = max(0.001, np.std(self.returns))
            return ("normal", (mu, sigma))

        try:
            df, loc, scale = stats.t.fit(self.returns)
            # Constrain pathological values
            df = np.clip(df, 3.0, 30.0)
            scale = np.clip(scale, 0.0005, 0.15)
            return ("t", (df, loc, scale))
        except Exception:
            mu = np.mean(self.returns)
            sigma = max(0.001, np.std(self.returns))
            return ("normal", (mu, sigma))

    # ──────────────────────────────────────────────────────────────
    # MONTE CARLO PROJECTION
    # ──────────────────────────────────────────────────────────────
    def monte_carlo(self, horizon=None, n_sims=None):
        """
        Run Monte Carlo simulation for price at `horizon` trading days.

        Methodology:
          1. Fit distribution to log returns (t-student if enough data).
          2. Forecast per-day volatility with GARCH(1,1) if possible;
             fall back to EWMA (flat across horizon) otherwise.
          3. Simulate n_sims paths using sampled increments scaled by
             the forecasted per-day volatility.
          4. Compute percentiles, VaR, CVaR.

        Returns dict with all results, sanitized and rounded.
        """
        horizon = horizon or self.DEFAULT_HORIZON_DAYS
        n_sims = n_sims or self.N_SIMULATIONS_DEFAULT

        if len(self.returns) < 20:
            return self._fallback_projection(horizon)

        dist_type, params = self._fit_return_distribution()

        # ── Volatility forecast: GARCH if possible, else EWMA ──
        garch_sigmas = self._garch_forecast(horizon)
        if garch_sigmas is not None:
            vol_method = "garch(1,1)"
            sigma_path = garch_sigmas  # shape (horizon,)
            # Use mean forecast vol as baseline reference
            baseline_vol = float(np.mean(garch_sigmas))
        else:
            vol_method = "ewma"
            ewma_vol = self._ewma_volatility()
            sigma_path = np.full(horizon, ewma_vol)
            baseline_vol = ewma_vol

        # Extract distribution parameters for sampling
        if dist_type == "t":
            df, loc, scale = params
            # Sample standardized t (mean 0, unit variance)
            # Raw t with df has variance df/(df-2); scale down accordingly.
            raw_t = stats.t.rvs(df, size=(n_sims, horizon))
            if df > 2:
                # Make unit variance
                standard_rvs = raw_t / np.sqrt(df / (df - 2))
            else:
                # Degrees of freedom ≤ 2: undefined variance.
                # Fallback to normal sampling to keep things sane.
                standard_rvs = np.random.normal(0, 1, size=(n_sims, horizon))
            mean_return = loc
        else:
            loc, sigma = params
            standard_rvs = np.random.normal(0, 1, size=(n_sims, horizon))
            mean_return = loc

        last_price = self.close[-1]

        # Apply time-varying volatility: increments[i,t] = μ + σ(t) · ε
        increments = mean_return + sigma_path[np.newaxis, :] * standard_rvs

        # Cumulative log returns → terminal prices
        cum_returns = np.cumsum(increments, axis=1)
        terminal_prices = last_price * np.exp(cum_returns[:, -1])

        # Statistics
        expected = float(np.mean(terminal_prices))
        median = float(np.median(terminal_prices))
        std = float(np.std(terminal_prices))

        # Confidence intervals
        ci_68 = (float(np.percentile(terminal_prices, 16)),
                 float(np.percentile(terminal_prices, 84)))
        ci_95 = (float(np.percentile(terminal_prices, 2.5)),
                 float(np.percentile(terminal_prices, 97.5)))

        # Risk metrics (VaR, CVaR)
        var_95 = float(np.percentile(terminal_prices, 5))  # 5th percentile
        tail_mask = terminal_prices <= var_95
        cvar_95 = float(np.mean(terminal_prices[tail_mask])) if np.any(tail_mask) else var_95

        # Probabilities at key levels
        prob_up = float(np.mean(terminal_prices > last_price))
        prob_tp3 = float(np.mean(terminal_prices > last_price * 1.03))
        prob_tp5 = float(np.mean(terminal_prices > last_price * 1.05))
        prob_tp10 = float(np.mean(terminal_prices > last_price * 1.10))
        prob_drop_5 = float(np.mean(terminal_prices < last_price * 0.95))

        # Higher moments (diagnostic)
        skewness = float(stats.skew(terminal_prices))
        kurtosis = float(stats.kurtosis(terminal_prices))

        return {
            "method": dist_type,
            "vol_method": vol_method,  # "garch(1,1)" or "ewma"
            "horizon_days": horizon,
            "n_simulations": n_sims,
            "current_price": round(last_price, 4),
            "expected": round(expected, 4),
            "median": round(median, 4),
            "std": round(std, 4),
            "ci_68_low": round(ci_68[0], 4),
            "ci_68_high": round(ci_68[1], 4),
            "ci_95_low": round(ci_95[0], 4),
            "ci_95_high": round(ci_95[1], 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "prob_up": round(prob_up * 100, 2),
            "prob_tp3": round(prob_tp3 * 100, 2),
            "prob_tp5": round(prob_tp5 * 100, 2),
            "prob_tp10": round(prob_tp10 * 100, 2),
            "prob_drop_5": round(prob_drop_5 * 100, 2),
            "volatility_annualized": round(baseline_vol * np.sqrt(252) * 100, 2),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "reliable": len(self.returns) >= self.MIN_SAMPLES_FOR_T_FIT,
        }

    def _fallback_projection(self, horizon):
        """Simple fallback when too little data."""
        last = self.close[-1] if len(self.close) else 100.0
        return {
            "method": "fallback",
            "vol_method": "fallback",
            "horizon_days": horizon,
            "n_simulations": 0,
            "current_price": round(last, 4),
            "expected": round(last * 1.02, 4),
            "median": round(last * 1.02, 4),
            "std": round(last * 0.05, 4),
            "ci_68_low": round(last * 0.95, 4),
            "ci_68_high": round(last * 1.05, 4),
            "ci_95_low": round(last * 0.90, 4),
            "ci_95_high": round(last * 1.10, 4),
            "var_95": round(last * 0.90, 4),
            "cvar_95": round(last * 0.85, 4),
            "prob_up": 55.0,
            "prob_tp3": 40.0,
            "prob_tp5": 30.0,
            "prob_tp10": 15.0,
            "prob_drop_5": 30.0,
            "volatility_annualized": 25.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "reliable": False,
        }

    # ──────────────────────────────────────────────────────────────
    # REGIME DETECTION
    # ──────────────────────────────────────────────────────────────
    def detect_regime(self):
        """
        Classify current market regime based on volatility structure
        and trend strength (ADX).

        Regimes:
          - STRONG_TREND  : ADX > 25 + expanding volatility
          - TREND         : ADX > 25 + normal volatility
          - COMPRESSION   : ADX < 20 + contracting volatility (pre-breakout)
          - RANGE         : ADX < 20 + normal volatility
          - HIGH_VOL      : volatility > 1.5x historical average
        """
        def _get_last(key, default):
            v = self.ind.get(key)
            if v is None:
                return default
            if hasattr(v, "iloc"):
                last = v.iloc[-1]
                return float(last) if pd.notna(last) else default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        adx = _get_last("adx", 20.0)
        rsi = _get_last("rsi", 50.0)

        # ATR ratio (current vs 50-period average)
        atr_ratio = 1.0
        atr_series = self.ind.get("atr")
        if atr_series is not None and hasattr(atr_series, "rolling"):
            try:
                current_atr = float(atr_series.iloc[-1])
                mean_atr = float(atr_series.rolling(50).mean().iloc[-1])
                if mean_atr > 0:
                    atr_ratio = current_atr / mean_atr
            except Exception:
                pass

        # Relative volume (if provided)
        rel_vol = _get_last("rel_vol", 1.0)

        # Classification
        if atr_ratio > 1.5 or rel_vol > 2.0:
            regime = "HIGH_VOL"
            ensemble_mc_weight = 0.70  # Trust MC more in volatile times
            description = ("Volatilidad extrema — operar con tamaños reducidos o "
                           "esperar estabilización.")
        elif adx > 25 and atr_ratio > 1.2:
            regime = "STRONG_TREND"
            ensemble_mc_weight = 0.55
            description = "Tendencia fuerte establecida."
        elif adx > 25:
            regime = "TREND"
            ensemble_mc_weight = 0.55
            description = "Tendencia establecida."
        elif adx < 20 and atr_ratio < 0.8:
            regime = "COMPRESSION"
            ensemble_mc_weight = 0.65
            description = "Compresión de volatilidad — posible breakout próximo."
        else:
            regime = "RANGE"
            ensemble_mc_weight = 0.45
            description = "Mercado lateral."

        return {
            "regime": regime,
            "description": description,
            "adx": round(adx, 2),
            "rsi": round(rsi, 2),
            "atr_ratio": round(atr_ratio, 3),
            "rel_vol": round(rel_vol, 3),
            "ensemble_mc_weight": ensemble_mc_weight,
        }

    # ──────────────────────────────────────────────────────────────
    # ENSEMBLE PROJECTION
    # ──────────────────────────────────────────────────────────────
    def ensemble_projection(self, horizon=None):
        """
        Combine Monte Carlo + technical target (ATR-based) using
        regime-dependent weights.

        Returns a clean summary suitable for UI display.
        """
        horizon = horizon or self.DEFAULT_HORIZON_DAYS

        mc = self.monte_carlo(horizon=horizon)
        regime = self.detect_regime()

        current = float(self.close[-1])

        # Technical target using ATR
        atr_series = self.ind.get("atr")
        if atr_series is not None and hasattr(atr_series, "iloc"):
            try:
                atr_val = float(atr_series.iloc[-1])
            except Exception:
                atr_val = current * 0.02
        else:
            atr_val = current * 0.02

        # Regime-based multiplier for technical target
        multiplier = {
            "STRONG_TREND": 2.5,
            "TREND": 2.0,
            "COMPRESSION": 3.0,   # large expected moves
            "RANGE": 1.5,
            "HIGH_VOL": 1.8,
        }.get(regime["regime"], 2.0)

        tech_target = current + (atr_val * multiplier)

        # Ensemble weights from regime
        w_mc = regime["ensemble_mc_weight"]
        w_tech = 1.0 - w_mc

        ensemble_target = mc["expected"] * w_mc + tech_target * w_tech
        upside_pct = (ensemble_target / current - 1) * 100

        # Confidence labeling — honest, not aspirational
        prob_up = mc["prob_up"]
        if not mc["reliable"]:
            confidence = "Baja (pocos datos)"
        elif prob_up >= 70 and regime["regime"] in ("STRONG_TREND", "COMPRESSION"):
            confidence = "Media-Alta"
        elif prob_up >= 60:
            confidence = "Media"
        elif prob_up >= 50:
            confidence = "Media-Baja"
        else:
            confidence = "Baja"

        return {
            "current": round(current, 4),
            "target": round(ensemble_target, 4),
            "upside_pct": round(upside_pct, 2),
            "prob_up": prob_up,                   # already percent
            "prob_tp3": mc["prob_tp3"],
            "prob_tp5": mc["prob_tp5"],
            "prob_tp10": mc["prob_tp10"],
            "prob_drop_5": mc["prob_drop_5"],
            "confidence": confidence,
            "monte_carlo": mc,
            "regime": regime,
            "technical_target": round(tech_target, 4),
            "ensemble_weights": {
                "monte_carlo": round(w_mc, 2),
                "technical": round(w_tech, 2),
            },
            "disclaimer": "Las probabilidades son ESTIMACIONES del modelo. No hay garantía de que se cumplan.",
        }

    # ──────────────────────────────────────────────────────────────
    # MULTI-HORIZON
    # ──────────────────────────────────────────────────────────────
    def multi_horizon_projection(self):
        """Project at 5/10/21/63/126 trading days."""
        horizons = {
            "1_semana":   5,
            "2_semanas": 10,
            "1_mes":     21,
            "3_meses":   63,
            "6_meses":  126,
        }
        out = {}
        for name, days in horizons.items():
            mc = self.monte_carlo(horizon=days)
            out[name] = {
                "days": days,
                "expected": mc["expected"],
                "ci_68_low": mc["ci_68_low"],
                "ci_68_high": mc["ci_68_high"],
                "prob_up": mc["prob_up"],
                "upside_pct": round((mc["expected"] / mc["current_price"] - 1) * 100, 2),
                "var_95": mc["var_95"],
            }
        return out


# ─────────────────────────────────────────────────────────────────
# Self-test (run with: python stochastic_projector.py)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test data — 1 year of daily prices with drift
    np.random.seed(42)
    n = 252
    daily_returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(daily_returns))

    df = pd.DataFrame({
        "Close": prices,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    })

    # Fake indicators
    atr_series = pd.Series(prices * 0.02, index=df.index)
    indicators = {
        "adx": 28.5,
        "rsi": 55.0,
        "atr": atr_series,
        "rel_vol": 1.2,
    }

    sp = StochasticProjector(df, indicators, fundamentals={"fund_score": 70})

    print("=" * 60)
    print("STOCHASTIC PROJECTION ENGINE — Self Test")
    print("=" * 60)

    ens = sp.ensemble_projection()
    print(f"\nCurrent: ${ens['current']}")
    print(f"Target (1 month): ${ens['target']} ({ens['upside_pct']:+.2f}%)")
    print(f"Prob UP: {ens['prob_up']:.1f}%")
    print(f"Prob TP+3%: {ens['prob_tp3']:.1f}%")
    print(f"Prob drop -5%: {ens['prob_drop_5']:.1f}%")
    print(f"Regime: {ens['regime']['regime']} — {ens['regime']['description']}")
    print(f"Confidence: {ens['confidence']}")
    print(f"\nRisk metrics:")
    print(f"  VaR(95%):  ${ens['monte_carlo']['var_95']}")
    print(f"  CVaR(95%): ${ens['monte_carlo']['cvar_95']}")

    print("\nMulti-horizon:")
    mh = sp.multi_horizon_projection()
    for name, p in mh.items():
        print(f"  {name:10s}  exp=${p['expected']:.2f}  "
              f"upside={p['upside_pct']:+.1f}%  "
              f"prob_up={p['prob_up']:.0f}%")

    print(f"\n{ens['disclaimer']}")
