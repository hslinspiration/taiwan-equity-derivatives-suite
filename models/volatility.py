"""Volatility Analytics, Historical Volatility Engine, and Vol Surface Modeling.

Role:
    Provides quantitative tools for historical realized volatility (Close-to-Close,
    Parkinson, Garman-Klass, Yang-Zhang), dynamic log-return rolling analytics,
    Black-Scholes-Merton implied volatility (IV) inversion via Brent's method,
    and 2D volatility surface representation across strike and tenor dimensions.

Financial Engineering & Taiwan Market Microstructure Context:
    - Adjusted Close (Adj Close) Requirement:
      Taiwan listed companies (notably TSMC 2330.TW, MediaTek 2454.TW, and constituent
      stocks of the TAIEX index) regularly pay substantial cash dividends (e.g., TSMC's
      quarterly dividend distributions). Under Taiwan Stock Exchange (TWSE) settlement rules,
      on the ex-dividend date (除息日), the reference opening price of the underlying equity
      is mechanically reduced by the exact cash dividend per share:
          S_ex = S_cum - D_cash
      If unadjusted close prices (Close) are used to compute log returns, this planned
      structural markdown produces an artificial, large negative return jump
      (e.g., -2% to -4% overnight) on every ex-dividend date. This spurious drop injects
      artificial variance into the sample variance estimator, drastically inflating
      historical volatility (HV) and distorting option pricing and Greek risk exposures.
      Using 'Adj Close' eliminates these artificial corporate action discontinuities through
      backward multiplicative adjustments, preserving true economic returns.
    - Annualization Convention:
      TWSE and TAIFEX operate on an annualized convention of 252 trading days per year
      (TRADING_DAYS_PER_YEAR = 252). Annualized volatility scales with the square root of time:
          sigma_annual = s * sqrt(252)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

import config


# =====================================================================
# Historical Volatility Engine
# =====================================================================
class HistoricalVolatility:
    """Historical and realized volatility engine for equity price time series.

    Computes continuously compounded log returns, sample variance with Bessel's
    correction (ddof=1), close-to-close historical volatility, and rolling volatility
    series over customizable observation windows.
    """

    def __init__(
        self,
        price_series: pd.Series,
        trading_days: int = config.TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize HistoricalVolatility with a clean historical price series.

        Args:
            price_series: pd.Series of asset prices (preferably Adjusted Close).
            trading_days: Annualized trading days convention (defaults to 252).

        Raises:
            TypeError: If price_series is not a pandas Series.
            ValueError: If price_series contains fewer than 30 valid observations,
                or contains non-positive price values.
        """
        if not isinstance(price_series, pd.Series):
            raise TypeError(
                f"price_series must be a pandas Series, received {type(price_series).__name__}."
            )

        clean_series = price_series.dropna()

        if len(clean_series) < 30:
            raise ValueError(
                f"price_series must contain at least 30 observations for statistical validity, "
                f"but only {len(clean_series)} valid observations were provided."
            )

        if (clean_series <= 0.0).any():
            raise ValueError(
                "price_series must contain strictly positive values to compute logarithmic returns."
            )

        self.price_series: pd.Series = clean_series.astype(float)
        self.trading_days: int = int(trading_days)

    def calculate_log_returns(self) -> pd.Series:
        """Compute continuously compounded logarithmic returns.

        Formula:
            u_t = ln(S_t / S_{t-1})

        Drops the initial NaN produced by the first observation difference.

        Returns:
            pd.Series: Log returns indexed by date, with the initial NaN dropped.
        """
        log_rets = np.log(self.price_series / self.price_series.shift(1))
        return log_rets.dropna()

    def close_to_close(self, window: int = config.DEFAULT_HV_WINDOW) -> float:
        """Compute annualized close-to-close historical volatility over the last window bars.

        Slices the most recent `window` observations of continuously compounded log returns,
        computes the unbiased sample standard deviation with Bessel's correction (ddof=1),
        and scales by the square root of annualized trading days.

        Formula:
            s = sqrt( 1/(N-1) * sum_{t=1}^N (u_t - u_mean)^2 )
            HV = s * sqrt(trading_days)

        Args:
            window: Number of return observations to include in volatility lookback
                (defaults to config.DEFAULT_HV_WINDOW = 60).

        Returns:
            float: Annualized historical sample volatility as a decimal (e.g. 0.225 for 22.5%).

        Raises:
            ValueError: If window is less than 2 observations.
        """
        if window < 2:
            raise ValueError(f"Lookback window must be at least 2 observations, got {window}.")

        log_returns = self.calculate_log_returns()
        if len(log_returns) < 2:
            raise ValueError("Insufficient return observations to compute sample standard deviation.")

        sliced_returns = log_returns.iloc[-window:]
        if len(sliced_returns) < 2:
            raise ValueError("Sliced return observations must contain at least 2 observations.")

        sample_std: float = float(sliced_returns.std(ddof=1))
        annualized_vol: float = sample_std * np.sqrt(self.trading_days)
        return float(annualized_vol)

    def rolling_hv(
        self,
        window: int = config.DEFAULT_HV_WINDOW,
        dropna: bool = True,
    ) -> pd.Series:
        """Compute rolling annualized historical volatility series.

        Evaluates sample volatility across a sliding window of `window` observations,
        scaling by sqrt(trading_days). Designed for volatility clustering diagnostics,
        regime detection, and graphical reporting.

        Args:
            window: Rolling window length in trading bars (defaults to config.DEFAULT_HV_WINDOW = 60).
            dropna: If True (default), drops the initial window - 1 NaN values so
                every entry in the returned Series is a valid float.

        Returns:
            pd.Series: Annualized rolling historical volatility series.

        Raises:
            ValueError: If window is less than 2 observations.
        """
        if window < 2:
            raise ValueError(f"Rolling window must be at least 2 observations, got {window}.")

        log_returns = self.calculate_log_returns()
        rolling_std = log_returns.rolling(window=window).std(ddof=1)
        annualized_rolling = rolling_std * np.sqrt(self.trading_days)

        if dropna:
            return annualized_rolling.dropna()
        return annualized_rolling


# =====================================================================
# Multi-Estimator Realized Volatility Calculator
# =====================================================================
class RealizedVolatilityEstimator:
    """Historical and realized volatility estimation algorithms across bar types."""

    @staticmethod
    def close_to_close(
        close_prices: pd.Series,
        annualization_factor: int = config.TRADING_DAYS_PER_YEAR,
        window: Optional[int] = None,
    ) -> float:
        """Compute standard close-to-close sample volatility."""
        clean = close_prices.dropna().astype(float)
        log_rets = np.log(clean / clean.shift(1)).dropna()
        if window is not None and window < len(log_rets):
            log_rets = log_rets.iloc[-window:]
        if len(log_rets) < 2:
            raise ValueError("Insufficient observations for close-to-close volatility.")
        return float(log_rets.std(ddof=1) * np.sqrt(annualization_factor))

    @staticmethod
    def parkinson(
        high_prices: pd.Series,
        low_prices: pd.Series,
        annualization_factor: int = config.TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Compute Parkinson (1980) extreme-value volatility based on High-Low range.

        Formula:
            sigma_P = sqrt( annualization / (4 * ln(2) * N) * sum( (ln(H_t / L_t))^2 ) )
        """
        h = high_prices.values.astype(float)
        l = low_prices.values.astype(float)
        if len(h) != len(l) or len(h) < 2:
            raise ValueError("High and Low price series must be of equal length >= 2.")

        hl_ratio = np.log(h / l)
        variance = np.sum(hl_ratio**2) / (4.0 * np.log(2.0) * len(h))
        return float(np.sqrt(variance * annualization_factor))

    @staticmethod
    def garman_klass(
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        annualization_factor: int = config.TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Compute Garman-Klass (1980) volatility utilizing full OHLC bars.

        Formula:
            sigma_GK = sqrt( (annualization / N) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2 ) )
        """
        o = open_prices.values.astype(float)
        h = high_prices.values.astype(float)
        l = low_prices.values.astype(float)
        c = close_prices.values.astype(float)
        n = len(c)
        if min(len(o), len(h), len(l), n) < 2:
            raise ValueError("All OHLC series must contain at least 2 observations.")

        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        variance = np.sum(0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)) / n
        return float(np.sqrt(max(0.0, variance) * annualization_factor))

    @staticmethod
    def yang_zhang(
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        annualization_factor: int = config.TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Compute Yang-Zhang (2000) drift-independent volatility handling overnight jumps."""
        o = open_prices.values.astype(float)
        h = high_prices.values.astype(float)
        l = low_prices.values.astype(float)
        c = close_prices.values.astype(float)
        n = len(c)
        if n < 3:
            raise ValueError("Yang-Zhang estimator requires at least 3 daily bars.")

        log_oc = np.log(o[1:] / c[:-1])  # Overnight returns
        log_co = np.log(c[1:] / o[1:])  # Open-to-close returns
        log_ho = np.log(h[1:] / o[1:])
        log_lo = np.log(l[1:] / o[1:])
        log_hc = np.log(h[1:] / c[1:])
        log_lc = np.log(l[1:] / c[1:])

        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var_overnight = np.var(log_oc, ddof=1)
        var_open_to_close = np.var(log_co, ddof=1)
        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

        total_var = var_overnight + k * var_open_to_close + (1.0 - k) * rs_var
        return float(np.sqrt(max(0.0, total_var) * annualization_factor))


# =====================================================================
# Implied Volatility Solver (Brent's Method)
# =====================================================================
class ImpliedVolatilitySolver:
    """Robust numerical solver for inverting market prices into implied volatilities."""

    def __init__(self, system_config: Optional[config.SystemConfig] = None) -> None:
        """Initialize IV solver with numerical tolerance configurations."""
        self.config = system_config or config.DEFAULT_CONFIG
        self.tolerance: float = self.config.model.iv_solver_tolerance
        self.max_iter: int = self.config.model.iv_solver_max_iter

    @staticmethod
    def _bsm_price(
        option_type: str,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float,
    ) -> float:
        """Closed-form Black-Scholes-Merton option price with dividend yield."""
        if time_to_maturity <= 1e-6 or volatility <= 1e-6:
            # Intrinsic value at expiration
            if option_type.lower() == "call":
                return max(0.0, spot - strike)
            return max(0.0, strike - spot)

        d1 = (
            np.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        df_r = np.exp(-risk_free_rate * time_to_maturity)
        df_q = np.exp(-dividend_yield * time_to_maturity)

        if option_type.lower() == "call":
            return float(spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2))
        return float(strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1))

    def solve(
        self,
        option_type: str,
        market_price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        initial_guess: float = 0.20,
        method: Optional[str] = None,
    ) -> float:
        """Extract implied volatility using Brent's method root-finding.

        Args:
            option_type: 'call' or 'put'.
            market_price: Observed market price of the option contract.
            spot: Underlying spot price.
            strike: Strike price.
            time_to_maturity: Time to expiration in years.
            risk_free_rate: Annualized risk-free rate.
            dividend_yield: Annualized continuous dividend yield.
            initial_guess: Initial volatility estimate (unused in bounded Brent).
            method: Solver method identifier ('brent').

        Returns:
            float: Implied volatility as a positive annualized decimal.

        Raises:
            ValueError: If option price violates no-arbitrage bounds or solver fails.
        """
        if market_price <= 0.0:
            raise ValueError(f"Market price must be strictly positive, got {market_price}.")
        if spot <= 0.0 or strike <= 0.0 or time_to_maturity <= 0.0:
            raise ValueError("Spot, strike, and time_to_maturity must be strictly positive.")

        # Theoretical no-arbitrage boundary checks
        df_q = np.exp(-dividend_yield * time_to_maturity)
        df_r = np.exp(-risk_free_rate * time_to_maturity)
        opt_type = option_type.lower()

        if opt_type == "call":
            intrinsic = max(0.0, spot * df_q - strike * df_r)
            upper_bound = spot * df_q
        else:
            intrinsic = max(0.0, strike * df_r - spot * df_q)
            upper_bound = strike * df_r

        if market_price < intrinsic:
            # Below theoretical lower bound: return minimum vol
            return 1e-4
        if market_price > upper_bound:
            raise ValueError(
                f"Market price {market_price:.2f} exceeds theoretical upper bound {upper_bound:.2f}."
            )

        def objective(sigma: float) -> float:
            return (
                self._bsm_price(
                    opt_type,
                    spot,
                    strike,
                    time_to_maturity,
                    sigma,
                    risk_free_rate,
                    dividend_yield,
                )
                - market_price
            )

        try:
            low_vol = 1e-4
            high_vol = 5.0
            solved_iv = optimize.brentq(
                objective,
                low_vol,
                high_vol,
                xtol=self.tolerance,
                maxiter=self.max_iter,
            )
            return float(solved_iv)
        except Exception as exc:
            raise ValueError(
                f"Brent solver failed to find implied volatility for {option_type} "
                f"[S={spot}, K={strike}, T={time_to_maturity:.4f}, Price={market_price}]: {exc}"
            ) from exc


# =====================================================================
# Volatility Surface Engine
# =====================================================================
class VolatilitySurface:
    """Implied volatility surface representation across strikes and maturities."""

    def __init__(self, spot: float) -> None:
        """Initialize VolatilitySurface with spot reference."""
        self.spot: float = float(spot)
        self._fitted: bool = False
        self._strikes: Optional[np.ndarray] = None
        self._maturities: Optional[np.ndarray] = None
        self._implied_vols: Optional[np.ndarray] = None

    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
    ) -> None:
        """Fit empirical implied volatility observations."""
        self._strikes = np.asarray(strikes, dtype=float)
        self._maturities = np.asarray(maturities, dtype=float)
        self._implied_vols = np.asarray(implied_vols, dtype=float)
        self._fitted = True

    def get_vol(self, strike: float, maturity: float) -> float:
        """Query implied volatility at specified strike and maturity."""
        if not self._fitted or self._implied_vols is None:
            raise RuntimeError("VolatilitySurface must be fitted prior to querying.")

        # Nearest neighbor interpolation heuristic for production robustness
        dists = (self._strikes - strike) ** 2 + ((self._maturities - maturity) * self.spot) ** 2
        nearest_idx = np.argmin(dists)
        return float(self._implied_vols[nearest_idx])


# =====================================================================
# Functional Helpers
# =====================================================================
def calculate_implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float = config.FALLBACK_RISK_FREE_RATE,
    dividend_yield: float = 0.035,
) -> float:
    """Convenience helper to extract implied volatility from a single market quote."""
    solver = ImpliedVolatilitySolver()
    return solver.solve(
        option_type=option_type,
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def calculate_historical_volatility(
    price_series: pd.Series,
    window: int = config.DEFAULT_HV_WINDOW,
    trading_days: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Convenience functional helper to calculate close-to-close historical volatility."""
    engine = HistoricalVolatility(price_series=price_series, trading_days=trading_days)
    return engine.close_to_close(window=window)
