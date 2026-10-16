"""Vectorized Black-Scholes-Merton (BSM) Analytical Option Pricing Engine.

Role:
    Implements closed-form, NumPy-vectorized valuation formulas for European-style
    equity and index options under the Black-Scholes-Merton framework extended with
    a continuous dividend yield (Merton, 1973). Tailored for Taiwan equity derivatives
    (e.g., TSMC 2330.TW stock options and TAIFEX TXO index options).

Mathematical Formulation:
    Given underlying spot S, strike K, time to maturity T (in years), continuously
    compounded risk-free rate r, continuous dividend yield q, and annualized
    volatility sigma:
        d1 = [ln(S / K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        Call = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
        Put  = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)

    Put-Call Parity:
        Call - Put = S * exp(-q * T) - K * exp(-r * T)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import norm

import config

ArrayLikeFloat = Union[float, np.ndarray]


class OptionType(str, Enum):
    """Option contract exercise right enumeration."""

    CALL = "call"
    PUT = "put"


def _is_vector_input(*args: object) -> bool:
    """Return True if any input argument is a non-scalar NumPy ndarray."""
    return any(isinstance(arg, np.ndarray) and arg.ndim > 0 for arg in args)


class BSEngine:
    """Vectorized Black-Scholes-Merton analytical pricing engine for European options.

    All pricing and intermediate moneyness methods support both scalar floats and
    NumPy arrays (`np.ndarray`) for spot `S` and strike `K` via broadcasting.
    """

    def __init__(self, system_config: Optional[config.SystemConfig] = None) -> None:
        """Initialize BSEngine with optional system configuration.

        Args:
            system_config: Optional SystemConfig instance.
        """
        self.config: config.SystemConfig = system_config or config.DEFAULT_CONFIG

    @staticmethod
    def d1(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute normalized log-moneyness parameter d1.

        Formula:
            d1 = (ln(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))

        Args:
            S: Underlying spot price(s) (scalar float or np.ndarray).
            K: Option strike price(s) (scalar float or np.ndarray).
            T: Time to expiration in years (scalar float or np.ndarray).
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: Computed d1 value(s).
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.log(s_arr / k_arr) + (r - q + 0.5 * np.square(sigma_arr)) * t_arr
            denominator = sigma_arr * np.sqrt(t_arr)
            result = np.where((t_arr > 0.0) & (sigma_arr > 0.0), numerator / denominator, 0.0)

        if is_vec:
            return np.asarray(result, dtype=float)
        return float(result)

    @staticmethod
    def d2(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute normalized log-moneyness parameter d2.

        Formula:
            d2 = d1 - sigma * sqrt(T)

        Args:
            S: Underlying spot price(s) (scalar float or np.ndarray).
            K: Option strike price(s) (scalar float or np.ndarray).
            T: Time to expiration in years (scalar float or np.ndarray).
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: Computed d2 value(s).
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        d1_val = np.asarray(BSEngine.d1(S, K, T, r, q, sigma), dtype=float)

        with np.errstate(invalid="ignore"):
            result = np.where(
                (t_arr > 0.0) & (sigma_arr > 0.0),
                d1_val - sigma_arr * np.sqrt(t_arr),
                0.0,
            )

        if is_vec:
            return np.asarray(result, dtype=float)
        return float(result)

    @staticmethod
    def call_price(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute theoretical European Call option fair value.

        Formula:
            Call = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

        Boundary & Edge Case Defense:
            - When T <= 0: Returns intrinsic payoff max(S - K, 0.0) directly.
            - When sigma <= 0 (and T > 0): Returns discounted intrinsic payoff
              max(S * exp(-q * T) - K * exp(-r * T), 0.0).

        Args:
            S: Underlying spot price(s) (scalar float or np.ndarray).
            K: Option strike price(s) (scalar float or np.ndarray).
            T: Time to expiration in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: European Call option price(s).
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        # Fast-path when T <= 0 across all elements
        if np.all(t_arr <= 0.0):
            intrinsic = np.maximum(s_arr - k_arr, 0.0)
            return np.asarray(intrinsic, dtype=float) if is_vec else float(intrinsic)

        # Fast-path when sigma <= 0 across all elements (for T > 0)
        if np.all(sigma_arr <= 0.0):
            discounted_intrinsic = np.maximum(
                s_arr * np.exp(-q * t_arr) - k_arr * np.exp(-r * t_arr), 0.0
            )
            return (
                np.asarray(discounted_intrinsic, dtype=float)
                if is_vec
                else float(discounted_intrinsic)
            )

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        d2_val = np.asarray(BSEngine.d2(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)

        bsm_call = s_arr * np.exp(-q * t_arr) * norm.cdf(d1_val) - k_arr * np.exp(
            -r * t_arr
        ) * norm.cdf(d2_val)

        intrinsic_expiry = np.maximum(s_arr - k_arr, 0.0)
        discounted_intrinsic = np.maximum(
            s_arr * np.exp(-q * t_arr) - k_arr * np.exp(-r * t_arr), 0.0
        )

        result = np.where(
            t_arr <= 0.0,
            intrinsic_expiry,
            np.where(sigma_arr <= 0.0, discounted_intrinsic, bsm_call),
        )

        if is_vec:
            return np.asarray(result, dtype=float)
        return float(result)

    @staticmethod
    def put_price(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute theoretical European Put option fair value.

        Formula:
            Put = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

        Boundary & Edge Case Defense:
            - When T <= 0: Returns intrinsic payoff max(K - S, 0.0) directly.
            - When sigma <= 0 (and T > 0): Returns discounted intrinsic payoff
              max(K * exp(-r * T) - S * exp(-q * T), 0.0).

        Args:
            S: Underlying spot price(s) (scalar float or np.ndarray).
            K: Option strike price(s) (scalar float or np.ndarray).
            T: Time to expiration in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: European Put option price(s).
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        # Fast-path when T <= 0 across all elements
        if np.all(t_arr <= 0.0):
            intrinsic = np.maximum(k_arr - s_arr, 0.0)
            return np.asarray(intrinsic, dtype=float) if is_vec else float(intrinsic)

        # Fast-path when sigma <= 0 across all elements (for T > 0)
        if np.all(sigma_arr <= 0.0):
            discounted_intrinsic = np.maximum(
                k_arr * np.exp(-r * t_arr) - s_arr * np.exp(-q * t_arr), 0.0
            )
            return (
                np.asarray(discounted_intrinsic, dtype=float)
                if is_vec
                else float(discounted_intrinsic)
            )

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        d2_val = np.asarray(BSEngine.d2(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)

        bsm_put = k_arr * np.exp(-r * t_arr) * norm.cdf(-d2_val) - s_arr * np.exp(
            -q * t_arr
        ) * norm.cdf(-d1_val)

        intrinsic_expiry = np.maximum(k_arr - s_arr, 0.0)
        discounted_intrinsic = np.maximum(
            k_arr * np.exp(-r * t_arr) - s_arr * np.exp(-q * t_arr), 0.0
        )

        result = np.where(
            t_arr <= 0.0,
            intrinsic_expiry,
            np.where(sigma_arr <= 0.0, discounted_intrinsic, bsm_put),
        )

        if is_vec:
            return np.asarray(result, dtype=float)
        return float(result)

    # =================================================================
    # Compatibility & Helper Interfaces
    # =================================================================
    @staticmethod
    def compute_d1_d2(
        spot: ArrayLikeFloat,
        strike: ArrayLikeFloat,
        time_to_maturity: ArrayLikeFloat,
        volatility: ArrayLikeFloat,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> Tuple[ArrayLikeFloat, ArrayLikeFloat]:
        """Compute both d1 and d2 parameters as a tuple."""
        d1_val = BSEngine.d1(
            spot, strike, time_to_maturity, risk_free_rate, dividend_yield, volatility
        )
        d2_val = BSEngine.d2(
            spot, strike, time_to_maturity, risk_free_rate, dividend_yield, volatility
        )
        return d1_val, d2_val

    def price(
        self,
        option_type: Union[OptionType, str],
        spot: ArrayLikeFloat,
        strike: ArrayLikeFloat,
        time_to_maturity: ArrayLikeFloat,
        volatility: ArrayLikeFloat,
        risk_free_rate: Optional[float] = None,
        dividend_yield: Optional[float] = None,
    ) -> ArrayLikeFloat:
        """Dispatch option pricing based on option_type ('call' or 'put')."""
        r = self.config.market.default_risk_free_rate if risk_free_rate is None else risk_free_rate
        q = self.config.market.default_dividend_yield if dividend_yield is None else dividend_yield
        opt = option_type.value if isinstance(option_type, OptionType) else str(option_type).lower()

        if opt == "call":
            return self.call_price(spot, strike, time_to_maturity, r, q, volatility)
        if opt == "put":
            return self.put_price(spot, strike, time_to_maturity, r, q, volatility)
        raise ValueError(f"Unsupported option_type '{option_type}'. Expected 'call' or 'put'.")


def black_scholes_price(
    option_type: Union[OptionType, str],
    spot: ArrayLikeFloat,
    strike: ArrayLikeFloat,
    time_to_maturity: ArrayLikeFloat,
    volatility: ArrayLikeFloat,
    risk_free_rate: float = config.FALLBACK_RISK_FREE_RATE,
    dividend_yield: float = 0.035,
) -> ArrayLikeFloat:
    """Standalone functional interface for Black-Scholes-Merton pricing."""
    engine = BSEngine()
    return engine.price(
        option_type=option_type,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
