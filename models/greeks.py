"""Analytical Option Greeks and Risk Sensitivities Engine.

Role:
    Computes closed-form analytical first- and second-order Greeks (Delta, Gamma,
    Vega, Theta, Rho) for European options under the Merton (1973) continuous
    dividend yield extension of the Black-Scholes-Merton framework. All methods
    support both scalar floats and vectorized NumPy arrays (`np.ndarray`).

Conventions & Scaling:
    - Delta (dV/dS): First partial derivative with respect to spot price S.
    - Gamma (d^2V/dS^2): Second partial derivative with respect to spot price S.
    - Vega (dV/dsigma): Normalized to a 1% (0.01) absolute change in implied volatility.
    - Theta (dV/dt): Normalized to 1 calendar day time decay (Annual Theta / 365.0).
    - Rho (dV/dr): Normalized to a 1% (0.01) absolute change in the risk-free rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
from scipy.stats import norm

import config
from .bs_engine import ArrayLikeFloat, BSEngine, OptionType, _is_vector_input


@dataclass(frozen=True)
class GreeksResult:
    """Container for option Greek sensitivities.

    Attributes:
        delta: Sensitivity to underlying price (dV/dS).
        gamma: Rate of change of delta per unit move in spot (d^2V/dS^2).
        vega: Sensitivity to 1% (0.01) change in volatility.
        theta: Daily time decay sensitivity (annual Theta / 365.0).
        rho: Sensitivity to 1% (0.01) change in risk-free interest rate.
    """

    delta: ArrayLikeFloat
    gamma: ArrayLikeFloat
    vega: ArrayLikeFloat
    theta: ArrayLikeFloat
    rho: ArrayLikeFloat


def _normalize_option_type(option_type: Union[OptionType, str]) -> str:
    """Validate and normalize option_type to lowercase 'call' or 'put'."""
    opt = option_type.value if isinstance(option_type, OptionType) else str(option_type).strip().lower()
    if opt not in ("call", "put"):
        raise ValueError(f"Invalid option_type '{option_type}'. Expected 'call' or 'put'.")
    return opt


class GreeksCalculator:
    """Closed-form analytical calculator for Black-Scholes-Merton option Greeks.

    Supports both scalar `float` and `np.ndarray` inputs across all methods.
    """

    def __init__(self, system_config: Optional[config.SystemConfig] = None) -> None:
        """Initialize GreeksCalculator with optional system configuration."""
        self.config: config.SystemConfig = system_config or config.DEFAULT_CONFIG

    @staticmethod
    def delta(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
        option_type: Union[OptionType, str] = "call",
    ) -> ArrayLikeFloat:
        """Compute analytical BSM option Delta.

        Formulas:
            Call Delta = exp(-q * T) * norm.cdf(d1)
            Put Delta  = -exp(-q * T) * norm.cdf(-d1)

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.
            option_type: 'call' or 'put' (defaults to 'call').

        Returns:
            float | np.ndarray: Option Delta value(s).
        """
        opt = _normalize_option_type(option_type)
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        if np.all(t_arr <= 0.0):
            if opt == "call":
                exp_delta = np.where(s_arr > k_arr, 1.0, np.where(s_arr == k_arr, 0.5, 0.0))
            else:
                exp_delta = np.where(s_arr < k_arr, -1.0, np.where(s_arr == k_arr, -0.5, 0.0))
            return np.asarray(exp_delta, dtype=float) if is_vec else float(exp_delta)

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        df_q = np.exp(-q * t_arr)

        if opt == "call":
            bsm_delta = df_q * norm.cdf(d1_val)
            expiry_fallback = np.where(s_arr > k_arr, 1.0, 0.0)
        else:
            bsm_delta = -df_q * norm.cdf(-d1_val)
            expiry_fallback = np.where(s_arr < k_arr, -1.0, 0.0)

        result = np.where((t_arr <= 0.0) | (sigma_arr <= 0.0), expiry_fallback, bsm_delta)
        return np.asarray(result, dtype=float) if is_vec else float(result)

    @staticmethod
    def gamma(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute analytical BSM option Gamma (identical for Call and Put).

        Formula:
            Gamma = (exp(-q * T) * norm.pdf(d1)) / (S * sigma * sqrt(T))

        Boundary:
            When T <= 0 or sigma <= 0, returns 0.0.

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: Option Gamma value(s).
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        if np.all(t_arr <= 0.0) or np.all(sigma_arr <= 0.0):
            zeros = np.zeros_like(s_arr + k_arr + t_arr + sigma_arr, dtype=float)
            return np.asarray(zeros, dtype=float) if is_vec else 0.0

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.exp(-q * t_arr) * norm.pdf(d1_val)
            denominator = s_arr * sigma_arr * np.sqrt(t_arr)
            result = np.where((t_arr > 0.0) & (sigma_arr > 0.0), numerator / denominator, 0.0)

        return np.asarray(result, dtype=float) if is_vec else float(result)

    @staticmethod
    def vega(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
    ) -> ArrayLikeFloat:
        """Compute analytical BSM option Vega normalized to a 1% (0.01) volatility change.

        Formula:
            Vega = 0.01 * S * exp(-q * T) * sqrt(T) * norm.pdf(d1)

        Boundary:
            When T <= 0 or sigma <= 0, returns 0.0.

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float | np.ndarray: Option Vega value(s) per 1% volatility point.
        """
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        if np.all(t_arr <= 0.0) or np.all(sigma_arr <= 0.0):
            zeros = np.zeros_like(s_arr + k_arr + t_arr + sigma_arr, dtype=float)
            return np.asarray(zeros, dtype=float) if is_vec else 0.0

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        with np.errstate(invalid="ignore"):
            raw_vega = 0.01 * s_arr * np.exp(-q * t_arr) * np.sqrt(t_arr) * norm.pdf(d1_val)
            result = np.where((t_arr > 0.0) & (sigma_arr > 0.0), raw_vega, 0.0)

        return np.asarray(result, dtype=float) if is_vec else float(result)

    @staticmethod
    def theta(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
        option_type: Union[OptionType, str] = "call",
    ) -> ArrayLikeFloat:
        """Compute analytical BSM option Theta normalized to 1 calendar day decay (/ 365.0).

        Formulas:
            Term1      = - (S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))
            Call Theta = (Term1 - r * K * exp(-r * T) * norm.cdf(d2)
                          + q * S * exp(-q * T) * norm.cdf(d1)) / 365.0
            Put Theta  = (Term1 + r * K * exp(-r * T) * norm.cdf(-d2)
                          - q * S * exp(-q * T) * norm.cdf(-d1)) / 365.0

        Boundary:
            When T <= 0 or sigma <= 0, returns 0.0.

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.
            option_type: 'call' or 'put' (defaults to 'call').

        Returns:
            float | np.ndarray: Daily Option Theta value(s).
        """
        opt = _normalize_option_type(option_type)
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        if np.all(t_arr <= 0.0) or np.all(sigma_arr <= 0.0):
            zeros = np.zeros_like(s_arr + k_arr + t_arr + sigma_arr, dtype=float)
            return np.asarray(zeros, dtype=float) if is_vec else 0.0

        d1_val = np.asarray(BSEngine.d1(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        d2_val = np.asarray(BSEngine.d2(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)

        df_q = np.exp(-q * t_arr)
        df_r = np.exp(-r * t_arr)

        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = -(s_arr * df_q * norm.pdf(d1_val) * sigma_arr) / (2.0 * np.sqrt(t_arr))
            if opt == "call":
                annual_theta = (
                    term1
                    - r * k_arr * df_r * norm.cdf(d2_val)
                    + q * s_arr * df_q * norm.cdf(d1_val)
                )
            else:
                annual_theta = (
                    term1
                    + r * k_arr * df_r * norm.cdf(-d2_val)
                    - q * s_arr * df_q * norm.cdf(-d1_val)
                )
            daily_theta = annual_theta / 365.0
            result = np.where((t_arr > 0.0) & (sigma_arr > 0.0), daily_theta, 0.0)

        return np.asarray(result, dtype=float) if is_vec else float(result)

    @staticmethod
    def rho(
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
        option_type: Union[OptionType, str] = "call",
    ) -> ArrayLikeFloat:
        """Compute analytical BSM option Rho normalized to a 1% (0.01) interest rate change.

        Formulas:
            Call Rho = 0.01 * K * T * exp(-r * T) * norm.cdf(d2)
            Put Rho  = -0.01 * K * T * exp(-r * T) * norm.cdf(-d2)

        Boundary:
            When T <= 0 or sigma <= 0, returns 0.0.

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.
            option_type: 'call' or 'put' (defaults to 'call').

        Returns:
            float | np.ndarray: Option Rho value(s) per 1% rate move.
        """
        opt = _normalize_option_type(option_type)
        is_vec = _is_vector_input(S, K, T, r, q, sigma)
        s_arr = np.asarray(S, dtype=float)
        k_arr = np.asarray(K, dtype=float)
        t_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)

        if np.all(t_arr <= 0.0) or np.all(sigma_arr <= 0.0):
            zeros = np.zeros_like(s_arr + k_arr + t_arr + sigma_arr, dtype=float)
            return np.asarray(zeros, dtype=float) if is_vec else 0.0

        d2_val = np.asarray(BSEngine.d2(s_arr, k_arr, t_arr, r, q, sigma_arr), dtype=float)
        df_r = np.exp(-r * t_arr)

        if opt == "call":
            raw_rho = 0.01 * k_arr * t_arr * df_r * norm.cdf(d2_val)
        else:
            raw_rho = -0.01 * k_arr * t_arr * df_r * norm.cdf(-d2_val)

        result = np.where((t_arr > 0.0) & (sigma_arr > 0.0), raw_rho, 0.0)
        return np.asarray(result, dtype=float) if is_vec else float(result)

    @classmethod
    def all_greeks(
        cls,
        S: ArrayLikeFloat,
        K: ArrayLikeFloat,
        T: ArrayLikeFloat,
        r: float,
        q: float,
        sigma: ArrayLikeFloat,
        option_type: Union[OptionType, str] = "call",
    ) -> Dict[str, ArrayLikeFloat]:
        """Compute all 5 primary BSM Greeks and return as a dictionary.

        Args:
            S: Underlying spot price(s).
            K: Option strike price(s).
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.
            option_type: 'call' or 'put' (defaults to 'call').

        Returns:
            dict: Dictionary with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
        """
        return {
            "delta": cls.delta(S, K, T, r, q, sigma, option_type=option_type),
            "gamma": cls.gamma(S, K, T, r, q, sigma),
            "vega": cls.vega(S, K, T, r, q, sigma),
            "theta": cls.theta(S, K, T, r, q, sigma, option_type=option_type),
            "rho": cls.rho(S, K, T, r, q, sigma, option_type=option_type),
        }


# Backward-compatible class alias
GreekCalculator = GreeksCalculator


def calculate_all_greeks(
    option_type: Union[OptionType, str],
    spot: ArrayLikeFloat,
    strike: ArrayLikeFloat,
    time_to_maturity: ArrayLikeFloat,
    volatility: ArrayLikeFloat,
    risk_free_rate: float = config.FALLBACK_RISK_FREE_RATE,
    dividend_yield: float = 0.035,
) -> GreeksResult:
    """Evaluate all 5 primary Greeks and return a structured GreeksResult container."""
    greeks_dict = GreeksCalculator.all_greeks(
        S=spot,
        K=strike,
        T=time_to_maturity,
        r=risk_free_rate,
        q=dividend_yield,
        sigma=volatility,
        option_type=option_type,
    )
    return GreeksResult(
        delta=greeks_dict["delta"],
        gamma=greeks_dict["gamma"],
        vega=greeks_dict["vega"],
        theta=greeks_dict["theta"],
        rho=greeks_dict["rho"],
    )
