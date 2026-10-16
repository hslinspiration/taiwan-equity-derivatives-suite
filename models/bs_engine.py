"""Black-Scholes-Merton (BSM) Analytical Option Pricing Engine.

Role:
    Implements closed-form valuation formulas for European-style equity and index
    options based on the Black-Scholes-Merton model extended with a continuous
    dividend yield (Merton, 1973). This extension is critical for Taiwan index
    options (TXO) and dividend-paying Taiwan equities (e.g., TSMC 2330.TW).

Dependencies:
    - enum: OptionType definition ('call', 'put')
    - typing: Static typing annotations (Union, Optional, Tuple)
    - numpy: Vectorized mathematical operations and arrays
    - scipy.stats: Cumulative standard normal distribution (norm.cdf) and density (norm.pdf)
    - config: Global configuration parameters

Core Classes to be Implemented:
    - OptionType: Enumeration defining CALL and PUT option contracts.
    - BSEngine: High-performance pricing engine capable of handling scalar and vectorized inputs.

Core Functions to be Implemented:
    - compute_d1_d2: Calculate intermediate normalized log-moneyness factors d1 and d2.
    - black_scholes_price: Analytical valuation formula returning theoretical option fair value.
    - verify_put_call_parity: Validate theoretical prices against European put-call parity relationship:
      C - P = S * exp(-q * T) - K * exp(-r * T).
"""

from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import norm

from config import DEFAULT_CONFIG, SystemConfig


class OptionType(str, Enum):
    """Option contract exercise type."""
    CALL = "call"
    PUT = "put"


class BSEngine:
    """Analytical pricing engine for European options with continuous dividend yield."""

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        """Initialize BSEngine with optional system configuration.

        Args:
            config: SystemConfig instance containing market and numerical defaults.
        """
        self.config = config or DEFAULT_CONFIG

    @staticmethod
    def compute_d1_d2(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_maturity: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Union[float, np.ndarray],
        dividend_yield: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute the d1 and d2 components of the BSM formula.

        Args:
            spot (S): Current price of the underlying asset.
            strike (K): Option strike price.
            time_to_maturity (T): Time to expiration in years.
            volatility (sigma): Annualized volatility of the underlying asset.
            risk_free_rate (r): Annualized continuously compounded risk-free rate.
            dividend_yield (q): Annualized continuously compounded dividend yield.

        Returns:
            Tuple[d1, d2]: Standard normal variates.
        """
        # Placeholder: Closed-form implementation to be written
        raise NotImplementedError("compute_d1_d2 will be implemented.")

    def price(
        self,
        option_type: Union[OptionType, str],
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_maturity: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Optional[float] = None,
        dividend_yield: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Calculate the theoretical Black-Scholes-Merton option price.

        Args:
            option_type: OptionType.CALL or OptionType.PUT.
            spot (S): Current underlying asset price.
            strike (K): Option strike price.
            time_to_maturity (T): Time to maturity in years (ACT/365 or ACT/252).
            volatility (sigma): Implied or forecasted volatility.
            risk_free_rate (r): Risk-free interest rate (defaults to config value).
            dividend_yield (q): Continuous dividend yield (defaults to config value).

        Returns:
            Union[float, np.ndarray]: Theoretical fair value of the option.
        """
        # Placeholder: Closed-form pricing logic to be implemented
        raise NotImplementedError("BSEngine.price will be implemented.")

    def check_parity(
        self,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> Tuple[bool, float]:
        """Verify whether call and put prices satisfy European put-call parity.

        Parity relationship:
            C - P = S * exp(-q * T) - K * exp(-r * T)

        Returns:
            Tuple[bool, float]: (is_parity_satisfied, parity_discrepancy)
        """
        # Placeholder: Put-call parity verification to be implemented
        raise NotImplementedError("check_parity will be implemented.")


def black_scholes_price(
    option_type: Union[OptionType, str],
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    risk_free_rate: float = 0.016,
    dividend_yield: float = 0.035,
) -> float:
    """Convenience function for calculating Black-Scholes price.

    Args:
        option_type: 'call' or 'put'.
        spot: Underlying asset price.
        strike: Strike price.
        time_to_maturity: Time to expiration in years.
        volatility: Annualized volatility.
        risk_free_rate: Annualized risk-free rate.
        dividend_yield: Annualized dividend yield.

    Returns:
        float: Option price.
    """
    engine = BSEngine()
    return float(
        engine.price(
            option_type, spot, strike, time_to_maturity, volatility, risk_free_rate, dividend_yield
        )
    )
