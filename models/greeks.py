"""Option Greeks and Risk Sensitivities Analytics Engine.

Role:
    Computes analytical first-order, second-order, and cross Greeks for European
    options under the Merton (1973) continuous dividend yield framework.
    Also provides numerical finite-difference bump-and-reprice estimators
    used to cross-validate analytical Greek derivatives against model errors.

Dependencies:
    - dataclasses: Structured container for Greek sensitivity metrics
    - typing: Static typing annotations (Dict, Optional, Union)
    - numpy: Vectorized array calculations and math functions
    - scipy.stats: Normal distribution functions (norm.pdf, norm.cdf)
    - config: Global configuration parameters
    - .bs_engine: Core Black-Scholes-Merton engine and OptionType enum

Core Classes to be Implemented:
    - GreeksResult: Immutable dataclass encapsulating all calculated Greeks
      (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm).
    - GreekCalculator: Engine offering both analytical formulas and numerical
      finite-difference calculations.

Core Functions to be Implemented:
    - delta: First derivative with respect to underlying spot price (dV/dS).
    - gamma: Second derivative with respect to underlying spot price (d2V/dS2).
    - vega: First derivative with respect to volatility (dV/dsigma).
    - theta: First derivative with respect to time decay (dV/dt).
    - rho: First derivative with respect to risk-free rate (dV/dr).
    - vanna: Cross derivative with respect to spot and volatility (d2V/dS dsigma).
    - volga: Second derivative with respect to volatility (d2V/dsigma2).
    - charm: Cross derivative with respect to spot and time decay (dDelta/dt).
    - calculate_all_greeks: Aggregate calculation returning a populated GreeksResult instance.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from config import DEFAULT_CONFIG, SystemConfig
from .bs_engine import OptionType


@dataclass(frozen=True)
class GreeksResult:
    """Container for option Greek sensitivities.

    Attributes:
        delta: Sensitivity to underlying price (dV/dS).
        gamma: Rate of change of delta per unit move in spot (d2V/dS2).
        vega: Sensitivity to 1% (or 100 bps) change in volatility (dV/dsigma).
        theta: Daily time decay sensitivity (dV/dt per calendar day).
        rho: Sensitivity to 1% change in risk-free interest rate (dV/dr).
        epsilon: Sensitivity to 1% change in continuous dividend yield (dV/dq).
        vanna: Cross-sensitivity of delta to volatility or vega to spot.
        volga: Sensitivity of vega to volatility (convexity of vega).
        charm: Rate of change of delta over time decay (delta bleed).
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    epsilon: float
    vanna: float
    volga: float
    charm: float


class GreekCalculator:
    """Quantitative calculator for option Greeks and sensitivity analytics."""

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        """Initialize GreekCalculator with system configuration.

        Args:
            config: Optional system configuration instance.
        """
        self.config = config or DEFAULT_CONFIG

    def delta(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Delta calculation."""
        raise NotImplementedError("Analytical delta will be implemented.")

    def gamma(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Gamma calculation."""
        raise NotImplementedError("Analytical gamma will be implemented.")

    def vega(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Vega calculation per 1% change in implied volatility."""
        raise NotImplementedError("Analytical vega will be implemented.")

    def theta(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical 1-day Theta calculation."""
        raise NotImplementedError("Analytical theta will be implemented.")

    def rho(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Rho calculation per 1% change in risk-free rate."""
        raise NotImplementedError("Analytical rho will be implemented.")

    def vanna(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Vanna (dDelta/dSigma)."""
        raise NotImplementedError("Analytical vanna will be implemented.")

    def volga(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Volga / Vomma (dVega/dSigma)."""
        raise NotImplementedError("Analytical volga will be implemented.")

    def charm(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Analytical Charm (dDelta/dt)."""
        raise NotImplementedError("Analytical charm will be implemented.")

    def compute_all(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> GreeksResult:
        """Calculate complete suite of option Greeks and return structured result."""
        raise NotImplementedError("compute_all will be implemented.")

    def numerical_greeks(
        self,
        option_type: Union[OptionType, str],
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> GreeksResult:
        """Compute Greeks using central finite differences for cross-validation."""
        raise NotImplementedError("numerical_greeks will be implemented.")


def calculate_all_greeks(
    option_type: Union[OptionType, str],
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    risk_free_rate: float = 0.016,
    dividend_yield: float = 0.035,
) -> GreeksResult:
    """Helper function to evaluate all Greeks for given contract parameters."""
    calculator = GreekCalculator()
    return calculator.compute_all(
        option_type, spot, strike, time_to_maturity, volatility, risk_free_rate, dividend_yield
    )
