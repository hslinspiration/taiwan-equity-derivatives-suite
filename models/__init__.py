"""Financial Engineering Pricing Models and Derivatives Analytics.

Role:
    Houses core quantitative engines for option valuation, Greek risk sensitivities,
    implied volatility root-finding, realized volatility estimation, and volatility
    surface representation tailored to Taiwan index options (TXO) and equity options.

Dependencies:
    - .bs_engine: Black-Scholes-Merton model with continuous dividend yield (Merton 1973).
    - .greeks: Analytical and numerical first, second, and third-order Greeks.
    - .volatility: Implied volatility solvers (Brent/Newton-Raphson), historical
      realized vol estimators (Garman-Klass, Parkinson, Yang-Zhang), and vol surface interpolators.

Core Classes & Functions Exposed:
    - BSEngine: Analytical Black-Scholes-Merton option pricer.
    - OptionType: Enum representing CALL or PUT.
    - GreekCalculator: High-order Greek sensitivity computation engine.
    - GreeksResult: Container for all first, second, and third-order Greeks.
    - ImpliedVolatilitySolver: High-performance IV solver for market quotes.
    - RealizedVolatilityEstimator: Multi-estimator historical volatility calculator.
    - VolatilitySurface: Strike/expiry implied volatility interpolator.
"""

from .bs_engine import BSEngine, OptionType, black_scholes_price
from .greeks import GreekCalculator, GreeksResult, calculate_all_greeks
from .volatility import (
    ImpliedVolatilitySolver,
    RealizedVolatilityEstimator,
    VolatilitySurface,
    calculate_implied_volatility,
)

__all__ = [
    "BSEngine",
    "OptionType",
    "black_scholes_price",
    "GreekCalculator",
    "GreeksResult",
    "calculate_all_greeks",
    "ImpliedVolatilitySolver",
    "RealizedVolatilityEstimator",
    "VolatilitySurface",
    "calculate_implied_volatility",
]
