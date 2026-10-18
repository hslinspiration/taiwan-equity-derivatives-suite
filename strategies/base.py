"""Foundational Strategy Abstractions and Portfolio Greek Superposition Engine.

Role:
    Defines the core building blocks (`OptionLeg` and `BaseStrategy`) for
    institutional equity derivatives overlay structures in the Taiwan market
    (1 TWSE standard board lot = 1,000 shares). Supports multi-leg expiration
    payoff vectorization and linear superposition of analytical Black-Scholes-Merton
    portfolio Greeks across underlying shares and option legs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from models.bs_engine import BSEngine
from models.greeks import GreeksCalculator


class PositionType(int, Enum):
    """Direction of an option or underlying leg."""

    LONG = 1
    SHORT = -1


@dataclass
class PayoffResult:
    """Container for vectorized strategy payoff and P&L across a spot grid."""

    spot_grid: np.ndarray
    payoff: np.ndarray
    net_pnl: Optional[np.ndarray] = None


@dataclass
class OptionLeg:
    """Represents a single option contract leg within a structured overlay.

    Attributes:
        option_type: 'call' or 'put' (case-insensitive).
        position: +1 for Long, -1 for Short.
        strike: Exercise strike price of the option.
        maturity: Time to expiration in years (e.g., 30 / 365.0).
        contracts: Number of option contracts / multiplier (default 1).
    """

    option_type: str
    position: int
    strike: float
    maturity: float
    contracts: int = 1

    def __post_init__(self) -> None:
        """Validate and normalize option leg parameters."""
        normalized_type = str(self.option_type).strip().lower()
        if normalized_type not in ("call", "put"):
            raise ValueError(
                f"Invalid option_type '{self.option_type}'. Must be 'call' or 'put'."
            )
        self.option_type = normalized_type

        if int(self.position) not in (1, -1):
            raise ValueError(
                f"Invalid position '{self.position}'. Must be +1 (Long) or -1 (Short)."
            )
        self.position = int(self.position)
        self.strike = float(self.strike)
        self.maturity = float(self.maturity)
        self.contracts = int(self.contracts)


class BaseStrategy(ABC):
    """Abstract base class for institutional equity + option overlay strategies.

    Models a core equity holding (`underlying_shares`, defaulting to 1 Taiwan
    board lot = 1,000 shares) combined with one or more European option legs.
    """

    def __init__(self, spot_price: float, underlying_shares: int = 1000) -> None:
        """Initialize strategy with spot price and underlying share quantity.

        Args:
            spot_price: Current underlying equity spot price in TWD.
            underlying_shares: Number of underlying shares held (1 TWSE board lot = 1,000 shares).
        """
        if spot_price <= 0.0:
            raise ValueError(f"spot_price must be strictly positive, got {spot_price}.")
        self.spot_price: float = float(spot_price)
        self.underlying_shares: int = int(underlying_shares)
        self.legs: List[OptionLeg] = []

    def add_leg(self, leg: OptionLeg) -> None:
        """Append an OptionLeg to the strategy portfolio.

        Args:
            leg: Configured OptionLeg instance.
        """
        if not isinstance(leg, OptionLeg):
            raise TypeError(f"Expected OptionLeg instance, got {type(leg).__name__}.")
        self.legs.append(leg)

    def calculate_payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        """Compute net terminal strategy payoff across an array of spot prices at expiration.

        Note:
            Computes terminal intrinsic value relative to initial spot entry for the
            underlying stock plus terminal option payoffs (excluding upfront option premiums):
                Stock Payoff = (S_T - S_0) * underlying_shares
                Call Leg     = position * max(S_T - K, 0) * contracts * underlying_shares
                Put Leg      = position * max(K - S_T, 0) * contracts * underlying_shares

        Args:
            spot_grid: 1D NumPy array of terminal spot prices S_T.

        Returns:
            np.ndarray: Total terminal payoff in TWD corresponding to `spot_grid`.
        """
        spots = np.asarray(spot_grid, dtype=float)
        total_payoff = (spots - self.spot_price) * float(self.underlying_shares)

        for leg in self.legs:
            scale = float(leg.position * leg.contracts * self.underlying_shares)
            if leg.option_type == "call":
                leg_payoff = scale * np.maximum(spots - leg.strike, 0.0)
            else:
                leg_payoff = scale * np.maximum(leg.strike - spots, 0.0)
            total_payoff = total_payoff + leg_payoff

        return np.asarray(total_payoff, dtype=float)

    def calculate_portfolio_greeks(
        self,
        bs_engine: BSEngine,
        r: float,
        q: float,
        sigma: float,
    ) -> Dict[str, float]:
        """Compute net portfolio Greeks via linear superposition across stock and option legs.

        Underlying stock sensitivities:
            Delta = +1.0 * underlying_shares
            Gamma = 0.0, Vega = 0.0, Theta = 0.0, Rho = 0.0

        Each option leg's analytical per-share Greeks are scaled by:
            leg.position * leg.contracts * self.underlying_shares

        Args:
            bs_engine: BSEngine instance.
            r: Annualized continuously compounded risk-free interest rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            dict[str, float]: Dictionary with keys 'net_delta', 'net_gamma',
                'net_vega', 'net_theta', and 'net_rho'.
        """
        _ = bs_engine  # Used for interface consistency alongside GreeksCalculator
        net_delta: float = 1.0 * float(self.underlying_shares)
        net_gamma: float = 0.0
        net_vega: float = 0.0
        net_theta: float = 0.0
        net_rho: float = 0.0

        calc = GreeksCalculator()

        for leg in self.legs:
            scale = float(leg.position * leg.contracts * self.underlying_shares)
            leg_greeks = calc.all_greeks(
                S=self.spot_price,
                K=leg.strike,
                T=leg.maturity,
                r=r,
                q=q,
                sigma=sigma,
                option_type=leg.option_type,
            )
            net_delta += scale * float(leg_greeks["delta"])
            net_gamma += scale * float(leg_greeks["gamma"])
            net_vega += scale * float(leg_greeks["vega"])
            net_theta += scale * float(leg_greeks["theta"])
            net_rho += scale * float(leg_greeks["rho"])

        return {
            "net_delta": float(net_delta),
            "net_gamma": float(net_gamma),
            "net_vega": float(net_vega),
            "net_theta": float(net_theta),
            "net_rho": float(net_rho),
        }

    @abstractmethod
    def evaluate(
        self,
        bs_engine: BSEngine,
        r: float,
        q: float,
        sigma: float,
    ) -> Dict[str, Any]:
        """Evaluate strategy pricing, institutional risk/return metrics, and portfolio Greeks."""


# Backward-compatible aliases
BaseOptionStrategy = BaseStrategy
StrategyLeg = OptionLeg
