"""Protective Put Option Strategy Module.

Role:
    Models the Protective Put (Portfolio Insurance) strategy, combining a long underlying
    equity or index position with a long out-of-the-money or at-the-money put option.
    Widely utilized by institutional pension funds and equity portfolios in Taiwan to
    establish a firm floor against catastrophic market selloffs while retaining
    unlimited upside participation in the underlying asset.

Dependencies:
    - typing: Optional, Union, List
    - numpy: Array manipulations for payoff evaluation
    - config: Centralized system configuration
    - models.bs_engine: OptionType
    - .base: BaseOptionStrategy, StrategyLeg, PositionType, PayoffResult

Core Classes to be Implemented:
    - ProtectivePutStrategy: Concrete implementation of BaseOptionStrategy tailored
      for downside risk immunization, floor protection, and hedging drag analysis.

Core Functions / Methods to be Implemented:
    - build_legs: Construct the Long Stock/Index leg and Long Put leg.
    - calculate_max_loss: Determine strictly capped maximum loss (Spot - Strike + Put Premium).
    - get_breakeven_points: Spot price plus paid put premium.
    - calculate_insurance_cost_pct: Annualized cost of protection relative to portfolio value.
"""

from typing import List, Optional
import numpy as np

from config import DEFAULT_CONFIG, SystemConfig
from models.bs_engine import OptionType
from .base import BaseOptionStrategy, PayoffResult, PositionType, StrategyLeg


class ProtectivePutStrategy(BaseOptionStrategy):
    """Protective Put (Portfolio Insurance) Strategy Implementation."""

    def __init__(
        self,
        underlying_symbol: str,
        spot_price: float,
        put_strike: float,
        put_premium: float,
        expiry_years: float,
        quantity: float = 1.0,
        multiplier: float = 1.0,
        config: Optional[SystemConfig] = None,
    ) -> None:
        """Initialize ProtectivePutStrategy with market and trade parameters.

        Args:
            underlying_symbol: Ticker symbol (e.g., '^TWII', 'TXO', '2330.TW').
            spot_price: Entry spot price of underlying asset.
            put_strike: Strike price of protective long put.
            put_premium: Market premium paid per unit of put option.
            expiry_years: Time to expiration in years.
            quantity: Number of underlying units/contracts.
            multiplier: Contract multiplier (e.g., 50 for TXO).
            config: Optional system configuration instance.
        """
        super().__init__(
            name="Protective Put",
            underlying_symbol=underlying_symbol,
            config=config,
        )
        self.spot_price = spot_price
        self.put_strike = put_strike
        self.put_premium = put_premium
        self.expiry_years = expiry_years
        self.quantity = quantity
        self.multiplier = multiplier

        self.build_legs()

    def build_legs(self, **kwargs) -> None:
        """Construct the long underlying position and long put option leg."""
        # Leg 1: Long Underlying Asset
        self.add_leg(
            StrategyLeg(
                instrument_type="underlying",
                position_type=PositionType.LONG,
                quantity=self.quantity,
                entry_price=self.spot_price,
                multiplier=self.multiplier,
            )
        )
        # Leg 2: Long Put Option
        self.add_leg(
            StrategyLeg(
                instrument_type="option",
                position_type=PositionType.LONG,
                quantity=self.quantity,
                entry_price=self.put_premium,
                strike=self.put_strike,
                option_type=OptionType.PUT,
                expiry_years=self.expiry_years,
                multiplier=self.multiplier,
            )
        )

    def calculate_max_loss(self) -> float:
        """Compute strictly capped maximum financial loss."""
        raise NotImplementedError("calculate_max_loss will be implemented.")

    def calculate_insurance_cost_pct(self) -> float:
        """Calculate annualized percentage drag of option premium over portfolio value."""
        raise NotImplementedError("calculate_insurance_cost_pct will be implemented.")
