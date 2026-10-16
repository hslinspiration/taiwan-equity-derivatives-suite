"""Collar (Corridor Hedge) Option Strategy Module.

Role:
    Models the Collar strategy, which combines a long underlying equity/index position
    with a protective long out-of-the-money put and an out-of-the-money short call.
    The call premium financed either partially or entirely offsets the cost of the put,
    establishing an affordable (or zero-cost) bounded risk-return corridor.
    Extensively used in Taiwan equity asset management to guard against systemic drawdown
    without paying excessive cash premiums.

Dependencies:
    - typing: Optional, Union, List
    - numpy: Array operations for payoff grids
    - config: Centralized system configuration
    - models.bs_engine: OptionType
    - .base: BaseOptionStrategy, StrategyLeg, PositionType, PayoffResult

Core Classes to be Implemented:
    - CollarStrategy: Concrete implementation of BaseOptionStrategy tailored
      for bounded corridor payoffs, zero-cost strike optimization, and net debit/credit balance.

Core Functions / Methods to be Implemented:
    - build_legs: Construct Long Underlying, Long Put, and Short Call legs.
    - is_zero_cost: Check if net upfront premium paid/received is within near-zero tolerance.
    - calculate_corridor_bounds: Return lower floor (put strike) and upper ceiling (call strike).
    - solve_zero_cost_call_strike: Solve for call strike that exactly funds the put premium.
"""

from typing import List, Optional, Tuple
import numpy as np

from config import DEFAULT_CONFIG, SystemConfig
from models.bs_engine import OptionType
from .base import BaseOptionStrategy, PayoffResult, PositionType, StrategyLeg


class CollarStrategy(BaseOptionStrategy):
    """Collar (Corridor Hedge) Strategy Implementation."""

    def __init__(
        self,
        underlying_symbol: str,
        spot_price: float,
        put_strike: float,
        put_premium: float,
        call_strike: float,
        call_premium: float,
        expiry_years: float,
        quantity: float = 1.0,
        multiplier: float = 1.0,
        config: Optional[SystemConfig] = None,
    ) -> None:
        """Initialize CollarStrategy with underlying and two option legs.

        Args:
            underlying_symbol: Ticker identifier (e.g., 'TXO', '2330.TW').
            spot_price: Current or acquisition price of underlying asset.
            put_strike: Strike price of long protective put (floor).
            put_premium: Premium paid per unit of put option.
            call_strike: Strike price of short call option (ceiling).
            call_premium: Premium received per unit of call option.
            expiry_years: Time to expiration in years.
            quantity: Number of underlying units/contracts.
            multiplier: Multiplier factor for the derivative contracts.
            config: Optional system configuration instance.
        """
        super().__init__(
            name="Collar Strategy",
            underlying_symbol=underlying_symbol,
            config=config,
        )
        self.spot_price = spot_price
        self.put_strike = put_strike
        self.put_premium = put_premium
        self.call_strike = call_strike
        self.call_premium = call_premium
        self.expiry_years = expiry_years
        self.quantity = quantity
        self.multiplier = multiplier

        self.build_legs()

    def build_legs(self, **kwargs) -> None:
        """Construct Long Underlying, Long Put, and Short Call legs."""
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
        # Leg 2: Long Put Option (Floor)
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
        # Leg 3: Short Call Option (Ceiling)
        self.add_leg(
            StrategyLeg(
                instrument_type="option",
                position_type=PositionType.SHORT,
                quantity=self.quantity,
                entry_price=self.call_premium,
                strike=self.call_strike,
                option_type=OptionType.CALL,
                expiry_years=self.expiry_years,
                multiplier=self.multiplier,
            )
        )

    def is_zero_cost(self, tolerance: float = 1e-2) -> bool:
        """Check if the strategy is zero-cost (put premium ≈ call premium)."""
        return abs(self.put_premium - self.call_premium) <= tolerance

    def get_corridor_bounds(self) -> Tuple[float, float]:
        """Return the lower floor strike and upper ceiling strike."""
        return (self.put_strike, self.call_strike)
