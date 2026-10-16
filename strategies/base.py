"""Abstract Base Class for Multi-Leg Option Strategies.

Role:
    Defines common abstractions, interfaces, data containers, and analytical
    methods for multi-leg equity and index option overlay strategies.
    Standardizes terminal payoff computations, mark-to-market PnL tracking,
    net upfront debit/credit evaluation, aggregate Greek risk aggregation,
    and breakeven point solvers.

Dependencies:
    - abc: Abstract Base Class primitives (ABC, abstractmethod)
    - dataclasses: Structured data representations
    - enum: Position direction enumeration (LONG / SHORT)
    - typing: Static typing annotations (Dict, List, Optional, Tuple, Union)
    - numpy: Vectorized evaluation across spot price domains
    - config: Centralized system configuration
    - models.bs_engine: Core BSEngine and OptionType
    - models.greeks: GreekCalculator and GreeksResult

Core Classes to be Implemented:
    - PositionType: Enum representing LONG (+1) or SHORT (-1) orientation.
    - StrategyLeg: Dataclass defining contract specification, quantity, and entry price.
    - PayoffResult: Dataclass capturing spot domain, gross terminal payoff, and net PnL.
    - BaseOptionStrategy: Abstract base class enforcing interface for strategy components.

Core Functions / Methods to be Implemented:
    - add_leg: Append an underlying or option leg to the strategy composition.
    - calculate_net_premium: Sum upfront premium received minus paid across all legs.
    - calculate_payoff: Evaluate terminal gross and net PnL across a spot price grid.
    - get_breakeven_points: Solve for terminal spot levels where net PnL equals zero.
    - aggregate_greeks: Compute portfolio net Greeks (Delta, Gamma, Vega, Theta, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from config import DEFAULT_CONFIG, SystemConfig
from models.bs_engine import OptionType
from models.greeks import GreeksResult


class PositionType(str, Enum):
    """Directional positioning for a strategy leg."""
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> float:
        """Return mathematical sign multiplier (+1.0 for Long, -1.0 for Short)."""
        return 1.0 if self == PositionType.LONG else -1.0


@dataclass
class StrategyLeg:
    """Individual instrument leg within an option strategy.

    Attributes:
        instrument_type: 'underlying' or 'option'.
        position_type: PositionType.LONG or PositionType.SHORT.
        quantity: Number of contracts or shares.
        entry_price: Traded execution price or current spot basis.
        strike: Option strike price (None for underlying).
        option_type: OptionType.CALL or OptionType.PUT (None for underlying).
        expiry_years: Time to expiration in years (None for underlying).
        multiplier: Contract point multiplier (e.g., 50 for TXO, 1 for stock shares).
    """
    instrument_type: str  # 'underlying' or 'option'
    position_type: PositionType
    quantity: float
    entry_price: float
    strike: Optional[float] = None
    option_type: Optional[OptionType] = None
    expiry_years: Optional[float] = None
    multiplier: float = 1.0


@dataclass
class PayoffResult:
    """Terminal payoff evaluation result.

    Attributes:
        spot_range: Array of underlying prices at expiration.
        gross_payoff: Intrinsic terminal payoff prior to initial premium.
        net_pnl: Net profit/loss taking into account initial debit/credit.
        max_profit: Theoretical upper bound on profit (or np.inf).
        max_loss: Theoretical upper bound on loss (or -np.inf).
        breakeven_points: List of underlying spot prices where net_pnl is 0.
    """
    spot_range: np.ndarray
    gross_payoff: np.ndarray
    net_pnl: np.ndarray
    max_profit: float
    max_loss: float
    breakeven_points: List[float]


class BaseOptionStrategy(ABC):
    """Abstract base class for all equity and index option strategies."""

    def __init__(
        self,
        name: str,
        underlying_symbol: str,
        config: Optional[SystemConfig] = None,
    ) -> None:
        """Initialize base strategy.

        Args:
            name: Human-readable strategy name.
            underlying_symbol: Symbol of underlying asset (e.g. 'TXO', '2330.TW').
            config: Optional system configuration instance.
        """
        self.name = name
        self.underlying_symbol = underlying_symbol
        self.config = config or DEFAULT_CONFIG
        self.legs: List[StrategyLeg] = []

    def add_leg(self, leg: StrategyLeg) -> None:
        """Add an execution leg to the strategy."""
        self.legs.append(leg)

    @abstractmethod
    def build_legs(self, **kwargs) -> None:
        """Construct the specific strategy legs based on input parameters."""
        pass

    def calculate_net_premium(self) -> float:
        """Calculate total initial net premium cash flow (positive = credit, negative = debit)."""
        raise NotImplementedError("calculate_net_premium will be implemented.")

    def calculate_payoff(
        self,
        spot_range: Optional[np.ndarray] = None,
        min_spot_pct: float = 0.80,
        max_spot_pct: float = 1.20,
        num_points: int = 200,
    ) -> PayoffResult:
        """Calculate terminal payoff and net PnL across a range of underlying spot prices."""
        raise NotImplementedError("calculate_payoff will be implemented.")

    def aggregate_greeks(
        self,
        current_spot: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> GreeksResult:
        """Compute aggregate portfolio Greeks summed across all active legs."""
        raise NotImplementedError("aggregate_greeks will be implemented.")

    def get_breakeven_points(self) -> List[float]:
        """Compute exact breakeven spot prices at expiration."""
        raise NotImplementedError("get_breakeven_points will be implemented.")
