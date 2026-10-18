"""Option Strategy Overlay and Quantitative Structure Modeling.

Role:
    Provides systematic option overlay strategy implementations widely utilized
    in Taiwan equity and index options markets for portfolio yield enhancement,
    downside tail-risk hedging, and cost-neutral corridor protection.

Dependencies:
    - .base: Abstract base class for multi-leg strategies and payoff structures.
    - .covered_call: Yield enhancement strategy (Long Underlying + Short OTM Call).
    - .protective_put: Portfolio insurance strategy (Long Underlying + Long OTM Put).
    - .collar: Cost-effective corridor hedging strategy (Long Underlying + Long Put + Short Call).

Core Classes Exposed:
    - BaseOptionStrategy: Abstract strategy base defining interface for payoff, PnL, and Greeks.
    - StrategyLeg: Representation of a single underlying or option position leg.
    - PositionType: Enum for LONG and SHORT.
    - PayoffResult: Structured container holding spot grid, payoff profile, and net PnL.
    - CoveredCallStrategy: Covered Call implementation.
    - ProtectivePutStrategy: Protective Put implementation.
    - CollarStrategy: Collar corridor strategy implementation.
"""

from .base import (
    BaseOptionStrategy,
    BaseStrategy,
    OptionLeg,
    PayoffResult,
    PositionType,
    StrategyLeg,
)
from .covered_call import CoveredCallStrategy
from .protective_put import ProtectivePutStrategy
from .collar import CollarStrategy

__all__ = [
    "BaseStrategy",
    "OptionLeg",
    "BaseOptionStrategy",
    "StrategyLeg",
    "PositionType",
    "PayoffResult",
    "CoveredCallStrategy",
    "ProtectivePutStrategy",
    "CollarStrategy",
]
