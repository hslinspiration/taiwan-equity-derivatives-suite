"""Dynamic Hedging and Portfolio Risk Replication Package.

Role:
    Provides simulation, execution, and performance attribution engines for
    dynamic delta hedging of option books and structured equity products
    under realistic Taiwan market frictions (transaction taxes, brokerage fees,
    discrete rebalancing, and execution slippage).

Dependencies:
    - .delta_hedger: Dynamic delta hedger implementation and simulation containers.

Core Classes & Functions Exposed:
    - DeltaHedger: Discrete and threshold-based delta hedging simulator.
    - HedgingStepRecord: Dataclass recording single rebalance step diagnostics.
    - HedgingSimulationResult: Aggregated summary of hedge effectiveness, tracking error,
      and PnL decomposition.
    - simulate_delta_hedging: High-level helper running an end-to-end delta hedging path.
"""

from .delta_hedger import (
    DeltaHedger,
    DeskPosition,
    HedgingSimulationResult,
    HedgingStepRecord,
    simulate_delta_hedging,
)

__all__ = [
    "DeltaHedger",
    "DeskPosition",
    "HedgingStepRecord",
    "HedgingSimulationResult",
    "simulate_delta_hedging",
]
