"""Central Configuration Module for Taiwan Equity Derivatives Suite.

Role:
    Provides centralized, immutable, and environment-configurable settings
    for financial engineering computations, market microstructure constants,
    numerical solver tolerances, and simulation parameters.

Dependencies:
    - dataclasses / pydantic: Configuration schema enforcement and typing
    - pathlib: File system path manipulation
    - typing: Static type hint annotations

Core Classes to be Implemented:
    - MarketConfig: Central bank risk-free rates, dividend yield seasonality,
      and trading calendar conventions.
    - TaifexContractConfig: Contract multipliers, tick sizes, exercise styles,
      and settlement specifications for TX, MTX, and TXO options.
    - TransactionCostConfig: Commission, exchange fees, and Taiwan futures/option
      transaction taxes (0.0001 for futures, 0.00125 for options premium).
    - ModelConfig: Numerical tolerance, max iterations for root finding,
      bump sizes for finite difference Greeks, and pricing engine defaults.
    - HedgingConfig: Rebalancing frequency, delta threshold bands, and risk limits.
    - SystemConfig: Master wrapper containing references to all sub-configs
      and project directories.

Core Functions to be Implemented:
    - load_config(config_path: Optional[str]) -> SystemConfig: Load configuration
      from YAML/JSON or fallback to default production settings.
    - get_default_config() -> SystemConfig: Retrieve pre-configured default settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MarketConfig:
    """Market conventions and baseline macroeconomic rates."""
    default_risk_free_rate: float = 0.0160  # ~1.60% Taiwan Central Bank discount rate / TAIBOR
    default_dividend_yield: float = 0.0350  # ~3.50% TAIEX baseline annualized dividend yield
    trading_days_per_year: int = 252
    calendar_days_per_year: float = 365.0
    day_count_convention: str = "ACT/365"


@dataclass(frozen=True)
class TaifexContractConfig:
    """TAIFEX derivative specifications for TAIEX Options (TXO) and Futures (TX/MTX)."""
    txo_multiplier: float = 50.0       # NT$50 per index point for TXO
    tx_multiplier: float = 200.0       # NT$200 per index point for TX (Large TAIEX Futures)
    mtx_multiplier: float = 50.0       # NT$50 per index point for MTX (Mini TAIEX Futures)
    stock_option_shares: int = 2000    # Standard Taiwan equity option contract = 2,000 shares
    exercise_style: str = "European"   # TAIFEX TXO is European-style exercise


@dataclass(frozen=True)
class TransactionCostConfig:
    """Taiwan market transaction taxes and execution costs."""
    option_tax_rate: float = 0.00125    # 0.125% of premium settled under Taiwan tax laws
    futures_tax_rate: float = 0.00002   # 0.002% (1/50000) for TAIEX futures
    stock_tax_rate: float = 0.0030      # 0.3% Taiwan securities transaction tax for underlying
    brokerage_fee_per_contract: float = 15.0  # Estimated standard broker commission in NTD
    slippage_ticks: float = 1.0         # Assumed execution slippage in minimum tick size


@dataclass(frozen=True)
class ModelConfig:
    """Numerical tolerances and algorithm parameters."""
    iv_solver_tolerance: float = 1e-6
    iv_solver_max_iter: int = 100
    iv_solver_method: str = "brent"     # 'brent', 'newton_raphson', or 'halley'
    fd_bump_spot: float = 1e-4          # Relative bump for numerical Delta/Gamma
    fd_bump_vol: float = 1e-4           # Bump for numerical Vega/Volga
    fd_bump_time: float = 1.0 / 365.0   # 1 day bump for Theta


@dataclass(frozen=True)
class HedgingConfig:
    """Parameters governing dynamic delta hedging simulations."""
    rebalance_frequency: str = "1D"     # Rebalancing period: '1D', '1H', 'continuous'
    delta_threshold_band: float = 0.05  # Rebalance only if |delta_current - target| > band
    max_leverage: float = 3.0


@dataclass(frozen=True)
class SystemConfig:
    """Master configuration encapsulating all subsystem settings and directory paths."""
    market: MarketConfig = field(default_factory=MarketConfig)
    taifex: TaifexContractConfig = field(default_factory=TaifexContractConfig)
    costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hedging: HedgingConfig = field(default_factory=HedgingConfig)

    # Path configurations
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "cache")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "output")


def get_default_config() -> SystemConfig:
    """Instantiate and return the production default configuration.

    Returns:
        SystemConfig: Initialized system configuration object.
    """
    return SystemConfig()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from a specified file or fallback to defaults.

    Args:
        config_path: Optional filesystem path to JSON/YAML configuration file.

    Returns:
        SystemConfig: Fully populated configuration object.
    """
    # Placeholder: In full implementation, deserialize from JSON/YAML
    return get_default_config()


# Global default configuration instance
DEFAULT_CONFIG = get_default_config()
