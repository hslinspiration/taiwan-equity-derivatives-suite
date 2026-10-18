"""Central Configuration Module for Taiwan Equity Derivatives Suite.

Role:
    Provides centralized, immutable, and production-grade configurations
    for financial engineering computations, Taiwan market microstructure constants,
    statutory transaction taxes, exchange fees, numerical solver tolerances,
    and simulation parameters.

Institutional Market Microstructure Context:
    - Securities Transaction Tax (證券交易稅): Governed by the Taiwan Securities
      Transaction Tax Act. A statutory tax of 0.3% (0.003) is levied on the gross
      sale proceeds of TWSE/TPEx listed equities.
    - Options & Futures Transaction Tax (期貨交易稅): Governed by the Taiwan Futures
      Transaction Tax Act. Options transactions are taxed based on premium settled.
    - Brokerage Commissions (經紀手續費): Standard statutory ceiling set by TWSE/TPEx
      at 0.1425% (0.001425) of traded notional before institutional tiered discounts.
    - Risk-Free Rate Benchmarks: Taiwan Central Bank (CBC) rediscount rate and
      short-term interbank offered rates (TAIBOR / CP 90-day rate) represent the
      standard risk-free benchmark for local NTD derivative discounting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =====================================================================
# Central Market Microstructure & Valuation Constants
# =====================================================================
DEFAULT_TICKER: str = "2330.TW"
"""Default target equity ticker (TSMC - Taiwan Semiconductor Manufacturing Co., Ltd.)."""

FALLBACK_RISK_FREE_RATE: float = 0.015
"""1.5% proxy for Taiwan CBC discount rate / CP benchmark interest rate."""

TRADING_DAYS_PER_YEAR: int = 252
"""Standard annualized trading days convention for TWSE and TAIFEX."""

DEFAULT_HV_WINDOW: int = 60
"""60-day lookback window for medium-term institutional volatility benchmark."""

STOCK_TRANSACTION_TAX: float = 0.003
"""0.3% Taiwan statutory securities transaction tax on underlying equity sell-side."""

OPTION_TRANSACTION_TAX: float = 0.00002
"""Taiwan options transaction tax rate specification."""

BROKERAGE_FEE_RATE: float = 0.001425
"""Standard TWSE/TPEx brokerage commission rate (0.1425%) before institutional discount."""

# =====================================================================
# Institutional Desk Quote Calibration Defaults
# =====================================================================
DEFAULT_SKEW_BETA: float = 0.45
"""Default volatility skew slope across log-moneyness (4.5 vol points per 10% moneyness)."""

DEFAULT_BORROW_COST: float = 0.020
"""Default annualized TWSE Securities Borrowing and Lending (SBL / 借券) fee rate (2.0%)."""

DEFAULT_HEDGE_FRICTION: float = 0.03
"""Default dynamic delta hedging friction and 0.3% Taiwan stock transaction tax buffer (3.0%)."""

DEFAULT_DESK_MARKUP: float = 0.08
"""Default OTC derivatives trading desk profit markup over hedging breakeven (8.0%)."""


# =====================================================================
# Configuration Dataclasses
# =====================================================================
@dataclass(frozen=True)
class MarketConfig:
    """Market conventions and baseline macroeconomic rates for Taiwan equities."""

    default_ticker: str = DEFAULT_TICKER
    default_risk_free_rate: float = FALLBACK_RISK_FREE_RATE
    default_dividend_yield: float = 0.0350  # ~3.50% TAIEX baseline annualized dividend yield
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    default_hv_window: int = DEFAULT_HV_WINDOW
    calendar_days_per_year: float = 365.0
    day_count_convention: str = "ACT/365"


@dataclass(frozen=True)
class TaifexContractConfig:
    """TAIFEX derivative specifications for TAIEX Options (TXO) and Futures (TX/MTX)."""

    txo_multiplier: float = 50.0  # NT$50 per index point for TXO options
    tx_multiplier: float = 200.0  # NT$200 per index point for TX (Large TAIEX Futures)
    mtx_multiplier: float = 50.0  # NT$50 per index point for MTX (Mini TAIEX Futures)
    stock_option_shares: int = 2000  # Standard Taiwan single equity option contract = 2,000 shares
    exercise_style: str = "European"  # TAIFEX TXO and stock options standard exercise style


@dataclass(frozen=True)
class TransactionCostConfig:
    """Taiwan market transaction taxes, statutory fees, and execution friction costs."""

    option_tax_rate: float = OPTION_TRANSACTION_TAX
    futures_tax_rate: float = 0.00002  # 0.002% (1/50,000) for index futures
    stock_tax_rate: float = STOCK_TRANSACTION_TAX
    brokerage_fee_rate: float = BROKERAGE_FEE_RATE
    brokerage_fee_per_contract: float = 15.0  # Estimated standard broker commission per contract (NTD)
    slippage_ticks: float = 1.0  # Assumed execution slippage in minimum tick size


@dataclass(frozen=True)
class ModelConfig:
    """Numerical solver tolerances, finite-difference bump sizes, and pricing engine defaults."""

    iv_solver_tolerance: float = 1e-6
    iv_solver_max_iter: int = 100
    iv_solver_method: str = "brent"  # 'brent' or 'newton_raphson'
    fd_bump_spot: float = 1e-4  # Relative bump size for numerical Delta/Gamma
    fd_bump_vol: float = 1e-4  # Absolute bump size for numerical Vega/Volga
    fd_bump_time: float = 1.0 / 365.0  # 1 calendar day bump for Theta


@dataclass(frozen=True)
class HedgingConfig:
    """Parameters governing dynamic delta hedging and portfolio rebalancing simulations."""

    rebalance_frequency: str = "1D"  # Rebalancing period: '1D', '1H', 'continuous'
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
    """Instantiate and return the production default system configuration.

    Returns:
        SystemConfig: Initialized system configuration object.
    """
    return SystemConfig()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from a specified file or fallback to production defaults.

    Args:
        config_path: Optional filesystem path to JSON/YAML configuration file.

    Returns:
        SystemConfig: Fully populated configuration object.
    """
    # Reserved for JSON/YAML custom overrides; currently returns production default
    return get_default_config()


# Global default configuration instance
DEFAULT_CONFIG: SystemConfig = get_default_config()
