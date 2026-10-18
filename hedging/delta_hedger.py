"""Trading Desk Dynamic Delta Hedging & PnL Attribution Engine.

Role:
    Simulates discrete daily delta hedging of an institutional OTC or exchange-traded
    option position on a Taiwan equity desk (1 contract = 1,000 shares = 1 TWSE board lot).
    Performs second-order Taylor series PnL decomposition (Delta, Gamma convexity drag,
    Theta time-decay collection, and statutory Taiwan transaction friction costs) to
    attribute realized hedging performance and isolate higher-order residual tracking error.

Taiwan Market Friction Mechanics:
    - Brokerage Commission (`config.BROKERAGE_FEE_RATE` = 0.1425%): Levied on both buy
      and sell stock rebalancing executions.
    - Securities Transaction Tax (`config.STOCK_TRANSACTION_TAX` = 0.30%): Levied on
      all stock sell executions (`delta_shares < 0`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

import config
from models.bs_engine import BSEngine, OptionType
from models.desk_calibrator import DeskMarketParams
from models.greeks import GreeksCalculator


@dataclass
class DeskPosition:
    """Represents an institutional option book position to be dynamically delta-hedged.

    Attributes:
        option_type: 'call' or 'put' (case-insensitive).
        position: +1 for Long option, -1 for Short option.
        strike: Option strike price in TWD.
        maturity_years: Initial time to maturity in years (e.g., 30 / 365.0).
        contracts: Number of option contracts (1 contract = 1,000 underlying shares).
        entry_premium: Per-share option premium at trade inception.
    """

    option_type: str
    position: int
    strike: float
    maturity_years: float
    contracts: int
    entry_premium: float

    def __post_init__(self) -> None:
        """Normalize and validate position fields."""
        opt = str(self.option_type).strip().lower()
        if opt not in ("call", "put"):
            raise ValueError(f"Invalid option_type '{self.option_type}'. Expected 'call' or 'put'.")
        self.option_type = opt

        if int(self.position) not in (1, -1):
            raise ValueError(f"Invalid position '{self.position}'. Expected +1 (Long) or -1 (Short).")
        self.position = int(self.position)
        self.strike = float(self.strike)
        self.maturity_years = float(self.maturity_years)
        self.contracts = int(self.contracts)
        self.entry_premium = float(self.entry_premium)


@dataclass
class HedgingStepRecord:
    """Single discrete rebalancing step diagnostic container."""

    step_index: int
    time_to_maturity: float
    spot_price: float
    option_price: float
    option_delta: float
    hedge_shares: float
    trade_shares: float
    cash_balance: float
    portfolio_value: float
    transaction_cost: float


@dataclass
class HedgingSimulationResult:
    """Aggregated output from a completed delta hedging simulation."""

    history: pd.DataFrame
    initial_spot: float
    strike: float
    total_pnl: float
    total_transaction_costs: float
    tracking_error: float
    rebalance_count: int


class DeltaHedger:
    """Institutional dynamic delta hedging and Taylor PnL attribution engine."""

    SHARES_PER_CONTRACT: int = 1000

    def __init__(
        self,
        bs_engine: Optional[BSEngine] = None,
        greeks_calc: Optional[GreeksCalculator] = None,
        desk_params: Optional[DeskMarketParams] = None,
    ) -> None:
        """Initialize DeltaHedger with pricing engine, Greek calculator, and desk parameters.

        Args:
            bs_engine: BSEngine instance (defaults to a new BSEngine()).
            greeks_calc: GreeksCalculator instance (defaults to a new GreeksCalculator()).
            desk_params: Optional DeskMarketParams configuration.
        """
        if isinstance(bs_engine, config.SystemConfig):
            self.bs_engine = BSEngine(bs_engine)
            self.greeks_calc = GreeksCalculator(bs_engine)
        else:
            self.bs_engine = bs_engine if bs_engine is not None else BSEngine()
            self.greeks_calc = greeks_calc if greeks_calc is not None else GreeksCalculator()
        self.desk_params: DeskMarketParams = (
            desk_params if desk_params is not None else DeskMarketParams()
        )

    def _unit_option_price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """Evaluate theoretical per-share option value."""
        if option_type == "call":
            return float(self.bs_engine.call_price(spot, strike, maturity, r, q, sigma))
        return float(self.bs_engine.put_price(spot, strike, maturity, r, q, sigma))

    @staticmethod
    def _compute_trade_friction(delta_shares: float, spot_price: float) -> float:
        """Calculate Taiwan equity transaction cost on a stock hedge trade.

        Applies standard TWSE brokerage fee (0.1425%) to all stock trades and the
        statutory Taiwan securities transaction tax (0.30%) to stock sale proceeds.
        """
        traded_notional = abs(float(delta_shares)) * float(spot_price)
        if traded_notional == 0.0:
            return 0.0

        brokerage_fee = traded_notional * config.BROKERAGE_FEE_RATE
        # Selling shares (delta_shares < 0) incurs the 0.3% Taiwan securities transaction tax
        transfer_tax = (
            traded_notional * config.STOCK_TRANSACTION_TAX if delta_shares < 0.0 else 0.0
        )
        return float(brokerage_fee + transfer_tax)

    def compute_initial_hedge(
        self,
        spot_price: float,
        desk_position: DeskPosition,
        r: float,
        q: float,
        sigma: float,
    ) -> Dict[str, Union[float, int]]:
        """Compute initial delta-neutral stock hedge and inception Greek exposures.

        Args:
            spot_price: Current underlying equity spot price in TWD.
            desk_position: DeskPosition describing the option book trade.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            dict[str, float]: Dictionary containing `net_option_delta`, `shares_to_hedge`,
                `hedge_notional_twd`, `initial_net_gamma`, `initial_net_vega`, and
                `initial_net_theta`.
        """
        multiplier = float(
            desk_position.position * desk_position.contracts * self.SHARES_PER_CONTRACT
        )
        greeks = self.greeks_calc.all_greeks(
            S=spot_price,
            K=desk_position.strike,
            T=desk_position.maturity_years,
            r=r,
            q=q,
            sigma=sigma,
            option_type=desk_position.option_type,
        )

        net_option_delta = float(multiplier * float(greeks["delta"]))
        shares_to_hedge = int(round(-net_option_delta))
        hedge_notional_twd = float(abs(shares_to_hedge) * spot_price)
        initial_net_gamma = float(multiplier * float(greeks["gamma"]))
        initial_net_vega = float(multiplier * float(greeks["vega"]))
        initial_net_theta = float(multiplier * float(greeks["theta"]))

        return {
            "net_option_delta": net_option_delta,
            "shares_to_hedge": shares_to_hedge,
            "hedge_notional_twd": hedge_notional_twd,
            "initial_net_gamma": initial_net_gamma,
            "initial_net_vega": initial_net_vega,
            "initial_net_theta": initial_net_theta,
        }

    def simulate_dynamic_hedging(
        self,
        price_path: np.ndarray,
        desk_position: DeskPosition,
        r: float,
        q: float,
        sigma: float,
        dt: float = 1.0 / 252.0,
    ) -> pd.DataFrame:
        """Simulate daily discrete delta rebalancing and Taylor PnL decomposition across a spot path.

        Args:
            price_path: 1D NumPy array of daily underlying spot prices.
            desk_position: DeskPosition being hedged.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.
            dt: Time step in years between observations (default 1/252).

        Returns:
            pd.DataFrame: Daily attribution table with columns:
                `['Day', 'Spot', 'Delta', 'Stock_Hedge_Shares', 'Option_PnL',
                  'Stock_PnL', 'Friction_Cost', 'Net_Daily_PnL', 'Cumulative_PnL',
                  'Gamma_PnL_Est', 'Theta_PnL_Est']`.
        """
        spots = np.asarray(price_path, dtype=float)
        if spots.ndim != 1 or len(spots) < 2:
            raise ValueError("price_path must be a 1D array with at least 2 daily observations.")

        multiplier = float(
            desk_position.position * desk_position.contracts * self.SHARES_PER_CONTRACT
        )
        t_initial = float(desk_position.maturity_years)

        records = []
        previous_stock: float = 0.0
        prev_option_val: float = 0.0
        prev_net_gamma: float = 0.0
        prev_net_theta: float = 0.0
        cumulative_pnl: float = 0.0

        for t, spot_t in enumerate(spots):
            remaining_t = max(t_initial - t * dt, 1e-5)

            unit_price = self._unit_option_price(
                spot=float(spot_t),
                strike=desk_position.strike,
                maturity=remaining_t,
                r=r,
                q=q,
                sigma=sigma,
                option_type=desk_position.option_type,
            )
            current_option_val = multiplier * unit_price

            unit_delta = float(
                self.greeks_calc.delta(
                    S=float(spot_t),
                    K=desk_position.strike,
                    T=remaining_t,
                    r=r,
                    q=q,
                    sigma=sigma,
                    option_type=desk_position.option_type,
                )
            )
            unit_gamma = float(
                self.greeks_calc.gamma(
                    S=float(spot_t),
                    K=desk_position.strike,
                    T=remaining_t,
                    r=r,
                    q=q,
                    sigma=sigma,
                )
            )
            unit_theta = float(
                self.greeks_calc.theta(
                    S=float(spot_t),
                    K=desk_position.strike,
                    T=remaining_t,
                    r=r,
                    q=q,
                    sigma=sigma,
                    option_type=desk_position.option_type,
                )
            )

            net_gamma = multiplier * unit_gamma
            net_theta = multiplier * unit_theta

            target_stock = -multiplier * unit_delta
            delta_shares = target_stock - previous_stock
            friction_cost = self._compute_trade_friction(delta_shares, float(spot_t))

            if t == 0:
                option_pnl = 0.0
                stock_pnl = 0.0
                gamma_pnl_approx = 0.0
                theta_pnl_approx = 0.0
            else:
                delta_s = float(spot_t - spots[t - 1])
                option_pnl = float(current_option_val - prev_option_val)
                stock_pnl = float(previous_stock * delta_s)
                gamma_pnl_approx = float(0.5 * prev_net_gamma * (delta_s**2))
                theta_pnl_approx = float(prev_net_theta * (dt * 365.0))

            net_daily_pnl = float(option_pnl + stock_pnl - friction_cost)
            cumulative_pnl += net_daily_pnl

            records.append(
                {
                    "Day": int(t),
                    "Spot": float(spot_t),
                    "Delta": float(unit_delta),
                    "Stock_Hedge_Shares": float(target_stock),
                    "Option_PnL": float(option_pnl),
                    "Stock_PnL": float(stock_pnl),
                    "Friction_Cost": float(friction_cost),
                    "Net_Daily_PnL": float(net_daily_pnl),
                    "Cumulative_PnL": float(cumulative_pnl),
                    "Gamma_PnL_Est": float(gamma_pnl_approx),
                    "Theta_PnL_Est": float(theta_pnl_approx),
                }
            )

            previous_stock = target_stock
            prev_option_val = current_option_val
            prev_net_gamma = net_gamma
            prev_net_theta = net_theta

        return pd.DataFrame(records)

    @staticmethod
    def pnl_attribution_summary(hedge_df: pd.DataFrame) -> Dict[str, float]:
        """Aggregate daily hedging PnL into institutional attribution components.

        Decomposes realized hedging PnL into Gamma convexity drag/gain, Theta decay
        collection, Taiwan transaction costs, and higher-order Taylor residual.

        Args:
            hedge_df: DataFrame returned by `simulate_dynamic_hedging`.

        Returns:
            dict[str, float]: Dictionary with keys `total_realized_pnl`,
                `cumulative_option_pnl`, `cumulative_stock_pnl`, `total_transaction_costs`,
                `gamma_drag_total`, `theta_collected_total`, and `unexplained_residual`.
        """
        total_realized_pnl = float(hedge_df["Net_Daily_PnL"].sum())
        cumulative_option_pnl = float(hedge_df["Option_PnL"].sum())
        cumulative_stock_pnl = float(hedge_df["Stock_PnL"].sum())
        total_transaction_costs = float(hedge_df["Friction_Cost"].sum())
        gamma_drag_total = float(hedge_df["Gamma_PnL_Est"].sum())
        theta_collected_total = float(hedge_df["Theta_PnL_Est"].sum())

        explained_pnl = gamma_drag_total + theta_collected_total - total_transaction_costs
        unexplained_residual = float(total_realized_pnl - explained_pnl)

        return {
            "total_realized_pnl": total_realized_pnl,
            "cumulative_option_pnl": cumulative_option_pnl,
            "cumulative_stock_pnl": cumulative_stock_pnl,
            "total_transaction_costs": total_transaction_costs,
            "gamma_drag_total": gamma_drag_total,
            "theta_collected_total": theta_collected_total,
            "unexplained_residual": unexplained_residual,
        }


def simulate_delta_hedging(
    spot_path: np.ndarray,
    time_grid: np.ndarray,
    strike: float,
    option_type: Union[OptionType, str],
    volatility: float,
    risk_free_rate: float = config.FALLBACK_RISK_FREE_RATE,
    dividend_yield: float = 0.035,
    band_threshold: Optional[float] = None,
) -> HedgingSimulationResult:
    """High-level helper running a dynamic delta hedge simulation."""
    _ = band_threshold
    hedger = DeltaHedger()
    opt = option_type.value if isinstance(option_type, OptionType) else str(option_type)
    maturity_years = float(time_grid[0]) if len(time_grid) > 0 else 30.0 / 365.0
    dt = float(maturity_years / max(len(spot_path) - 1, 1))

    entry_prem = hedger._unit_option_price(
        spot=float(spot_path[0]),
        strike=strike,
        maturity=maturity_years,
        r=risk_free_rate,
        q=dividend_yield,
        sigma=volatility,
        option_type=opt.lower(),
    )
    pos = DeskPosition(
        option_type=opt,
        position=-1,
        strike=strike,
        maturity_years=maturity_years,
        contracts=1,
        entry_premium=entry_prem,
    )
    df = hedger.simulate_dynamic_hedging(
        price_path=spot_path,
        desk_position=pos,
        r=risk_free_rate,
        q=dividend_yield,
        sigma=volatility,
        dt=dt,
    )
    return HedgingSimulationResult(
        history=df,
        initial_spot=float(spot_path[0]),
        strike=float(strike),
        total_pnl=float(df["Net_Daily_PnL"].sum()),
        total_transaction_costs=float(df["Friction_Cost"].sum()),
        tracking_error=float(df["Net_Daily_PnL"].std(ddof=1)),
        rebalance_count=len(df),
    )
