"""Institutional OTC Derivatives Desk Quote Calibration Layer.

Role:
    Bridges theoretical Black-Scholes-Merton fair values (flat-volatility mid prices)
    with realistic institutional OTC / structured equity derivatives desk quotes in
    the Taiwan market. Accounts for volatility skew across moneyness, TWSE Securities
    Borrowing and Lending (SBL / 借券) costs when short-hedging put deltas, dynamic
    delta rebalancing frictions driven by Taiwan's 0.3% securities transaction tax,
    and commercial desk profit margins.

Taiwan Market Microstructure Context:
    1. Volatility Skew (`skew_beta`):
       Taiwan institutional investors (insurers, domestic funds, and foreign QFIIs)
       exhibit strong structural demand for downside tail protection on heavyweight
       semiconductor equities (e.g., TSMC 2330.TW) and TAIEX index exposure. This
       creates a pronounced negative volatility skew where out-of-the-money (OTM)
       puts trade at significantly higher implied volatilities than at-the-money (ATM)
       or OTM calls.
    2. Securities Borrowing & Lending Cost (`borrow_cost` / 借券費率):
       When a derivatives desk sells a Put (or buys a Call) and delta-hedges by
       shorting the underlying Taiwan stock via the TWSE SBL platform, it incurs an
       annualized borrowing fee (typically 1.5% to 3.0% for liquid large-caps),
       which acts as an additional carry drag alongside the dividend yield `q`.
    3. Dynamic Hedging Friction & Statutory Tax Buffer (`hedge_friction_pct`):
       Every time the trading desk sells underlying shares to rebalance its delta
       hedge, it pays the statutory 0.3% (30 bps) Taiwan securities transaction tax
       plus brokerage commissions and bid-ask spread slippage. Desks embed a
       hedging friction buffer into their internal breakeven price.
    4. Desk Markup & Negotiation Slack (`desk_markup_pct`, `negotiation_slack`):
       The difference between the initial `client_quote` and the internal
       `desk_breakeven` represents the commercial margin (`negotiation_slack`)
       available for institutional price discovery and structuring negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

import config
from .bs_engine import BSEngine


@dataclass
class DeskMarketParams:
    """Institutional trading desk calibration parameters for Taiwan equity derivatives.

    Attributes:
        skew_beta: Volatility skew slope across log-moneyness (default 0.45).
        borrow_cost: Annualized TWSE stock borrowing cost (SBL fee, default 0.020).
        hedge_friction_pct: Dynamic delta hedging friction & 0.3% stock transaction
            tax buffer (default 0.03).
        desk_markup_pct: Desk commercial profit markup over breakeven (default 0.08).
    """

    skew_beta: float = config.DEFAULT_SKEW_BETA
    borrow_cost: float = config.DEFAULT_BORROW_COST
    hedge_friction_pct: float = config.DEFAULT_HEDGE_FRICTION
    desk_markup_pct: float = config.DEFAULT_DESK_MARKUP


class DeskCalibrator:
    """Calibrates theoretical BSM prices into actionable Taiwan institutional desk quotes."""

    def __init__(
        self,
        bs_engine: BSEngine,
        params: Optional[DeskMarketParams] = None,
    ) -> None:
        """Initialize DeskCalibrator with a pricing engine and market friction parameters.

        Args:
            bs_engine: Initialized BSEngine instance.
            params: Optional DeskMarketParams configuration; defaults to DeskMarketParams().
        """
        self.bs_engine: BSEngine = bs_engine
        self.params: DeskMarketParams = params if params is not None else DeskMarketParams()

    def get_skew_adjusted_vol(
        self,
        S: float,
        K: float,
        base_sigma: float,
    ) -> float:
        """Calculate skew-adjusted implied volatility for strike K relative to spot S.

        Applies a log-moneyness skew parametrization:
            sigma_skew(K) = base_sigma * (1.0 - skew_beta * ln(K / S))

        For downside strikes (K < S), ln(K / S) < 0, which elevates implied volatility
        to reflect institutional put protection demand. Bounded below by 5% (0.05) to
        prevent degenerate volatility on extreme upside strikes.

        Args:
            S: Current underlying spot price.
            K: Option strike price.
            base_sigma: Baseline at-the-money (ATM) annualized volatility.

        Returns:
            float: Skew-adjusted annualized volatility.
        """
        if S <= 0.0 or K <= 0.0:
            raise ValueError(f"Spot S ({S}) and strike K ({K}) must be strictly positive.")
        if base_sigma <= 0.0:
            return 0.0

        log_moneyness = float(np.log(K / S))
        skew_factor = 1.0 - self.params.skew_beta * log_moneyness
        adjusted_vol = float(base_sigma * max(skew_factor, 0.25))
        return max(adjusted_vol, 0.05)

    def calculate_desk_quotes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        base_sigma: float,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """Compute theoretical mid, desk hedging breakeven, and client quote breakdown.

        Methodology:
            1. `theoretical_mid`: Standard BSM price using flat `base_sigma`, `r`, and `q`.
            2. `skew_adjusted_vol`: Implied volatility adjusted for strike skew via
               `get_skew_adjusted_vol(S, K, base_sigma)`.
            3. `desk_breakeven`: BSM price evaluated at `skew_adjusted_vol` and effective
               carry (incorporating `borrow_cost` into dividend/borrow drag when shorting
               stock to hedge), scaled by `(1.0 + hedge_friction_pct)` to cover Taiwan's
               0.3% equity transaction tax and gamma rebalancing slippage.
            4. `client_quote`: `desk_breakeven * (1.0 + desk_markup_pct)`.
            5. `negotiation_slack`: `client_quote - desk_breakeven` (maximum pricing
               concession available during institutional client negotiation).
            6. `markup_pct`: Effective commercial markup percentage (`desk_markup_pct`).

        Args:
            S: Underlying spot price in TWD.
            K: Option strike price in TWD.
            T: Time to maturity in years.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            base_sigma: Baseline ATM volatility.
            option_type: 'call' or 'put' (case-insensitive, default 'call').

        Returns:
            dict[str, float]: Dictionary containing `theoretical_mid`, `skew_adjusted_vol`,
                `desk_breakeven`, `client_quote`, `negotiation_slack`, and `markup_pct`.
        """
        opt = str(option_type).strip().lower()
        if opt not in ("call", "put"):
            raise ValueError(f"Unsupported option_type '{option_type}'. Expected 'call' or 'put'.")

        skew_vol = self.get_skew_adjusted_vol(S=S, K=K, base_sigma=base_sigma)

        if opt == "call":
            theoretical_mid = float(
                self.bs_engine.call_price(S=S, K=K, T=T, r=r, q=q, sigma=base_sigma)
            )
            # Short call delta-hedging requires buying underlying shares; borrow cost applies
            # minimally on unwind, while skew and dynamic tax/rebalancing friction dominate.
            effective_q = q
            skew_price = float(
                self.bs_engine.call_price(S=S, K=K, T=T, r=r, q=effective_q, sigma=skew_vol)
            )
        else:
            theoretical_mid = float(
                self.bs_engine.put_price(S=S, K=K, T=T, r=r, q=q, sigma=base_sigma)
            )
            # Selling a put requires shorting underlying shares via TWSE SBL to delta-hedge,
            # increasing effective carry drag by `borrow_cost`.
            effective_q = q + self.params.borrow_cost
            skew_price = float(
                self.bs_engine.put_price(S=S, K=K, T=T, r=r, q=effective_q, sigma=skew_vol)
            )

        desk_breakeven = float(skew_price * (1.0 + self.params.hedge_friction_pct))
        client_quote = float(desk_breakeven * (1.0 + self.params.desk_markup_pct))
        negotiation_slack = float(client_quote - desk_breakeven)
        markup_pct = float(self.params.desk_markup_pct)

        return {
            "theoretical_mid": theoretical_mid,
            "skew_adjusted_vol": skew_vol,
            "desk_breakeven": desk_breakeven,
            "client_quote": client_quote,
            "negotiation_slack": negotiation_slack,
            "markup_pct": markup_pct,
        }
