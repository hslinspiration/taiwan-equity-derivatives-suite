"""Covered Call Yield Enhancement Overlay Strategy.

Role:
    Implements an institutional Covered Call (Buy-Write) strategy combining a
    core Taiwan equity holding (1 board lot = 1,000 shares) with a short
    out-of-the-money (OTM) European Call option. Widely deployed by Taiwanese
    insurance funds, high-dividend ETFs, and long-only equity mandates to harvest
    volatility risk premium and monetize upside beyond target price levels.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models.bs_engine import BSEngine
from .base import BaseStrategy, OptionLeg


class CoveredCallStrategy(BaseStrategy):
    """Institutional Covered Call (Long Stock + Short OTM Call) strategy engine."""

    def __init__(
        self,
        spot_price: float,
        underlying_shares: int = 1000,
        call_strike: Optional[float] = None,
        maturity_days: int = 30,
    ) -> None:
        """Initialize CoveredCallStrategy and attach the short call leg.

        Args:
            spot_price: Current underlying equity spot price in TWD.
            underlying_shares: Number of underlying shares held (default 1,000 shares).
            call_strike: Call exercise strike price. Defaults to 5% OTM (`spot_price * 1.05`).
            maturity_days: Days to expiration (default 30 calendar days).
        """
        super().__init__(spot_price=spot_price, underlying_shares=underlying_shares)

        if maturity_days <= 0:
            raise ValueError(f"maturity_days must be strictly positive, got {maturity_days}.")

        self.maturity_days: int = int(maturity_days)
        self.maturity: float = float(self.maturity_days) / 365.0
        self.call_strike: float = (
            float(spot_price * 1.05) if call_strike is None else float(call_strike)
        )

        self.add_leg(
            OptionLeg(
                option_type="call",
                position=-1,
                strike=self.call_strike,
                maturity=self.maturity,
                contracts=1,
            )
        )

    def evaluate(
        self,
        bs_engine: BSEngine,
        r: float,
        q: float,
        sigma: float,
    ) -> Dict[str, Any]:
        """Evaluate Covered Call pricing, institutional yield metrics, and portfolio Greeks.

        Args:
            bs_engine: BSEngine instance for pricing the short call leg.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            dict: Complete dictionary of strategy metrics and portfolio Greeks.
        """
        call_price = float(
            bs_engine.call_price(
                S=self.spot_price,
                K=self.call_strike,
                T=self.maturity,
                r=r,
                q=q,
                sigma=sigma,
            )
        )

        premium_received_twd = float(call_price * self.underlying_shares)
        yield_enhancement_annual = float(
            (call_price / self.spot_price) * (365.0 / self.maturity_days)
        )
        downside_buffer_pct = float(call_price / self.spot_price)
        max_upside_return_pct = float(
            (self.call_strike - self.spot_price + call_price) / self.spot_price
        )
        breakeven_price = float(self.spot_price - call_price)

        greeks = self.calculate_portfolio_greeks(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

        return {
            "strategy_name": "Covered Call",
            "spot_price": self.spot_price,
            "call_strike": self.call_strike,
            "maturity_days": self.maturity_days,
            "call_price": call_price,
            "premium_received_twd": premium_received_twd,
            "yield_enhancement_annual": yield_enhancement_annual,
            "downside_buffer_pct": downside_buffer_pct,
            "max_upside_return_pct": max_upside_return_pct,
            "breakeven_price": breakeven_price,
            "portfolio_greeks": greeks,
            **greeks,
        }
