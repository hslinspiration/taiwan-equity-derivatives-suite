"""Protective Put Downside Tail-Risk Hedging Strategy.

Role:
    Implements an institutional Protective Put (Married Put / Portfolio Insurance)
    overlay combining a core Taiwan equity holding (1 board lot = 1,000 shares)
    with a long out-of-the-money (OTM) European Put option. Establishes a hard
    downside floor price while preserving uncapped upside participation during
    geopolitical or semiconductor cycle drawdowns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models.bs_engine import BSEngine
from .base import BaseStrategy, OptionLeg


class ProtectivePutStrategy(BaseStrategy):
    """Institutional Protective Put (Long Stock + Long OTM Put) strategy engine."""

    def __init__(
        self,
        spot_price: float,
        underlying_shares: int = 1000,
        put_strike: Optional[float] = None,
        maturity_days: int = 30,
    ) -> None:
        """Initialize ProtectivePutStrategy and attach the long put leg.

        Args:
            spot_price: Current underlying equity spot price in TWD.
            underlying_shares: Number of underlying shares held (default 1,000 shares).
            put_strike: Put exercise strike price. Defaults to 7% OTM (`spot_price * 0.93`).
            maturity_days: Days to expiration (default 30 calendar days).
        """
        super().__init__(spot_price=spot_price, underlying_shares=underlying_shares)

        if maturity_days <= 0:
            raise ValueError(f"maturity_days must be strictly positive, got {maturity_days}.")

        self.maturity_days: int = int(maturity_days)
        self.maturity: float = float(self.maturity_days) / 365.0
        self.put_strike: float = (
            float(spot_price * 0.93) if put_strike is None else float(put_strike)
        )

        self.add_leg(
            OptionLeg(
                option_type="put",
                position=1,
                strike=self.put_strike,
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
        """Evaluate Protective Put pricing, tail-risk insurance metrics, and portfolio Greeks.

        Args:
            bs_engine: BSEngine instance for pricing the long put leg.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            dict: Complete dictionary of strategy metrics and portfolio Greeks.
        """
        put_price = float(
            bs_engine.put_price(
                S=self.spot_price,
                K=self.put_strike,
                T=self.maturity,
                r=r,
                q=q,
                sigma=sigma,
            )
        )

        premium_paid_twd = float(put_price * self.underlying_shares)
        hedging_cost_annual = float(
            (put_price / self.spot_price) * (365.0 / self.maturity_days)
        )
        max_loss_pct = float(
            (self.spot_price - self.put_strike + put_price) / self.spot_price
        )
        floor_price = float(self.put_strike)
        breakeven_price = float(self.spot_price + put_price)

        greeks = self.calculate_portfolio_greeks(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

        return {
            "strategy_name": "Protective Put",
            "spot_price": self.spot_price,
            "put_strike": self.put_strike,
            "maturity_days": self.maturity_days,
            "put_price": put_price,
            "premium_paid_twd": premium_paid_twd,
            "hedging_cost_annual": hedging_cost_annual,
            "max_loss_pct": max_loss_pct,
            "floor_price": floor_price,
            "breakeven_price": breakeven_price,
            "portfolio_greeks": greeks,
            **greeks,
        }
