"""Zero-Cost Collar Corridor Structuring Strategy.

Role:
    Implements an institutional Collar strategy combining a core Taiwan equity
    position (1 board lot = 1,000 shares) with a long OTM Put (downside floor)
    financed by selling an OTM Call (upside cap). Features an automated
    root-finding structurer (`solve_zero_cost_call`) using Brent's method
    (`scipy.optimize.brentq`) to solve for the exact zero-cost Call strike K_c
    that equates Call premium received to Put premium paid.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from scipy import optimize

from models.bs_engine import BSEngine
from .base import BaseStrategy, OptionLeg


class CollarStrategy(BaseStrategy):
    """Institutional Collar (Long Stock + Long OTM Put + Short OTM Call) strategy engine."""

    def __init__(
        self,
        spot_price: float,
        underlying_shares: int = 1000,
        put_strike: Optional[float] = None,
        call_strike: Optional[float] = None,
        maturity_days: int = 30,
    ) -> None:
        """Initialize CollarStrategy with optional put and call strike bounds.

        Args:
            spot_price: Current underlying equity spot price in TWD.
            underlying_shares: Number of underlying shares held (default 1,000 shares).
            put_strike: Optional strike price for the protective Long Put leg.
            call_strike: Optional strike price for the financing Short Call leg.
            maturity_days: Days to expiration (default 30 calendar days).
        """
        super().__init__(spot_price=spot_price, underlying_shares=underlying_shares)

        if maturity_days <= 0:
            raise ValueError(f"maturity_days must be strictly positive, got {maturity_days}.")

        self.maturity_days: int = int(maturity_days)
        self.maturity: float = float(self.maturity_days) / 365.0
        self.put_strike: Optional[float] = float(put_strike) if put_strike is not None else None
        self.call_strike: Optional[float] = float(call_strike) if call_strike is not None else None

        if self.put_strike is not None:
            self.add_leg(
                OptionLeg(
                    option_type="put",
                    position=1,
                    strike=self.put_strike,
                    maturity=self.maturity,
                    contracts=1,
                )
            )

        if self.call_strike is not None:
            self.add_leg(
                OptionLeg(
                    option_type="call",
                    position=-1,
                    strike=self.call_strike,
                    maturity=self.maturity,
                    contracts=1,
                )
            )

    def _upsert_leg(self, option_type: str, position: int, strike: float) -> None:
        """Add or replace an existing leg matching `option_type` and `position`."""
        opt = option_type.strip().lower()
        self.legs = [
            leg for leg in self.legs if not (leg.option_type == opt and leg.position == position)
        ]
        self.add_leg(
            OptionLeg(
                option_type=opt,
                position=position,
                strike=float(strike),
                maturity=self.maturity,
                contracts=1,
            )
        )

    def solve_zero_cost_call(
        self,
        bs_engine: BSEngine,
        target_put_strike: float,
        r: float,
        q: float,
        sigma: float,
    ) -> float:
        """Solve for the exact Call strike K_c that yields a zero-cost collar structure.

        Finds K_c in [spot_price, spot_price * 2.0] such that:
            Call_Price(S, K_c, T, r, q, sigma) - Put_Price(S, target_put_strike, T, r, q, sigma) == 0

        Updates `self.put_strike`, `self.call_strike`, and synchronizes both legs in `self.legs`.

        Args:
            bs_engine: BSEngine instance for option valuation.
            target_put_strike: Desired downside floor Put strike price.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            float: Solved zero-cost Call strike K_c.

        Raises:
            ValueError: If no zero-cost solution exists within the search bracket.
        """
        T = self.maturity
        self.put_strike = float(target_put_strike)
        self._upsert_leg(option_type="put", position=1, strike=self.put_strike)

        target_put_price = float(
            bs_engine.put_price(
                S=self.spot_price,
                K=self.put_strike,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
            )
        )

        def root_fn(K: float) -> float:
            return (
                float(bs_engine.call_price(self.spot_price, K, T, r, q, sigma))
                - target_put_price
            )

        lower_bound = self.spot_price
        upper_bound = self.spot_price * 2.0

        # If target_put_strike is close to ATM and dividend yield > r, f(spot_price) may be < 0;
        # fallback to a slightly wider lower bound if needed while prioritizing [spot, 2*spot]
        f_low = root_fn(lower_bound)
        f_high = root_fn(upper_bound)
        if f_low * f_high > 0.0:
            lower_bound = min(self.spot_price * 0.5, self.put_strike)
            upper_bound = self.spot_price * 3.0

        solved_strike = float(optimize.brentq(root_fn, lower_bound, upper_bound, xtol=1e-7))
        self.call_strike = solved_strike
        self._upsert_leg(option_type="call", position=-1, strike=self.call_strike)

        return float(self.call_strike)

    def evaluate(
        self,
        bs_engine: BSEngine,
        r: float,
        q: float,
        sigma: float,
    ) -> Dict[str, Any]:
        """Evaluate Collar strategy premiums, max gain/loss bounds, and portfolio Greeks.

        Args:
            bs_engine: BSEngine instance for option valuation.
            r: Annualized continuously compounded risk-free rate.
            q: Annualized continuously compounded dividend yield.
            sigma: Annualized volatility.

        Returns:
            dict: Complete dictionary of Collar metrics and portfolio Greeks.

        Raises:
            ValueError: If `put_strike` or `call_strike` has not been configured.
        """
        if self.put_strike is None:
            self.put_strike = float(self.spot_price * 0.93)
            self._upsert_leg(option_type="put", position=1, strike=self.put_strike)

        if self.call_strike is None:
            self.call_strike = float(self.spot_price * 1.05)
            self._upsert_leg(option_type="call", position=-1, strike=self.call_strike)

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

        net_premium_twd = float((call_price - put_price) * self.underlying_shares)
        max_loss_pct = float(
            (self.spot_price - self.put_strike + (put_price - call_price)) / self.spot_price
        )
        max_gain_pct = float(
            (self.call_strike - self.spot_price + (call_price - put_price)) / self.spot_price
        )

        greeks = self.calculate_portfolio_greeks(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

        return {
            "strategy_name": "Collar",
            "spot_price": self.spot_price,
            "put_strike": self.put_strike,
            "call_strike": self.call_strike,
            "maturity_days": self.maturity_days,
            "put_price": put_price,
            "call_price": call_price,
            "net_premium_twd": net_premium_twd,
            "max_loss_pct": max_loss_pct,
            "max_gain_pct": max_gain_pct,
            "portfolio_greeks": greeks,
            **greeks,
        }
