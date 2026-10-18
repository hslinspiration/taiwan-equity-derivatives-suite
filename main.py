"""Taiwan Equity Derivatives Suite - Institutional CLI & Structuring Pipeline.

Role:
    Master command-line entry point orchestrating the complete quantitative workflow:
    1. Live Taiwan market data ingestion (TWSE adjusted close history, CBC discount
       rate, and trailing 12-month cash dividend yield) with offline resilience.
    2. Annualized Historical Volatility (HV) estimation.
    3. Vectorized Black-Scholes-Merton pricing and analytical Greeks calculation.
    4. Institutional overlay structuring (Covered Call, Protective Put, and
       Brent-solved Zero-Cost Collar).
    5. OTC Trading Desk quote calibration, negotiation slack quantification, and
       initial delta-hedging share computation.
    6. Terminal ASCII Executive Dashboard and 300-DPI visualization generation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

import config
from data.data_fetcher import MarketDataFetcher
from hedging.delta_hedger import DeltaHedger, DeskPosition
from models.bs_engine import BSEngine
from models.desk_calibrator import DeskCalibrator, DeskMarketParams
from models.greeks import GreeksCalculator
from models.volatility import HistoricalVolatility
from strategies.collar import CollarStrategy
from strategies.covered_call import CoveredCallStrategy
from strategies.protective_put import ProtectivePutStrategy
from visualization.plots import DerivativesPlotter

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the Taiwan Equity Derivatives structuring pipeline."""
    parser = argparse.ArgumentParser(
        description="Taiwan Equity Derivatives Suite - Institutional Structuring & Desk Analytics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=config.DEFAULT_TICKER,
        help="Target Taiwan equity ticker (e.g., '2330.TW' for TSMC)",
    )
    parser.add_argument(
        "--shares",
        type=int,
        default=1000,
        help="Underlying shares to structure (1 Taiwan board lot = 1,000 shares; e.g., 1000000 for 1,000 lots)",
    )
    parser.add_argument(
        "--maturity",
        type=int,
        default=30,
        help="Option time to maturity in calendar days",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=config.DEFAULT_HV_WINDOW,
        help="Lookback window in trading days for Historical Volatility (HV)",
    )
    parser.add_argument(
        "--cc-otm",
        type=float,
        default=1.05,
        help="Call strike moneyness multiplier for Covered Call (e.g., 1.05 for 5%% OTM)",
    )
    parser.add_argument(
        "--put-otm",
        type=float,
        default=0.93,
        help="Put strike moneyness multiplier for Protective Put and Collar (e.g., 0.93 for 7%% OTM)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory path to save generated visualization plots",
    )
    return parser.parse_args()


def _load_market_parameters(ticker: str, window: int) -> Dict[str, float]:
    """Fetch live market data and compute annualized HV, with offline fallback resilience."""
    fetcher = MarketDataFetcher(ticker=ticker)
    r = fetcher.fetch_taiwan_risk_free_rate(fallback_rate=config.FALLBACK_RISK_FREE_RATE)

    try:
        hist_df = fetcher.fetch_history()
        spot_price = fetcher.get_latest_price()
        div_yield = fetcher.calculate_dividend_yield()
        hv_engine = HistoricalVolatility(
            price_series=hist_df["Adj Close"],
            trading_days=config.TRADING_DAYS_PER_YEAR,
        )
        sigma = hv_engine.close_to_close(window=window)
    except Exception as exc:
        logger.warning(
            "Live market data fetch unavailable (%s). Using deterministic TSMC benchmark series.",
            exc,
        )
        rng = np.random.default_rng(seed=2330)
        n_days = max(window + 40, 120)
        daily_rets = rng.normal(loc=0.0006, scale=0.0175, size=n_days)
        synthetic_prices = pd.Series(1000.0 * np.exp(np.cumsum(daily_rets)))
        spot_price = float(synthetic_prices.iloc[-1])
        div_yield = 0.018
        hv_engine = HistoricalVolatility(
            price_series=synthetic_prices,
            trading_days=config.TRADING_DAYS_PER_YEAR,
        )
        sigma = hv_engine.close_to_close(window=window)

    return {
        "spot_price": float(spot_price),
        "risk_free_rate": float(r),
        "dividend_yield": float(div_yield),
        "volatility": float(sigma),
    }


def _print_ascii_dashboard(
    ticker: str,
    shares: int,
    maturity_days: int,
    window: int,
    market: Dict[str, float],
    cc_eval: Dict[str, object],
    pp_eval: Dict[str, object],
    collar_eval: Dict[str, object],
    desk_analysis: Dict[str, Dict[str, float]],
) -> None:
    """Render formatted institutional ASCII dashboard tables to terminal stdout."""
    s0 = market["spot_price"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]
    sigma = market["volatility"]
    lots = shares / 1000.0

    sep = "=" * 96
    sub_sep = "-" * 96

    print(f"\n{sep}")
    print(" TAIWAN EQUITY DERIVATIVES SUITE - INSTITUTIONAL STRUCTURING & DESK ANALYTICS DASHBOARD")
    print(sep)

    # Table 1: Market & Underlying Diagnostics
    print("\n[TABLE 1] MARKET & UNDERLYING DIAGNOSTICS")
    print(sub_sep)
    print(f"  {'Metric':<36} | {'Value':<24} | {'Institutional Convention'}")
    print(sub_sep)
    print(f"  {'Underlying Ticker':<36} | {ticker:<24} | {'TWSE Listed Equity'}")
    print(f"  {'Position Size':<36} | {f'{shares:,} shs ({lots:,.1f} lots)':<24} | {'1 Board Lot = 1,000 Shares'}")
    print(f"  {'Latest Adjusted Spot (S0)':<36} | {f'NT${s0:,.2f}':<24} | {'Ex-Dividend Adjusted Close'}")
    print(f"  {'Taiwan CBC Risk-Free Rate (r)':<36} | {f'{r * 100:.2f}%':<24} | {'Central Bank Rediscount / CP'}")
    print(f"  {'Trailing 12M Dividend Yield (q)':<36} | {f'{q * 100:.2f}%':<24} | {'Continuous Merton (1973) Yield'}")
    print(f"  {f'Annualized {window}D Historical Vol (HV)':<36} | {f'{sigma * 100:.2f}%':<24} | {'252 Trading Days Convention'}")
    print(sub_sep)

    # Table 2: Institutional Derivatives Structuring Matrix
    print("\n[TABLE 2] INSTITUTIONAL DERIVATIVES STRUCTURING MATRIX " f"(Tenor: {maturity_days} Calendar Days)")
    print(sub_sep)
    header_t2 = (
        f"  {'Strategy Overlay':<20} | {'Strikes (TWD)':<20} | {'Net Premium (TWD)':<18} | "
        f"{'Ann. Yield/Cost':<15} | {'Breakeven / Floor'}"
    )
    print(header_t2)
    print(sub_sep)

    cc_strike_str = f"C: {float(cc_eval['call_strike']):,.1f}"
    cc_prem_str = f"+NT${float(cc_eval['premium_received_twd']):,.0f}"
    cc_yield_str = f"+{float(cc_eval['yield_enhancement_annual']) * 100:.2f}% (Yield)"
    cc_be_str = f"BE: NT${float(cc_eval['breakeven_price']):,.1f} (-{float(cc_eval['downside_buffer_pct']) * 100:.2f}%)"
    print(f"  {'Covered Call':<20} | {cc_strike_str:<20} | {cc_prem_str:<18} | {cc_yield_str:<15} | {cc_be_str}")

    pp_strike_str = f"P: {float(pp_eval['put_strike']):,.1f}"
    pp_prem_str = f"-NT${float(pp_eval['premium_paid_twd']):,.0f}"
    pp_cost_str = f"-{float(pp_eval['hedging_cost_annual']) * 100:.2f}% (Cost)"
    pp_be_str = f"Floor: NT${float(pp_eval['floor_price']):,.1f} | BE: NT${float(pp_eval['breakeven_price']):,.1f}"
    print(f"  {'Protective Put':<20} | {pp_strike_str:<20} | {pp_prem_str:<18} | {pp_cost_str:<15} | {pp_be_str}")

    col_strike_str = f"P:{float(collar_eval['put_strike']):,.0f} / C:{float(collar_eval['call_strike']):,.0f}"
    col_prem_str = f"NT${float(collar_eval['net_premium_twd']):,.0f}"
    col_cost_str = "0.00% (Zero-Cost)"
    col_bounds_str = (
        f"MaxLoss: -{float(collar_eval['max_loss_pct']) * 100:.2f}% | "
        f"MaxGain: +{float(collar_eval['max_gain_pct']) * 100:.2f}%"
    )
    print(f"  {'Zero-Cost Collar':<20} | {col_strike_str:<20} | {col_prem_str:<18} | {col_cost_str:<15} | {col_bounds_str}")
    print(sub_sep)

    # Table 3: Trading Desk Risk & Negotiation Slack
    print("\n[TABLE 3] TRADING DESK RISK CALIBRATION & CLIENT NEGOTIATION SLACK")
    print(sub_sep)
    header_t3 = (
        f"  {'Desk Leg':<22} | {'Skew Vol':<10} | {'Desk Floor (TWD)':<18} | "
        f"{'Client Quote (TWD)':<18} | {'Neg. Slack (TWD)':<16} | {'Init Hedge'}"
    )
    print(header_t3)
    print(sub_sep)
    for leg_label, d_info in desk_analysis.items():
        skew_v = f"{d_info['skew_adjusted_vol'] * 100:.2f}%"
        d_floor = f"NT${d_info['desk_breakeven_twd']:,.0f}"
        c_quote = f"NT${d_info['client_quote_twd']:,.0f}"
        slack = f"NT${d_info['negotiation_slack_twd']:,.0f}"
        hedge_sh = f"{int(d_info['initial_hedge_shares']):+,d} shs"
        print(
            f"  {leg_label:<22} | {skew_v:<10} | {d_floor:<18} | "
            f"{c_quote:<18} | {slack:<16} | {hedge_sh}"
        )
    print(sep)


def main() -> int:
    """Execute the end-to-end Taiwan Equity Derivatives structuring and visualization pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1 & 2. Market Data & Historical Volatility
    market = _load_market_parameters(ticker=args.ticker, window=args.window)
    s0 = market["spot_price"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]
    sigma = market["volatility"]
    maturity_years = float(args.maturity) / 365.0
    contracts = max(int(round(args.shares / 1000)), 1)

    # 4. Initialize Core Pricing, Greek & Desk Engines
    bs_engine = BSEngine()
    greeks_calc = GreeksCalculator()
    desk_calibrator = DeskCalibrator(bs_engine=bs_engine, params=DeskMarketParams())
    delta_hedger = DeltaHedger(bs_engine=bs_engine, greeks_calc=greeks_calc)

    # 5. Structure Institutional Strategies
    cc_strike = float(s0 * args.cc_otm)
    put_strike = float(s0 * args.put_otm)

    cc_strategy = CoveredCallStrategy(
        spot_price=s0,
        underlying_shares=args.shares,
        call_strike=cc_strike,
        maturity_days=args.maturity,
    )
    cc_eval = cc_strategy.evaluate(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

    pp_strategy = ProtectivePutStrategy(
        spot_price=s0,
        underlying_shares=args.shares,
        put_strike=put_strike,
        maturity_days=args.maturity,
    )
    pp_eval = pp_strategy.evaluate(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

    collar_strategy = CollarStrategy(
        spot_price=s0,
        underlying_shares=args.shares,
        put_strike=put_strike,
        maturity_days=args.maturity,
    )
    solved_collar_call = collar_strategy.solve_zero_cost_call(
        bs_engine=bs_engine,
        target_put_strike=put_strike,
        r=r,
        q=q,
        sigma=sigma,
    )
    collar_eval = collar_strategy.evaluate(bs_engine=bs_engine, r=r, q=q, sigma=sigma)

    # 6. Trading Desk Counterparty Analysis
    # Client Short Call -> Desk Long Call (+1); Client Long Put -> Desk Short Put (-1)
    cc_desk_quotes = desk_calibrator.calculate_desk_quotes(
        S=s0, K=cc_strike, T=maturity_years, r=r, q=q, base_sigma=sigma, option_type="call"
    )
    cc_desk_pos = DeskPosition(
        option_type="call",
        position=1,
        strike=cc_strike,
        maturity_years=maturity_years,
        contracts=contracts,
        entry_premium=cc_desk_quotes["client_quote"],
    )
    cc_hedge = delta_hedger.compute_initial_hedge(
        spot_price=s0, desk_position=cc_desk_pos, r=r, q=q, sigma=sigma
    )

    pp_desk_quotes = desk_calibrator.calculate_desk_quotes(
        S=s0, K=put_strike, T=maturity_years, r=r, q=q, base_sigma=sigma, option_type="put"
    )
    pp_desk_pos = DeskPosition(
        option_type="put",
        position=-1,
        strike=put_strike,
        maturity_years=maturity_years,
        contracts=contracts,
        entry_premium=pp_desk_quotes["client_quote"],
    )
    pp_hedge = delta_hedger.compute_initial_hedge(
        spot_price=s0, desk_position=pp_desk_pos, r=r, q=q, sigma=sigma
    )

    col_call_quotes = desk_calibrator.calculate_desk_quotes(
        S=s0, K=solved_collar_call, T=maturity_years, r=r, q=q, base_sigma=sigma, option_type="call"
    )
    col_call_pos = DeskPosition(
        option_type="call",
        position=1,
        strike=solved_collar_call,
        maturity_years=maturity_years,
        contracts=contracts,
        entry_premium=col_call_quotes["client_quote"],
    )
    col_call_hedge = delta_hedger.compute_initial_hedge(
        spot_price=s0, desk_position=col_call_pos, r=r, q=q, sigma=sigma
    )

    desk_analysis = {
        "Covered Call (Leg)": {
            "skew_adjusted_vol": cc_desk_quotes["skew_adjusted_vol"],
            "desk_breakeven_twd": cc_desk_quotes["desk_breakeven"] * args.shares,
            "client_quote_twd": cc_desk_quotes["client_quote"] * args.shares,
            "negotiation_slack_twd": cc_desk_quotes["negotiation_slack"] * args.shares,
            "initial_hedge_shares": float(cc_hedge["shares_to_hedge"]),
        },
        "Protective Put (Leg)": {
            "skew_adjusted_vol": pp_desk_quotes["skew_adjusted_vol"],
            "desk_breakeven_twd": pp_desk_quotes["desk_breakeven"] * args.shares,
            "client_quote_twd": pp_desk_quotes["client_quote"] * args.shares,
            "negotiation_slack_twd": pp_desk_quotes["negotiation_slack"] * args.shares,
            "initial_hedge_shares": float(pp_hedge["shares_to_hedge"]),
        },
        "Collar Net Package": {
            "skew_adjusted_vol": 0.5
            * (pp_desk_quotes["skew_adjusted_vol"] + col_call_quotes["skew_adjusted_vol"]),
            "desk_breakeven_twd": (
                pp_desk_quotes["desk_breakeven"] + col_call_quotes["desk_breakeven"]
            )
            * args.shares,
            "client_quote_twd": (
                pp_desk_quotes["client_quote"] + col_call_quotes["client_quote"]
            )
            * args.shares,
            "negotiation_slack_twd": (
                pp_desk_quotes["negotiation_slack"] + col_call_quotes["negotiation_slack"]
            )
            * args.shares,
            "initial_hedge_shares": float(
                int(pp_hedge["shares_to_hedge"]) + int(col_call_hedge["shares_to_hedge"])
            ),
        },
    }

    # 7. Print Formatted ASCII Dashboard
    _print_ascii_dashboard(
        ticker=args.ticker,
        shares=args.shares,
        maturity_days=args.maturity,
        window=args.window,
        market=market,
        cc_eval=cc_eval,
        pp_eval=pp_eval,
        collar_eval=collar_eval,
        desk_analysis=desk_analysis,
    )

    # 8. Generate and Save Plots
    plotter = DerivativesPlotter(output_dir=args.output_dir)
    spot_grid = np.linspace(s0 * 0.80, s0 * 1.20, 200)

    stock_payoff = (spot_grid - s0) * float(args.shares)
    cc_net_pnl = cc_strategy.calculate_payoff(spot_grid) + float(cc_eval["premium_received_twd"])
    pp_net_pnl = pp_strategy.calculate_payoff(spot_grid) - float(pp_eval["premium_paid_twd"])
    collar_net_pnl = collar_strategy.calculate_payoff(spot_grid) + float(
        collar_eval["net_premium_twd"]
    )

    pnl_path = plotter.plot_multi_strategy_pnl(
        spot_grid=spot_grid,
        payoffs={
            "Underlying Stock": stock_payoff,
            "Covered Call": cc_net_pnl,
            "Protective Put": pp_net_pnl,
            "Zero-Cost Collar": collar_net_pnl,
        },
        breakevens={
            "Covered Call": float(cc_eval["breakeven_price"]),
            "Protective Put": float(pp_eval["breakeven_price"]),
            "Zero-Cost Collar": s0,
        },
        save_path=os.path.join(args.output_dir, "pnl_comparison.png"),
    )

    # Covered Call Interim PnL & Theta Decay curves across 3 horizons
    initial_call_prem = float(cc_eval["call_price"])
    t_full = float(args.maturity) / 365.0
    t_mid = (0.5 * float(args.maturity)) / 365.0

    mtm_inception = (
        (spot_grid - s0)
        + (initial_call_prem - bs_engine.call_price(spot_grid, cc_strike, t_full, r, q, sigma))
    ) * float(args.shares)
    mtm_midterm = (
        (spot_grid - s0)
        + (initial_call_prem - bs_engine.call_price(spot_grid, cc_strike, t_mid, r, q, sigma))
    ) * float(args.shares)
    mtm_expiry = (
        (spot_grid - s0)
        + (initial_call_prem - bs_engine.call_price(spot_grid, cc_strike, 0.0, r, q, sigma))
    ) * float(args.shares)

    theta_path = plotter.plot_theta_decay(
        spot_grid=spot_grid,
        decay_curves={
            "T=30 Days (Inception)": np.asarray(mtm_inception, dtype=float),
            "T=15 Days (Mid-term)": np.asarray(mtm_midterm, dtype=float),
            "T=0 Days (Expiration Payoff)": np.asarray(mtm_expiry, dtype=float),
        },
        save_path=os.path.join(args.output_dir, "theta_decay.png"),
    )

    # 3D Gamma Surface across Spot and Time to Expiration (Pin Risk demonstration)
    surf_spots = np.linspace(s0 * 0.85, s0 * 1.15, 60)
    surf_days = np.linspace(1.0, float(max(args.maturity, 5)), 40)
    s_mesh, d_mesh = np.meshgrid(surf_spots, surf_days)
    t_mesh_years = d_mesh / 365.0
    gamma_surf = np.asarray(
        greeks_calc.gamma(S=s_mesh, K=cc_strike, T=t_mesh_years, r=r, q=q, sigma=sigma),
        dtype=float,
    )

    surface_path = plotter.plot_greeks_surface(
        spot_range=s_mesh,
        time_range=d_mesh,
        gamma_surface=gamma_surf,
        strike=cc_strike,
        save_path=os.path.join(args.output_dir, "greeks_surface.png"),
    )

    print("\n[VISUALIZATION ARTIFACTS SAVED]")
    print(f"  1. Multi-Strategy PnL Comparison : {pnl_path}")
    print(f"  2. Covered Call Theta Decay      : {theta_path}")
    print(f"  3. 3D Gamma Pin-Risk Surface     : {surface_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
