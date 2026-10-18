# 🇹🇼 Taiwan Equity Derivatives & Structuring Suite
### Institutional Pricing, Structuring Engine & Desk Risk Management Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-vectorized-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-optimization-orange.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Core Positioning**: A production-grade quantitative toolkit tailored for Taiwan sell-side Equity Derivatives Desks, Institutional Sales Advisors, and Corporate Structurers handling large-cap, high-dividend Taiwan equities (e.g., TSMC 2330.TW, MediaTek 2454.TW). The system bridges empirical market microstructure, stochastic calculus pricing, volatility skew and borrowing cost calibration, institutional derivative structuring, and trading desk Delta-neutral dynamic hedging with PnL attribution.

---

## Table of Contents
1. [System Overview & Commercial Utility](#1-system-overview--commercial-utility)
2. [Architecture & Mathematical Formulations](#2-architecture--mathematical-formulations)
3. [Institutional Structuring Playbook](#3-institutional-structuring-playbook)
4. [Visual Diagnostics Manual](#4-visual-diagnostics-manual)
5. [Quickstart & Pytest Verification](#5-quickstart--pytest-verification)
6. [Author & Disclaimer](#6-author--disclaimer)

---

## 1. System Overview & Commercial Utility

### Executive Summary
In Taiwan's equity capital markets, corporate insiders, institutional asset managers, and family offices hold multi-billion TWD equity portfolios concentrated in blue-chip, high-dividend equities (such as TSMC `2330.TW` and MediaTek `2454.TW`). However, due to regulatory insider holding disclosures, collateral maintenance ratios, and corporate governance mandates, outright equity liquidation is often legally restricted or commercially unviable.

Standard academic option pricing models assume frictionless markets, flat implied volatility, and zero stock borrow costs, creating a severe operational disconnect from sell-side trading desk execution. This suite delivers an **end-to-end quantitative pipeline** engineered to bridge theoretical financial engineering with executable desk pricing:

```text
[ Market Data & Macro Rates ] ──▶ [ Annualized HV & Dividend Imputation ] ──▶ [ Vectorized BSM & Friction Calibration ]
                                                                                               │
                                                                                               ▼
[ Desk Delta-Neutral Hedging & PnL ] ◀── [ Executive Dashboard & Plots ] ◀── [ Institutional Structuring Assembly ]
```

### End-to-End Workflow

1. **Dynamic Market Ingestion & Data Hygiene**: Automatically ingests historical equity OHLCV bars, strictly enforces the use of **Adjusted Close** to eliminate artificial ex-dividend price drops, and dynamically extracts trailing-twelve-month (TTM) cash dividend yields alongside Taiwan Central Bank (CBC) rediscount rate open data APIs with graceful fallback.
2. **Volatility & Pricing Matrix**: Computes the sell-side standard 60-day historical volatility (HV) benchmark, executes fully vectorized Black-Scholes-Merton calculations, and incorporates moneyness-adjusted volatility skew and TWSE Securities Borrowing and Lending (SBL / 借券) costs.
3. **Institutional Structuring**: Assembles and evaluates **Covered Call** (Yield Enhancement), **Protective Put** (Downside Floor), and **Zero-Cost Collar** structures with automated numerical root-finding.
4. **Market-Maker Desk Simulation**: Adopts the trading desk counterparty perspective to calculate initial Delta-neutral stock hedge ratios, quantifies institutional negotiation slack, and executes discrete daily delta rebalancing with 2nd-order Taylor expansion PnL attribution.

### Commercial Outcomes

* **Corporate Equity Monetization (Yield Enhancement)**: Empowers corporate treasury desks and long-only funds to monetize stagnant inventory shares by capturing extrinsic time value (Theta) via out-of-the-money (OTM) Covered Calls, generating an annualized **+3.0% to +8.0% incremental cash yield** without selling underlying inventory.
* **Costless Downside Tail-Risk Protection**: Deploys continuous root-finding algorithms ahead of earnings releases, geopolitical risk events, or macro rate decisions to automatically solve for zero-premium Collar bounds, locking in a hard floor with **Zero Net Cash Outflow**.
* **Narrowing the Theory vs. Execution Gap**: By explicitly parameterizing SBL stock borrow fees (2.0%), downside volatility skew ($\beta = 0.45$), Taiwan statutory securities transaction taxes (0.3%), and brokerage commissions (0.1425%), the suite aligns theoretical midpoints with real-world desk execution, enhancing structuring credibility.

---

## 2. Architecture & Mathematical Formulations

### Design Philosophy

* **Decoupled Pipeline Architecture**: Ingestion (`data/`), quantitative models (`models/`), institutional strategies (`strategies/`), risk management (`hedging/`), and rendering (`visualization/`) are strictly separated. Connecting live exchange feeds (e.g., Bloomberg B-PIPE or TAIFEX native feeds) requires zero refactoring of upstream strategy layers.
* **Full NumPy Vectorization**: Eliminates iterative Python loops with vectorized array broadcasting, evaluating multi-strike spot grids ($200+$ points) and multi-maturity sensitivity manifolds in sub-millisecond execution times.
* **Linear Superposition Principle**: Deconstructs multi-leg structures into atomic `OptionLeg` primitives. Aggregate portfolio Greeks and payoffs are resolved via algebraic linear addition:

$$\text{Portfolio Greek} = \sum_{i} \text{Position}_i \times \text{Contracts}_i \times \text{Multiplier}_i \times \text{Greek}_i$$

---

### Mathematical Formulations

#### A. Vectorized BSM with Merton (1973) Dividend Discount (`models/bs_engine.py`)

To capture the continuous dividend yield $q$ characteristic of major TWSE dividend-paying stocks, the pricing engine implements the closed-form Merton (1973) continuous dividend yield extension:

$$d_1 = \frac{\ln(S / K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$\text{Call} = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$\text{Put} = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$\text{Put-Call Parity: } \text{Call} - \text{Put} = S e^{-qT} - K e^{-rT}$$

* **Singularity & Boundary Defense**: When $T \le 0$ (at expiration) or $\sigma \le 0$, normal CDF denominators collapse into `ZeroDivisionError`. The engine enforces asymptotic fallback to intrinsic payoffs:

$$\lim_{T \to 0} \text{Call} = \max(S - K, 0), \quad \lim_{T \to 0} \text{Put} = \max(K - S, 0)$$

$$\lim_{\sigma \to 0} \text{Call} = \max(S e^{-qT} - K e^{-rT}, 0), \quad \lim_{\sigma \to 0} \text{Put} = \max(K e^{-rT} - S e^{-qT}, 0)$$

#### B. Standardized Analytical Greeks (`models/greeks.py`)

Calculates closed-form partial derivatives and scales them to institutional trading conventions:

| Greek | Analytical Closed-Form Formulation | Standardized Trading Unit | Financial & Risk Intuition |
| :--- | :--- | :--- | :--- |
| **Delta ($\Delta$)** | $\text{Call: } e^{-qT} N(d_1)$<br>$\text{Put: } -e^{-qT} N(-d_1)$ | Ratio $[0, 1]$ or $[-1, 0]$ | Replicating hedge ratio; asset-weighted exercise probability. |
| **Gamma ($\Gamma$)** | $\frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt{T}}$ | Identical for Call and Put | Curvature/convexity; rate of change of Delta per NT$1 spot move. |
| **Vega ($\mathcal{V}$)** | $0.01 \times S e^{-qT} \sqrt{T} \phi(d_1)$ | $\mathcal{V} \div 100$ | Dollar price change per **1 percentage point (100 bps)** shift in volatility. |
| **Theta ($\Theta$)** | $\text{Daily } \Theta = \text{Annual } \Theta_{\text{BSM}} \div 365$ | $\Theta \div 365$ | Extrinsic dollar value decayed per **1 calendar day**. |
| **Rho ($\rho$)** | $\text{Call: } 0.01 \times K T e^{-rT} N(d_2)$<br>$\text{Put: } -0.01 \times K T e^{-rT} N(-d_2)$ | $\rho \div 100$ | Dollar price sensitivity per **1 percentage point (100 bps)** interest rate shift. |

* **Gamma-Theta Trade-off (Black-Scholes PDE)**:
Under a Delta-neutral portfolio ($\Delta = 0, q = 0$), the Black-Scholes PDE reveals the foundational market-making invariance principle:

$$\Theta = \frac{1}{2}\sigma^2 S^2 \Gamma - rV$$

To collect positive time-decay rent ($\Theta > 0$), the trading desk must absorb negative convexity risk ($\Gamma < 0$) against market jump shocks.

#### C. Microstructural Friction Calibration (`models/desk_calibrator.py`)

To align theoretical midpoint prices with institutional execution reality:

1. **Moneyness-Adjusted Volatility Skew**:
Reflects structural institutional demand where downside put protection commands higher implied volatility:

$$\sigma(K) = \sigma_{\text{ATM}} \times \left(1 - \beta \cdot \ln\left(\frac{K}{S}\right)\right), \quad \text{Default Taiwan Skew Slope } \beta = 0.45$$

2. **Short-Hedge Stock Borrowing Cost ($b$)**:
When a client executes a Protective Put (Client Buys Put $\to$ Desk Sells Put), the desk must short underlying stock to delta-hedge. Incorporating annual TWSE SBL borrow cost $b$ (default 2.0%) modifies the effective carry:

$$q_{\text{eff}} = q + b$$

3. **Dynamic Rebalancing Friction**: Embeds a 3.0% buffer against Taiwan's 0.3% securities transaction tax (`STOCK_TRANSACTION_TAX`) and 0.1425% brokerage fees (`BROKERAGE_FEE_RATE`) incurred during dynamic gamma rebalancing.

#### D. Numerical Root-Finding for Zero-Cost Collars (`strategies/collar.py`)

A costless Collar requires solving for the exact Call strike $K_{\text{Call}}$ such that:

$$f(K_{\text{Call}}) = \text{BSM}_{\text{Call}}(S, K_{\text{Call}}, T, r, q, \sigma(K_{\text{Call}})) - \text{Price}_{\text{Put target}} = 0$$

Because $K_{\text{Call}}$ is embedded inside non-invertible cumulative normal integrals, analytical inversion is impossible. The engine applies **Brent's Method (`scipy.optimize.brentq`)** over search bounds $[S, 2S]$, converging on a root tolerance of $< 10^{-8}$ within milliseconds.

#### E. 2nd-Order Taylor PnL Attribution (`hedging/delta_hedger.py`)

Simulates discrete daily delta rebalancing, decomposing realized desk PnL into fundamental risk drivers:

$$\Delta \Pi \approx \underbrace{(\Delta_{\text{opt}} \cdot \Delta S + N_{\text{stock}} \cdot \Delta S)}_{\text{Delta Residual (Directional Leakage)}} + \underbrace{\frac{1}{2}\Gamma_{\text{opt}} \cdot (\Delta S)^2}_{\text{Gamma Convexity Drag / Gain}} + \underbrace{\Theta_{\text{opt}} \cdot \Delta t}_{\text{Theta Decay PnL}} - \text{Friction Costs}$$

---

## 3. Institutional Structuring Playbook

**Base Scenario**: Underlying: **TSMC (2330.TW)** | Spot Price $S_0 = \text{NT}\$2,400$ | Notional: **1,000 Lots (1,000,000 Shares / NT$2.4 Billion Market Value)** | Maturity: **30 Days** | Base HV: **24.50%**:

```text
================================================================================================
【Taiwan Equity Derivatives Executive Dashboard】 2330.TW | Spot: NT$2,400.0 | Base HV: 24.50%
================================================================================================
Strategy Structure  Contract Spec            Net Cash Premium  Ann. Yield / Cost Downside Buffer Max Upside
------------------------------------------------------------------------------------------------
Underlying Stock    Long 1,000 Lots          NT$ 0.00          0.00%             None (Full Risk) Unlimited
Covered Call        Short 1M K=2520 C        +NT$ 15.2M (Recv) +7.70% (Yield)    -0.63% (Buffer)  +5.63%
Protective Put      Long 1M K=2232 P         -NT$ 14.5M (Paid) -7.35% (Cost)     Capped at -7.60% Unlimited
Zero-Cost Collar    Long P2232 / Short C2541 NT$ 0.00          0.00% (Costless)  Capped at -7.00% +5.87%
================================================================================================
[Desk Hedging Perspective] Initial Delta-Neutral Stock Short Hedge: 285 Lots (285,000 Shares)
================================================================================================
```

### Strategy Deep-Dives

#### 1. Covered Call (Corporate Yield Enhancement)
* **Position**: Long 1,000 Lots Stock + Short 1,000 Contracts OTM 5% Call ($K = \text{NT}\$2,520$).
* **Mechanism**: The shareholder forfeits capital appreciation above NT$2,520 in exchange for an immediate cash premium of NT$15.2 Million.
* **Payoff Profile**: Generates an annualized **+7.70% yield enhancement**. If the stock trades flat, the client retains full shares while locking in the premium. If the stock rallies, shares are called away at a net exit price of NT$2,535.2.

#### 2. Protective Put (Institutional Tail-Risk Hedge)
* **Position**: Long 1,000 Lots Stock + Long 1,000 Contracts OTM 7% Put ($K = \text{NT}\$2,232$).
* **Mechanism**: The client pays NT$14.5 Million in premium (an annualized cost of -7.35%) to secure a catastrophic loss floor.
* **Payoff Profile**: Maximum downside risk is strictly locked at **-7.60%** regardless of market declines, preserving uncapped upside participation.

#### 3. Zero-Cost Collar (Costless Corridor)
* **Position**: Long 1,000 Lots Stock + Long OTM 7% Put ($K = \text{NT}\$2,232$) + Short Solved OTM 5.87% Call ($K = \text{NT}\$2,541$).
* **Mechanism**: The premium collected from writing the Call completely finances the purchase of the protective Put.
* **Payoff Profile**: **Zero cash outflow**. 30-day portfolio returns are strictly bounded within the corridor of **$[-7.00\%, +5.87\%]$**.

---

## 4. Visual Diagnostics Manual

High-resolution 300 DPI analytical charts are automatically rendered into `output/`:

### 1. Multi-Strategy Payoff Geometry (`output/pnl_comparison.png`)

![Multi-Strategy PnL Comparison](output/pnl_comparison.png)

* **Key Features**: Overlays payoff trajectories across all four configurations at expiration. Features clean legend integration with formatted Breakeven points, vertical reference lines matching strategy colors, and a lower-right **Breakeven (BE) Summary Box** that prevents text collisions along the $y=0$ line.

---

### 2. Interim PnL & Theta Decay Dynamics (`output/theta_decay.png`)

![Covered Call Theta Decay Dynamics](output/theta_decay.png)

* **Key Features**: Illustrates the time-decay progression of the Covered Call position across $T=30$ (Inception), $T=15$ (Mid-term), and $T=0$ (Expiration). The shaded **Positive Theta Capture Region** demonstrates how extrinsic value decay automatically lifts mark-to-market portfolio value over time during stock consolidation.

---

### 3. Gamma Sensitivity Surface & Pin Risk (`output/greeks_surface.png`)

![Gamma Sensitivity Surface](output/greeks_surface.png)

* **Key Features**: Renders a 3D sensitivity manifold across spot prices and days to expiration. Highlights the **Pin Risk Axis** ($K = \text{NT}\$2,541$). As $T \to 0$, Gamma spikes exponentially near the strike due to the $\frac{1}{\sqrt{T}}$ denominator, illustrating the non-linear slippage risk market makers absorb when dynamically hedging ATM options into expiration.

---

## 5. Quickstart & Pytest Verification

### 1. Installation

Clone the repository and install dependencies in a Python 3.9+ environment:

```bash
git clone https://github.com/hslinspiration/taiwan-equity-derivatives-suite.git
cd taiwan-equity-derivatives-suite
pip install -r requirements.txt
```

### 2. Execution via CLI Pipeline

Run the end-to-end structuring and pricing pipeline via command-line arguments:

```bash
# Standard Execution: TSMC (2330.TW), 1,000 Lots (1,000,000 Shares), 30-Day Tenor
python main.py --ticker 2330.TW --shares 1000000 --maturity 30

# Custom Execution: MediaTek (2454.TW), 500 Lots, 45-Day Tenor, Call OTM 7%, Put OTM 5%
python main.py --ticker 2454.TW --shares 500000 --maturity 45 --cc-otm 1.07 --put-otm 0.95
```

### 3. Automated Quantitative Testing (Pytest)

Run the test suite to verify numerical stability, Greeks bounds, and Put-Call parity:

```bash
pytest tests/ -v
```

* `test_put_call_parity`: Verifies no-arbitrage parity $C - P = S e^{-qT} - K e^{-rT}$ across all strikes with absolute tolerance $< 10^{-7}$.
* `test_expiration_boundary`: Enforces smooth convergence to intrinsic values as $T \to 0$.
* `test_delta_bounds`: Confirms Call Delta $\in [0, 1]$, Put Delta $\in [-1, 0]$, and $\Delta_{\text{Call}} - \Delta_{\text{Put}} = e^{-qT}$.
* `test_gamma_vega_positivity`: Confirms strictly positive Gamma and Vega for non-expired options.
* `test_theta_decay_sign`: Confirms negative daily time decay ($\Theta < 0$) for ATM options.

---

## 6. Author & Disclaimer

* **Developer**: Lin He-Shen (Antonio Lin)
* **Academic Focus**: Economics and Statistics, specializing in empirical econometrics, stochastic calculus, and quantitative derivatives engineering.
* **Disclaimer**: This software is developed for financial engineering research, quantitative modeling, and institutional scenario structuring. It does not constitute investment advice or a direct solicitation for securities trading.
