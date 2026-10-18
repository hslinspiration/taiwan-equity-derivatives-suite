
---

```markdown
# Taiwan Equity Derivatives & Structuring Suite
### Institutional Pricing, Structuring Engine & Desk Risk Management Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-vectorized-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-optimization-orange.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Core Positioning**: A production-grade quantitative toolkit tailored for Taiwan sell-side Equity Derivatives Desks and Institutional Sales Advisors (Structurers). The system bridges empirical market microstructural data, stochastic calculus pricing, volatility skew and borrowing cost calibration, institutional derivative structuring, and trading desk Delta-neutral dynamic hedging with PnL attribution.

---

##  Table of Contents
1. [System Overview & Commercial Utility](#1-system-overview--commercial-utility)
2. [Architecture & Mathematical Formulations](#2-architecture--mathematical-formulations)
3. [Institutional Structuring Playbook](#3-institutional-structuring-playbook)
4. [Visual Diagnostics Manual](#4-visual-diagnostics-manual)
5. [Quickstart & Pytest Verification](#5-quickstart--pytest-verification)
6. [Author & Disclaimer](#6-author--disclaimer)

---

## 1. System Overview & Commercial Utility

In Taiwan's equity capital markets, corporate insiders, major shareholders, and family offices hold massive equity positions in blue-chip, high-dividend stocks (e.g., TSMC, MediaTek). However, due to regulatory insider holding disclosure requirements, collateral maintenance ratios, and corporate governance constraints, these underlying shares often suffer from liquidity lock-up.

Standard academic option pricing models assume frictionless markets, zero borrowing fees, and constant volatility, creating a substantial disconnect from real-world trading desk execution. This suite delivers an **end-to-end quantitative pipeline** designed to bridge this gap:

```text
[ Market Data & Macro Rates ] ──▶ [ Annualized HV & Dividend Imputation ] ──▶ [ Vectorized BSM & Friction Calibration ]
                                                                                               │
                                                                                               ▼
[ Desk Delta-Neutral Hedging & PnL ] ◀── [ Executive Dashboard & Plots ] ◀── [ Institutional Structuring Assembly ]

```

###  End-to-End Quantitative Workflow

1. **Dynamic Market Ingestion & Data Hygiene**: Automatically ingests historical equity data, enforces the use of **Adjusted Close** to eliminate artificial ex-dividend price gaps, and fetches dynamic trailing-twelve-month (TTM) cash dividend yields alongside central bank policy rates.
2. **Volatility & Pricing Matrix**: Computes the sell-side standard 60-day historical volatility (HV) benchmark, drives a vectorized Black-Scholes-Merton engine, and incorporates moneyness-adjusted volatility skew and stock borrowing costs.
3. **Institutional Structuring**: Modularly packages **Covered Call** (Yield Enhancement), **Protective Put** (Downside Floor), and **Zero-Cost Collar** structures with automated numerical root-finding.
4. **Market-Maker Desk Simulation**: Adopts the trading desk counterparty perspective to compute initial Delta-neutral stock hedge ratios and conducts post-trade 2nd-order Taylor expansion PnL attribution.

###  Commercial Outcomes

* **Corporate Equity Monetization (Yield Enhancement)**: Empowers corporate clients to monetize stagnant inventory shares by capturing extrinsic time value (Theta) via out-of-the-money (OTM) Covered Calls, generating an annualized **3% to 6% incremental non-leveraged cash flow**.
* **Costless Downside Tail-Risk Protection**: Utilizes continuous root-finding algorithms ahead of earnings calls or macro risk events to automatically identify zero-premium Collar bands, achieving complete downside tail protection with **Zero Cash Outflow**.
* **Bridging Theory and Executable Pricing**: By parameterizing securities borrowing lending (SBL) borrow fees, volatility skews, and transaction taxes (0.3% Taiwan securities transaction tax), the system narrows the pricing divergence between theoretical models and executable desk quotes, significantly enhancing client advisory credibility.

---

## 2. Architecture & Mathematical Formulations

###  Software Engineering & Design Philosophy

* **Decoupled Pipeline Architecture**: Ingestion (`data/`), financial engineering engines (`models/`), institutional strategies (`strategies/`), risk management (`hedging/`), and rendering (`visualization/`) are strictly isolated. Connecting live Bloomberg API feeds or proprietary desk pricing libraries requires zero refactoring of upstream strategy layers.
* **Full NumPy Vectorization**: Replaces iterative Python loops with vectorized array broadcasting, calculating multi-strike spot grids ($200+$ price points) and multi-maturity sensitivity surfaces in milliseconds.
* **Linear Superposition Principle**: Deconstructs multi-leg structures into atomic `OptionLeg` primitives. Aggregate portfolio Greeks and payoffs are resolved via algebraic linear addition:

$$\text{Portfolio Greek} = \sum_{i} \text{Position}_i \times \text{Contracts}_i \times \text{Multiplier}_i \times \text{Greek}_i$$



---

###  Mathematical & Algorithmic Foundations

#### A. Vectorized BSM Engine with Merton Dividend Discount (`models/bs_engine.py`)

To capture the continuous dividend yield $q$ characteristic of major TWSE dividend-paying stocks, the pricing engine implements the closed-form Merton (1973) formulation:

$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

* **Singularity & Boundary Defense**: When $T \le 0$ (at expiration) or $\sigma \to 0$, normal CDF denominators collapse into `ZeroDivisionError`. The engine implements smooth asymptotic fallback to intrinsic payoffs:

$$\lim_{T \to 0} C = \max(S - K, 0), \quad \lim_{T \to 0} P = \max(K - S, 0)$$



#### B. Standardized Analytical Greeks & BS PDE Invariance (`models/greeks.py`)

Calculates analytical partial derivatives and scales them to institutional trading conventions:

| Greek | Analytical Closed-Form Formulation | Standardized Trading Unit | Financial & Risk Intuition |
| --- | --- | --- | --- |
| **Delta ($\Delta$)** | $\text{Call: } e^{-qT} N(d_1)$<br>

<br>$\text{Put: } -e^{-qT} N(-d_1)$ | Ratio $[0, 1]$ or $[-1, 0]$ | Replicating hedge ratio; asset-weighted exercise probability. |
| **Gamma ($\Gamma$)** | $\frac{e^{-qT} N'(d_1)}{S \sigma \sqrt{T}}$ | Identical for Call/Put | Curvature/convexity; rate of change of Delta per $1 stock move. |
| **Vega ($\mathcal{V}$)** | $S e^{-qT} \sqrt{T} N'(d_1)$ | $\mathcal{V} \div 100$ | Dollar price change per **1 percentage point (1%)** shift in implied vol. |
| **Theta ($\Theta$)** | $\Theta_{\text{BSM}}$ Merton Formulation | $\Theta \div 365$ | Extrinsic dollar value decayed per **1 calendar day**. |
| **Rho ($\rho$)** | $\frac{\partial V}{\partial r}$ Formulation | $\rho \div 100$ | Dollar price sensitivity per **100 bps** interest rate shift. |

* **Gamma-Theta Trade-off (Black-Scholes PDE)**:
Under a Delta-neutral portfolio ($\Delta = 0, q = 0$), the Black-Scholes PDE reveals the foundational market-making invariance principle:

$$\Theta = \frac{1}{2} \sigma^2 S^2 \Gamma - rV$$



To collect positive time-decay rent ($\Theta > 0$), the desk must absorb negative convexity risk ($\Gamma < 0$) against market jump shocks.

#### C. Microstructural Friction & Volatility Skew Calibration (`models/desk_calibrator.py`)

To align theoretical midpoint prices with institutional execution reality:

1. **Moneyness-Adjusted Volatility Skew**:
Reflects the structural market bias where downside put protection commands higher implied volatility:

$$\sigma(K) = \sigma_{\text{ATM}} \times \left(1 - \beta \cdot \ln\left(\frac{K}{S}\right)\right), \quad \text{Default Taiwan Skew Slope } \beta = 0.45$$


2. **Short-Hedge Stock Borrowing Cost ($b$)**:
When clients execute Covered Calls (Client Sells Call $\to$ Desk Buys Call), the desk must short stock to hedge positive Delta. Incorporating annual borrow cost $b$ (default 2.0%) modifies the effective carrying cost:

$$q_{\text{eff}} = q + b$$



Directly reflecting borrow cost drag into executable option premiums.
3. **Dynamic Rebalancing Friction**: Deducts a 3.0% buffer against Taiwan's 0.3% securities transaction tax and execution slippage during rapid rebalancing.

#### D. Numerical Root-Finding for Zero-Cost Collars (`strategies/collar.py`)

A costless Collar requires solving for the exact Call strike $K_{\text{Call}}$ such that:


$$f(K_{\text{Call}}) = \text{BSM}_{\text{Call}}(S, K_{\text{Call}}, T, r, q, \sigma(K_{\text{Call}})) - \text{Price}_{\text{Put target}} = 0$$


Because $K_{\text{Call}}$ is embedded inside non-invertible cumulative normal integrals, analytical inversion is impossible. The engine applies **Brent's Method (`scipy.optimize.brentq`)**, combining robust bisection with superlinear inverse quadratic interpolation to converge on a tolerance of $< 10^{-8}$ within milliseconds.

#### E. 2nd-Order Taylor PnL Attribution (`hedging/delta_hedger.py`)

Simulates daily discrete delta rebalancing, decomposing daily desk PnL into fundamental risk drivers:


$$\Delta \Pi \approx \underbrace{(\Delta_{\text{opt}} \cdot \Delta S + N_{\text{stock}} \cdot \Delta S)}_{\text{Delta Leakage (Directional Residual)}} + \underbrace{\frac{1}{2}\Gamma_{\text{opt}} \cdot (\Delta S)^2}_{\text{Gamma Drag / Convexity PnL}} + \underbrace{\Theta_{\text{opt}} \cdot \Delta t}_{\text{Theta Decay PnL}} - \text{Rebalancing Taxes \& Fees}$$

---

## 3. Institutional Structuring Playbook

**Base Scenario**: Underlying: **TSMC (2330.TW)** | Spot Price $S_0 = \text{NT}\$2,400$ | Notional: **1,000 Lots (1,000,000 Shares / NT$2.4 Billion Market Value)** | Maturity: **30 Days** | Base HV: **24.50%**:

```text
================================================================================================
【Taiwan Equity Derivatives Executive Dashboard】 2330.TW | Spot: NT$2,400.0 | Base HV: 24.50%
================================================================================================
Strategy Structure  Contract Spec       Net Cash Premium  Ann. Yield / Cost Downside Buffer Max Upside
------------------------------------------------------------------------------------------------
Underlying Stock    Long 1,000 Lots     NT$ 0.00          0.00%             None (Full Risk) Unlimited
Covered Call        Short 1M K=2520 C   +NT$ 15.2M (Recv) +7.70% (Yield)    -0.63% (Buffer)  +5.63%
Protective Put      Long 1M K=2232 P    -NT$ 14.5M (Paid) -7.35% (Cost)     Capped at -7.60% Unlimited
Zero-Cost Collar    Long P2232 / Short C2541 NT$ 0.00     0.00% (Costless)  Capped at -7.00% +5.87%
================================================================================================
[Desk Hedging Perspective] Initial Delta-Neutral Stock Short Hedge: 285 Lots (285,000 Shares)
================================================================================================

```

###  Strategy Deep-Dives

#### 1. Covered Call (Corporate Yield Enhancement)

* **Position**: Long 1,000 Lots Stock + Short 1,000 Contracts OTM 5% Call ($K = \text{NT}\$2,520$).
* **Mechanism**: The shareholder forfeits capital appreciation above NT$2,520 in exchange for an immediate cash premium of NT$15.2 Million.
* **Payoff Profile**: Generates an annualized **+7.70% yield enhancement**. If the stock trades flat, the client retains full shares while locking in the premium. If the stock rallies, shares are effectively called away at a net exit price of NT$2,535.2.

#### 2. Protective Put (Institutional Tail-Risk Hedge)

* **Position**: Long 1,000 Lots Stock + Long 1,000 Contracts OTM 7% Put ($K = \text{NT}\$2,232$).
* **Mechanism**: The client pays NT$14.5 Million in premium (an annualized drag of -7.35%) to secure a catastrophic loss floor.
* **Payoff Profile**: Maximum downside risk is strictly locked at **-7.60%** regardless of market declines, while maintaining full upside participation.

#### 3. Zero-Cost Collar (Costless Bound Corridor)

* **Position**: Long 1,000 Lots Stock + Long OTM 7% Put ($K = \text{NT}\$2,232$) + Short Solved OTM 5.87% Call ($K = \text{NT}\$2,541$).
* **Mechanism**: The premium collected from writing the Call completely finances the purchase of the protective Put.
* **Payoff Profile**: **Zero cash outflow**. Future 30-day portfolio returns are strictly bounded within the corridor of **$[-7.00\%, +5.87\%]$**.

---

## 4. Visual Diagnostics Manual

High-resolution 300 DPI analytical charts are automatically generated in `output/`:

### 1. Multi-Strategy Payoff Geometry (`output/pnl_comparison.png`)

Overlays payoff trajectories across all four configurations at expiration.

* **Key Feature**: Clean legend integration and a lower-right **Breakeven (BE) Summary Box** prevent text label collisions, clearly identifying breakeven boundaries across all strategies.


---

### 2. Interim PnL & Theta Decay Dynamics (`output/theta_decay.png`)

Illustrates the time-decay progression of the Covered Call position across $T=30$ (Inception), $T=15$ (Mid-term), and $T=0$ (Expiration).

* **Key Feature**: The shaded **Positive Theta Capture Region** demonstrates how extrinsic value decay automatically lifts the portfolio's mark-to-market value over time as spot prices consolidate.


---

### 3. Gamma Sensitivity Surface & Pin Risk (`output/greeks_surface.png`)

Renders a 3D sensitivity manifold across spot prices ($X$) and days to expiration ($Y$).

* **Key Feature**: Highlights the **Pin Risk Axis** ($K = \text{NT}\$2,541$). As $T \to 0$, Gamma spikes exponentially near the strike due to the $\frac{1}{\sqrt{T}}$ denominator, illustrating the non-linear slippage risk market makers absorb when dynamically hedging ATM options into expiration.


---

## 5. Quickstart & Pytest Verification

### 1. Installation

Clone the repository and install core dependencies in a Python 3.9+ environment:

```bash
git clone [https://github.com/hslinspiration/taiwan-equity-derivatives-suite.git](https://github.com/hslinspiration/taiwan-equity-derivatives-suite.git)
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

Run the test suite to verify numerical stability and arbitrage-free pricing boundaries:

```bash
pytest tests/ -v

```

* `test_put_call_parity`: Verifies no-arbitrage parity $C - P = S e^{-qT} - K e^{-rT}$ across all strikes with absolute tolerance $< 10^{-7}$.
* `test_expiration_boundary`: Enforces smooth convergence to intrinsic values as $T \to 0$.
* `test_greeks_signs_and_finite_difference`: Validates sign constraints (Call Delta in $[0, 1]$, positive Gamma/Vega) and confirms analytical Greeks against numerical finite-difference calculus.

---

## 6. Author & Disclaimer

* **Developer**: Lin He-Shen (Antonio Lin)
* **Academic Focus**: Economics and Statistics, specializing in empirical econometrics, stochastic calculus, and quantitative derivatives engineering.
* **Disclaimer**: This software is developed for financial engineering research, quantitative modeling, and institutional scenario structuring. It does not constitute investment advice or a direct solicitation for securities trading.

```

```
