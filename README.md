# Taiwan Equity Derivatives Suite (`taiwan_equity_derivatives_suite`)

A quantitative financial engineering library and derivatives analytics platform designed for the Taiwan equity derivatives market (TAIFEX & TWSE).

---

## 1. Project Overview

The **Taiwan Equity Derivatives Suite** provides quantitative tools for pricing, risk management, sensitivity analytics, and strategy simulation for Taiwan equity and index derivatives. 

Key market focal points include:
- **TAIFEX TAIEX Options (TXO)**: European index options with a multiplier of NT$50 per point.
- **Single-Stock Options (STOs)**: Options on TWSE/TPEx listed equities (e.g., TSMC `2330.TW`).
- **Taiwan Market Microstructure**: Continuous dividend yield adjustments for heavy summer ex-dividend seasonality, Taiwan futures/securities transaction taxes, and exchange fee structures.

---

## 2. Architecture

```text
taiwan_equity_derivatives_suite/
│
├── config.py                 # TAIFEX specifications, market rates, and solver settings
├── main.py                   # Master CLI entry point and execution pipeline
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation and outline
│
├── data/                     # Market data acquisition & synthetic option chains
│   ├── __init__.py
│   └── data_fetcher.py       # Historical price fetcher and quote container
│
├── models/                   # Financial engineering pricing & volatility engines
│   ├── __init__.py
│   ├── bs_engine.py          # Black-Scholes-Merton model with continuous dividend yield
│   ├── greeks.py             # 1st, 2nd, and cross Greeks calculator
│   └── volatility.py         # Implied volatility solver and realized volatility estimators
│
├── strategies/               # Quantitative option overlay strategies
│   ├── __init__.py
│   ├── base.py               # Abstract base class for multi-leg strategies
│   ├── covered_call.py       # Covered Call yield enhancement
│   ├── protective_put.py     # Protective Put downside floor
│   └── collar.py             # Collar corridor hedging
│
├── hedging/                  # Dynamic risk management and replication
│   ├── __init__.py
│   └── delta_hedger.py       # Discrete delta hedging simulator with transaction costs
│
├── visualization/            # Visual analytics and reporting
│   ├── __init__.py
│   └── plots.py              # Comparative PnL, Theta decay, and 3D Greek surfaces
│
└── tests/                    # Quantitative validation test suite
    ├── __init__.py
    ├── test_bs_engine.py     # Put-Call Parity and asymptotic boundary tests
    └── test_greeks.py        # Greek sign and bounds verification
```

---

## 3. Math & Theoretical Foundations

### 3.1 Black-Scholes-Merton with Continuous Dividend Yield (Merton 1973)
Given underlying spot $S$, strike $K$, risk-free rate $r$, continuous dividend yield $q$, volatility $\sigma$, and time to maturity $T$:

$$d_1 = \frac{\ln(S / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$$

$$d_2 = d_1 - \sigma \sqrt{T}$$

- **European Call Price ($C$)**:
  $$C = S e^{-q T} N(d_1) - K e^{-r T} N(d_2)$$

- **European Put Price ($P$)**:
  $$P = K e^{-r T} N(-d_2) - S e^{-q T} N(-d_1)$$

- **Put-Call Parity**:
  $$C - P = S e^{-q T} - K e^{-r T}$$

### 3.2 Analytical Greeks
- **Delta ($\Delta$)**:
  $$\Delta_{\text{call}} = e^{-q T} N(d_1), \quad \Delta_{\text{put}} = -e^{-q T} N(-d_1)$$
- **Gamma ($\Gamma$)**:
  $$\Gamma = \frac{e^{-q T} N'(d_1)}{S \sigma \sqrt{T}}$$
- **Vega ($\mathcal{V}$)**:
  $$\mathcal{V} = S e^{-q T} \sqrt{T} N'(d_1)$$
- **Theta ($\Theta$)**:
  $$\Theta_{\text{call}} = -\frac{S e^{-q T} N'(d_1) \sigma}{2 \sqrt{T}} + q S e^{-q T} N(d_1) - r K e^{-r T} N(d_2)$$

---

## 4. Institutional Strategies

- **Covered Call (Buy-Write)**:
  - Long Underlying + Short OTM Call.
  - Used for income generation and yield enhancement on core holdings.
- **Protective Put (Portfolio Insurance)**:
  - Long Underlying + Long OTM/ATM Put.
  - Establishes a hard floor on portfolio losses while keeping upside open.
- **Collar (Corridor Hedge)**:
  - Long Underlying + Long OTM Put + Short OTM Call.
  - Financed corridor protection, often structured as near zero-cost.
- **Dynamic Delta Hedging**:
  - Replicating portfolio rebalanced discretely subject to Taiwan transaction costs and slippage.

---

## 5. How to Run

### 5.1 Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Running the Test Suite
```bash
pytest -v
```

### 5.3 Executing the Pipeline CLI
```bash
# Run default demo
python main.py

# Run specific strategy analysis with custom spot
python main.py --symbol TXO --spot 20500 --strategy collar

# Run dynamic delta hedging simulation
python main.py --simulate-hedge --verbose
```
