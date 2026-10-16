"""Automated Unit Tests for the Vectorized Black-Scholes-Merton Pricing Engine.

Role:
    Validates European Put-Call parity across moneyness regimes (ITM, ATM, OTM),
    expiration boundary behavior (T <= 0), zero-volatility limits (sigma <= 0),
    and NumPy array vectorization for spot and strike price grids.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.bs_engine import BSEngine


def test_put_call_parity() -> None:
    """Verify Call - Put == S * exp(-q * T) - K * exp(-r * T) across ITM, ATM, and OTM strikes."""
    engine = BSEngine()
    S = 1000.0
    T = 0.5
    r = 0.015
    q = 0.022
    sigma = 0.28

    # Strikes spanning ITM, ATM, and OTM regimes
    strikes = [800.0, 900.0, 1000.0, 1100.0, 1200.0]

    for K in strikes:
        call = engine.call_price(S, K, T, r, q, sigma)
        put = engine.put_price(S, K, T, r, q, sigma)
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-7, (
            f"Put-Call parity failed at strike K={K}: |{lhs} - {rhs}| = {abs(lhs - rhs)}"
        )


def test_expiration_boundary() -> None:
    """Verify that when T = 0, call_price == max(S - K, 0) and put_price == max(K - S, 0)."""
    engine = BSEngine()
    r = 0.015
    q = 0.022
    sigma = 0.25

    test_cases = [
        (1050.0, 1000.0),  # Call ITM / Put OTM
        (1000.0, 1000.0),  # ATM
        (920.0, 1000.0),   # Call OTM / Put ITM
    ]

    for S, K in test_cases:
        call_t0 = engine.call_price(S, K, 0.0, r, q, sigma)
        put_t0 = engine.put_price(S, K, 0.0, r, q, sigma)

        expected_call = max(S - K, 0.0)
        expected_put = max(K - S, 0.0)

        assert isinstance(call_t0, float)
        assert isinstance(put_t0, float)
        assert abs(call_t0 - expected_call) < 1e-12
        assert abs(put_t0 - expected_put) < 1e-12

        # Also verify negative T edge case defense
        assert abs(engine.call_price(S, K, -0.01, r, q, sigma) - expected_call) < 1e-12
        assert abs(engine.put_price(S, K, -0.01, r, q, sigma) - expected_put) < 1e-12


def test_vectorized_inputs() -> None:
    """Pass a NumPy array of spot prices and verify output is a matching ndarray."""
    engine = BSEngine()
    spots = np.linspace(800.0, 1200.0, 50)
    K = 1000.0
    T = 0.25
    r = 0.015
    q = 0.022
    sigma = 0.25

    calls = engine.call_price(spots, K, T, r, q, sigma)
    puts = engine.put_price(spots, K, T, r, q, sigma)

    assert isinstance(calls, np.ndarray)
    assert isinstance(puts, np.ndarray)
    assert calls.shape == spots.shape
    assert puts.shape == spots.shape

    # Call prices must be monotonically increasing in spot S; Puts monotonically decreasing
    assert np.all(np.diff(calls) > 0.0)
    assert np.all(np.diff(puts) < 0.0)

    # Vectorized Put-Call parity check across the entire array
    parity_diff = (calls - puts) - (spots * np.exp(-q * T) - K * np.exp(-r * T))
    assert np.all(np.abs(parity_diff) < 1e-7)


def test_zero_volatility_boundary() -> None:
    """Verify discounted intrinsic payoff when sigma <= 0."""
    engine = BSEngine()
    S = 1050.0
    K = 1000.0
    T = 0.5
    r = 0.015
    q = 0.02

    call_zero_vol = engine.call_price(S, K, T, r, q, 0.0)
    put_zero_vol = engine.put_price(S, K, T, r, q, 0.0)

    expected_call = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    expected_put = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    assert abs(call_zero_vol - expected_call) < 1e-12
    assert abs(put_zero_vol - expected_put) < 1e-12
