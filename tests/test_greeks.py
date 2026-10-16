"""Automated Unit Tests for Analytical Option Greeks.

Role:
    Validates analytical bounds, parity relationships, positivity constraints,
    and time decay signs for Black-Scholes-Merton Greeks (Delta, Gamma, Vega,
    Theta, Rho) across scalar and vectorized inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.greeks import GreeksCalculator


def test_delta_bounds() -> None:
    """Verify Call/Put Delta bounds and Call-Put Delta parity relationship."""
    calc = GreeksCalculator()
    spots = np.linspace(700.0, 1300.0, 50)
    K = 1000.0
    T = 0.5
    r = 0.015
    q = 0.022
    sigma = 0.25

    call_deltas = calc.delta(spots, K, T, r, q, sigma, option_type="call")
    put_deltas = calc.delta(spots, K, T, r, q, sigma, option_type="put")

    # Call Delta strictly within [0.0, 1.0]
    assert np.all((call_deltas >= 0.0) & (call_deltas <= 1.0))

    # Put Delta strictly within [-1.0, 0.0]
    assert np.all((put_deltas >= -1.0) & (put_deltas <= 0.0))

    # Delta parity: Delta_Call - Delta_Put == exp(-q * T)
    expected_diff = np.exp(-q * T)
    assert np.allclose(call_deltas - put_deltas, expected_diff, atol=1e-7)


def test_gamma_vega_positivity() -> None:
    """Verify Gamma and Vega are strictly positive (> 0) for non-expired options."""
    calc = GreeksCalculator()
    spots = np.linspace(800.0, 1200.0, 25)
    K = 1000.0
    T = 0.5
    r = 0.015
    q = 0.022
    sigma = 0.28

    gammas = calc.gamma(spots, K, T, r, q, sigma)
    vegas = calc.vega(spots, K, T, r, q, sigma)

    assert np.all(gammas > 0.0)
    assert np.all(vegas > 0.0)

    # Boundary check at expiration (T = 0): Gamma and Vega must be 0.0
    assert calc.gamma(1000.0, K, 0.0, r, q, sigma) == 0.0
    assert calc.vega(1000.0, K, 0.0, r, q, sigma) == 0.0


def test_theta_decay_sign() -> None:
    """Verify ATM Call and Put Theta are strictly negative (decaying with time)."""
    calc = GreeksCalculator()
    S = 1000.0
    K = 1000.0
    T = 0.25
    r = 0.015
    q = 0.020
    sigma = 0.25

    call_theta = calc.theta(S, K, T, r, q, sigma, option_type="call")
    put_theta = calc.theta(S, K, T, r, q, sigma, option_type="put")

    assert isinstance(call_theta, float)
    assert isinstance(put_theta, float)
    assert call_theta < 0.0
    assert put_theta < 0.0


def test_rho_and_all_greeks_reporting() -> None:
    """Verify Rho signs and all_greeks dictionary structure."""
    calc = GreeksCalculator()
    S = 1000.0
    K = 1000.0
    T = 0.5
    r = 0.015
    q = 0.022
    sigma = 0.25

    call_greeks = calc.all_greeks(S, K, T, r, q, sigma, option_type="call")
    put_greeks = calc.all_greeks(S, K, T, r, q, sigma, option_type="put")

    expected_keys = {"delta", "gamma", "vega", "theta", "rho"}
    assert set(call_greeks.keys()) == expected_keys
    assert set(put_greeks.keys()) == expected_keys

    # Call Rho > 0, Put Rho < 0 for T > 0
    assert call_greeks["rho"] > 0.0
    assert put_greeks["rho"] < 0.0
