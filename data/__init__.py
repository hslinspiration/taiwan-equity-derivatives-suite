"""Data Ingestion and Market Feeds Package for Taiwan Derivatives.

Role:
    Exposes unified interfaces for acquiring historical spot prices, option chain
    quotes, interest rates, and dividend yields for Taiwan markets (TAIFEX, TWSE).

Dependencies:
    - .data_fetcher: Core market data acquisition and synthetic data generation classes.

Core Classes & Functions Exposed:
    - MarketDataFetcher: Main ingestion client for underlying and derivatives data.
    - OptionChainData: Data container representing an options chain snapshot.
    - MarketQuote: Single option contract quote container.
    - fetch_underlying_history: Helper function to retrieve historical price series.
    - generate_synthetic_option_chain: Helper for offline simulation and testing.
"""

from .data_fetcher import (
    MarketDataFetcher,
    OptionChainData,
    MarketQuote,
    fetch_underlying_history,
    generate_synthetic_option_chain,
)

__all__ = [
    "MarketDataFetcher",
    "OptionChainData",
    "MarketQuote",
    "fetch_underlying_history",
    "generate_synthetic_option_chain",
]
