"""Market Data Acquisition and Processing Module for Taiwan Equities and Derivatives.

Role:
    Handles dynamic ingestion, sanitization, corporate action adjustments,
    and caching of Taiwan financial market data (TWSE equities, TAIFEX derivatives,
    Taiwan Central Bank risk-free benchmarks, and trailing dividend yields).

Financial Engineering & Market Microstructure Context:
    - Adjusted Close (Adj Close) vs Unadjusted Close:
      In the Taiwan equity market, listed corporations (such as TSMC 2330.TW) frequently
      pay substantial cash and stock dividends (e.g., TSMC's quarterly dividend distributions).
      On the ex-dividend date (除息日), TWSE mechanically reduces the opening reference
      price by the exact cash dividend per share:
          S_ex = S_cum - Dividend
      Using raw unadjusted 'Close' prices creates artificial negative return shocks
      (e.g., -2% to -4% on ex-dividend day) that do not reflect true economic loss.
      These phantom jumps severely contaminate sample variance and artificially inflate
      historical volatility (HV) estimators. Therefore, 'Adj Close' is strictly required
      to eliminate corporate-action-induced price discontinuities and preserve continuous
      log returns.
    - Taiwan Risk-Free Benchmark:
      Taiwan derivatives (TAIFEX TXO and single stock options) typically benchmark
      the risk-free rate (r) against the Central Bank of the Republic of China (CBC)
      discount rate (重貼現率) or short-term interbank money market benchmarks (TAIBOR /
      90-day Commercial Paper secondary rate). When live CBC open data APIs are
      unreachable, the engine gracefully falls back to the institutional fallback rate.
    - Trailing 12-Month Continuous Dividend Yield:
      The Black-Scholes-Merton continuous dividend extension (Merton, 1973) requires
      an annualized continuous dividend yield parameter q:
          C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
      For Taiwan equities with quarterly or annual payouts, the sum of all cash
      dividends distributed across the trailing 365 calendar days divided by the latest
      adjusted spot price provides a consistent and robust empirical proxy for q.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# =====================================================================
# Option Quote & Chain Data Containers
# =====================================================================
@dataclass
class MarketQuote:
    """Individual option contract quote and market state snapshot.

    Attributes:
        symbol: Contract identifier (e.g., 'TXO20000C4' or '2330_202610_1000_C').
        strike: Option exercise strike price in index points or NTD.
        expiry: Option expiration date.
        option_type: Option contract right ('call' or 'put').
        bid: Current highest unexecuted buy quote.
        ask: Current lowest unexecuted sell quote.
        last: Most recent executed transaction price.
        volume: Accumulated transaction volume in contracts.
        open_interest: Number of outstanding open interest contracts.
        implied_vol: Annualized implied volatility implied by market price.
    """

    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int = 0
    open_interest: int = 0
    implied_vol: Optional[float] = None

    @property
    def mid_price(self) -> float:
        """Calculate the bid-ask midpoint price, falling back to last trade price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


@dataclass
class OptionChainData:
    """Complete option chain snapshot across strikes and maturities.

    Attributes:
        underlying_symbol: Ticker of the underlying asset (e.g., '2330.TW', '^TWII').
        underlying_price: Current spot price of the underlying asset.
        timestamp: Time at which the snapshot was captured.
        quotes: Dictionary mapping (expiry, strike, option_type) to MarketQuote.
    """

    underlying_symbol: str
    underlying_price: float
    timestamp: datetime
    quotes: Dict[Tuple[date, float, str], MarketQuote] = field(default_factory=dict)

    def get_strikes(self, expiry: date) -> List[float]:
        """Return sorted unique strike prices available for a specific expiration date."""
        strikes = {k[1] for k in self.quotes.keys() if k[0] == expiry}
        return sorted(strikes)

    def get_expiries(self) -> List[date]:
        """Return sorted unique expiration dates available in the chain."""
        expiries = {k[0] for k in self.quotes.keys()}
        return sorted(expiries)


# =====================================================================
# Dynamic Market Data Fetcher Engine
# =====================================================================
class MarketDataFetcher:
    """Production market data ingestion pipeline for Taiwan equities and derivatives."""

    def __init__(
        self,
        ticker: str = config.DEFAULT_TICKER,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        system_config: Optional[config.SystemConfig] = None,
    ) -> None:
        """Initialize MarketDataFetcher with ticker and historical date bounds.

        Args:
            ticker: TWSE equity ticker (e.g. '2330.TW') or index symbol ('^TWII').
            start_date: Start date string in 'YYYY-MM-DD' format. If None, defaults
                to 2 years (730 calendar days) prior to today.
            end_date: End date string in 'YYYY-MM-DD' format. If None, defaults to today.
            system_config: Optional system configuration instance.
        """
        # Handle cases where system_config is passed as first positional arg for backward compatibility
        if isinstance(ticker, config.SystemConfig):
            self.system_config: config.SystemConfig = ticker
            self.ticker: str = config.DEFAULT_TICKER
        else:
            self.system_config = system_config or config.DEFAULT_CONFIG
            self.ticker = str(ticker)

        today: date = datetime.now().date()
        if end_date is None:
            self.end_date: str = today.strftime("%Y-%m-%d")
        else:
            self.end_date = str(end_date)

        if start_date is None:
            two_years_prior: date = today - timedelta(days=730)
            self.start_date: str = two_years_prior.strftime("%Y-%m-%d")
        else:
            self.start_date = str(start_date)

        self._cached_history: Optional[pd.DataFrame] = None

    def fetch_taiwan_risk_free_rate(
        self, fallback_rate: float = config.FALLBACK_RISK_FREE_RATE
    ) -> float:
        """Dynamically query Taiwan Central Bank (CBC) rediscount rate via Open Data API.

        Queries the official Taiwan Central Bank OpenAPI or Open Data JSON endpoint
        with a strict 3.0-second timeout. If the request encounters a timeout,
        network connectivity failure, HTTP status error, or JSON schema mismatch,
        a warning is emitted via standard logging and the pre-configured fallback rate
        is returned.

        Args:
            fallback_rate: Rate to return if the external endpoint is unreachable
                or cannot be parsed (defaults to config.FALLBACK_RISK_FREE_RATE).

        Returns:
            float: Annualized risk-free rate as a decimal (e.g., 0.015 for 1.50%).
        """
        # Official Taiwan CBC Open Data endpoints for Central Bank Discount Rates
        endpoints: List[str] = [
            "https://cpx.cbc.gov.tw/api/OpenData/DataSet?set_id=6022&index=0",
            "https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=EG28D01",
        ]

        headers: Dict[str, str] = {
            "User-Agent": "TaiwanEquityDerivativesSuite/2.0 (Quantitative Research)",
            "Accept": "application/json",
        }

        for url in endpoints:
            try:
                response = requests.get(url, headers=headers, timeout=3.0)
                if response.status_code != 200:
                    continue

                payload: Any = response.json()
                parsed_rate: Optional[float] = self._parse_cbc_rate_payload(payload)
                if parsed_rate is not None and 0.0 < parsed_rate < 0.25:
                    logger.info(
                        "Successfully retrieved dynamic Taiwan CBC discount rate: %.4f (%.2f%%) from %s",
                        parsed_rate,
                        parsed_rate * 100.0,
                        url,
                    )
                    return parsed_rate

            except requests.RequestException as exc:
                logger.warning(
                    "Network error connecting to CBC Open Data API endpoint %s: %s",
                    url,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected error querying CBC Open Data API endpoint %s: %s",
                    url,
                    exc,
                )

        logger.warning(
            "Dynamic CBC risk-free rate query failed or returned no valid data. "
            "Falling back to institutional default rate: %.4f (%.2f%%)",
            fallback_rate,
            fallback_rate * 100.0,
        )
        return float(fallback_rate)

    @staticmethod
    def _parse_cbc_rate_payload(payload: Any) -> Optional[float]:
        """Parse raw JSON payload from Taiwan Central Bank API to extract discount rate.

        Args:
            payload: Deserialized JSON payload (dict or list).

        Returns:
            Optional[float]: Extracted rate in decimal format (e.g., 0.02 for 2%), or None.
        """
        try:
            # Case 1: CBC OpenData OAS format: {"Data": [{"value": "..."}, ...]} or dict
            data_node: Any = payload.get("Data") if isinstance(payload, dict) else payload

            # If Data is a string with key-value pairs
            if isinstance(data_node, str):
                match = re.search(r"重貼現率['\":\s]+([0-9.]+)", data_node)
                if match:
                    val = float(match.group(1))
                    return val / 100.0 if val > 0.20 else val

            # If Data is a list of records
            if isinstance(data_node, list) and len(data_node) > 0:
                latest_entry = data_node[0]  # Or last depending on sorting
                if isinstance(latest_entry, dict):
                    # Check for explicit keys
                    for key in ["重貼現率", "discount_rate", "rate", "value"]:
                        if key in latest_entry:
                            raw_val = latest_entry[key]
                            if isinstance(raw_val, (int, float)):
                                return float(raw_val) / 100.0 if raw_val > 0.20 else float(raw_val)
                            if isinstance(raw_val, str):
                                match = re.search(r"([0-9.]+)", raw_val)
                                if match:
                                    val = float(match.group(1))
                                    return val / 100.0 if val > 0.20 else val

            # Case 2: Statistical Database DataSet format
            if isinstance(payload, dict) and "DataSet" in payload:
                ds = payload["DataSet"]
                if isinstance(ds, list) and len(ds) > 0:
                    for row in ds:
                        if isinstance(row, dict) and "重貼現率" in row:
                            val = float(row["重貼現率"])
                            return val / 100.0 if val > 0.20 else val

        except (ValueError, TypeError, IndexError) as parse_err:
            logger.debug("CBC payload parsing heuristic encountered exception: %s", parse_err)

        return None

    def fetch_history(self) -> pd.DataFrame:
        """Fetch historical daily OHLCV bar data using yfinance.

        Guarantees that the returned DataFrame:
            - Has a timezone-naive pd.DatetimeIndex sorted ascending.
            - Guarantees columns: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
            - Drops all rows with missing values (NaN).
            - Raises ValueError if the final dataset is empty.

        Returns:
            pd.DataFrame: Cleaned, sorted historical price dataset.

        Raises:
            ValueError: If ticker is invalid or returns zero rows after cleaning.
        """
        logger.debug(
            "Fetching historical bars for %s between %s and %s via yfinance.",
            self.ticker,
            self.start_date,
            self.end_date,
        )

        ticker_obj = yf.Ticker(self.ticker)
        raw_df: pd.DataFrame = ticker_obj.history(
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
        )

        if raw_df is None or raw_df.empty:
            raise ValueError(
                f"Historical market data for ticker '{self.ticker}' is empty or unavailable "
                f"between {self.start_date} and {self.end_date}."
            )

        df: pd.DataFrame = raw_df.copy()

        # Flatten MultiIndex columns if produced by recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize DatetimeIndex: ensure DatetimeIndex and timezone-naive
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        # Sort chronologically ascending
        df = df.sort_index(ascending=True)

        # Guarantee 'Adj Close' presence
        if "Adj Close" not in df.columns:
            if "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            else:
                raise ValueError(
                    f"Neither 'Adj Close' nor 'Close' found in historical data for '{self.ticker}'."
                )

        guaranteed_columns: List[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing_columns = [col for col in guaranteed_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Historical data for ticker '{self.ticker}' missing mandatory columns: {missing_columns}"
            )

        # Retain only standard columns and drop missing observations
        clean_df = df[guaranteed_columns].dropna()

        if clean_df.empty:
            raise ValueError(
                f"Historical data for ticker '{self.ticker}' contained only NaN values after cleaning."
            )

        self._cached_history = clean_df
        return clean_df

    def get_latest_price(self) -> float:
        """Return the latest available Adjusted Close price.

        Returns:
            float: Latest adjusted closing spot price.

        Raises:
            ValueError: If historical data cannot be retrieved or is empty.
        """
        df = self._cached_history if self._cached_history is not None else self.fetch_history()
        if "Adj Close" not in df.columns or df["Adj Close"].empty:
            raise ValueError(f"Adj Close price not available for {self.ticker}.")
        return float(df["Adj Close"].iloc[-1])

    def calculate_dividend_yield(self) -> float:
        """Calculate trailing 12-month cash dividend yield for continuous BSM models.

        Retrieves cash dividend distributions from yfinance for the ticker, sums
        all cash dividend distributions occurred within the preceding 365 calendar days,
        and divides by the latest available Adjusted Close price.

        Returns:
            float: Annualized dividend yield as a decimal (e.g., 0.022 for 2.2%).
                   Returns 0.0 if no dividends were recorded or distributed.
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            dividends: Optional[pd.Series] = ticker_obj.dividends

            if dividends is None or len(dividends) == 0:
                logger.info("No dividend records found for %s. Yield assumed 0.0.", self.ticker)
                return 0.0

            div_series: pd.Series = dividends.copy()

            # Ensure DatetimeIndex and timezone-naive
            if not isinstance(div_series.index, pd.DatetimeIndex):
                div_series.index = pd.to_datetime(div_series.index)

            if getattr(div_series.index, "tz", None) is not None:
                div_series.index = div_series.index.tz_localize(None)

            # Lookback exactly 365 calendar days from now
            cutoff_datetime: pd.Timestamp = pd.Timestamp.now().normalize() - pd.Timedelta(days=365)
            recent_divs = div_series[div_series.index >= cutoff_datetime]

            total_dividends: float = float(recent_divs.sum())
            if total_dividends <= 0.0:
                return 0.0

            latest_spot: float = self.get_latest_price()
            if latest_spot <= 0.0:
                logger.warning(
                    "Latest price for %s is non-positive (%.2f). Cannot compute dividend yield.",
                    self.ticker,
                    latest_spot,
                )
                return 0.0

            div_yield: float = float(total_dividends / latest_spot)
            logger.info(
                "Calculated trailing 12M cash dividend yield for %s: %.4f (%.2f%%) "
                "[Total Div: NT$%.2f, Spot: NT$%.2f]",
                self.ticker,
                div_yield,
                div_yield * 100.0,
                total_dividends,
                latest_spot,
            )
            return div_yield

        except Exception as exc:
            logger.warning(
                "Failed to calculate trailing dividend yield for %s: %s. Defaulting to 0.0.",
                self.ticker,
                exc,
            )
            return 0.0

    # =================================================================
    # Backward Compatibility Aliases & Convenience Methods
    # =================================================================
    def get_underlying_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Retrieve historical price series, honoring symbol overrides if supplied."""
        if symbol is not None and symbol != self.ticker:
            fetcher = MarketDataFetcher(ticker=symbol, start_date=start_date, end_date=end_date)
            return fetcher.fetch_history()
        return self.fetch_history()

    def get_current_risk_free_rate(self) -> float:
        """Alias returning Taiwan risk-free rate proxy."""
        return self.fetch_taiwan_risk_free_rate()

    def get_dividend_yield(self, symbol: Optional[str] = None) -> float:
        """Alias returning annualized dividend yield for the target symbol."""
        if symbol is not None and symbol != self.ticker:
            fetcher = MarketDataFetcher(ticker=symbol)
            return fetcher.calculate_dividend_yield()
        return self.calculate_dividend_yield()


# =====================================================================
# Standalone Helper Functions
# =====================================================================
def fetch_underlying_history(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Standalone functional helper to retrieve historical OHLCV data.

    Args:
        symbol: Target equity or index ticker.
        start_date: Optional start date string ('YYYY-MM-DD').
        end_date: Optional end date string ('YYYY-MM-DD').

    Returns:
        pd.DataFrame: Cleaned historical price dataframe.
    """
    fetcher = MarketDataFetcher(ticker=symbol, start_date=start_date, end_date=end_date)
    return fetcher.fetch_history()


def generate_synthetic_option_chain(
    spot: float,
    expiry_days: List[int],
    strikes: List[float],
    base_iv: float = 0.18,
    risk_free_rate: float = config.FALLBACK_RISK_FREE_RATE,
    dividend_yield: float = 0.035,
) -> OptionChainData:
    """Generate a realistic synthetic option chain snapshot for simulations and tests.

    Args:
        spot: Underlying asset spot price.
        expiry_days: Days to maturity for each option expiration slice.
        strikes: Available strike prices.
        base_iv: Baseline at-the-money implied volatility.
        risk_free_rate: Continuous risk-free interest rate.
        dividend_yield: Continuous dividend yield.

    Returns:
        OptionChainData: Populated synthetic option chain.
    """
    now = datetime.now()
    chain = OptionChainData(
        underlying_symbol="TXO_SYNTHETIC",
        underlying_price=spot,
        timestamp=now,
    )

    for d in expiry_days:
        exp_date: date = (now + timedelta(days=d)).date()
        t_years = max(d / 365.0, 1e-4)

        for k in strikes:
            # Quadratic volatility smile heuristic: IV increases with OTM moneyness
            log_moneyness = np.log(spot / k)
            iv = float(base_iv + 0.12 * (log_moneyness**2) - 0.03 * log_moneyness)
            iv = max(0.05, min(iv, 1.50))

            # Approximation for Call/Put synthetic mid prices using intrinsic + time value
            d1 = (np.log(spot / k) + (risk_free_rate - dividend_yield + 0.5 * iv**2) * t_years) / (
                iv * np.sqrt(t_years)
            )
            d2 = d1 - iv * np.sqrt(t_years)

            # Standard normal CDF approximation
            from scipy.stats import norm

            call_price = float(
                spot * np.exp(-dividend_yield * t_years) * norm.cdf(d1)
                - k * np.exp(-risk_free_rate * t_years) * norm.cdf(d2)
            )
            put_price = float(
                k * np.exp(-risk_free_rate * t_years) * norm.cdf(-d2)
                - spot * np.exp(-dividend_yield * t_years) * norm.cdf(-d1)
            )

            # Generate Call Quote
            call_sym = f"TXO_{exp_date.strftime('%Y%m')}_{int(k)}_C"
            chain.quotes[(exp_date, float(k), "call")] = MarketQuote(
                symbol=call_sym,
                strike=float(k),
                expiry=exp_date,
                option_type="call",
                bid=max(0.0, call_price - 1.0),
                ask=call_price + 1.0,
                last=call_price,
                volume=150,
                open_interest=1200,
                implied_vol=iv,
            )

            # Generate Put Quote
            put_sym = f"TXO_{exp_date.strftime('%Y%m')}_{int(k)}_P"
            chain.quotes[(exp_date, float(k), "put")] = MarketQuote(
                symbol=put_sym,
                strike=float(k),
                expiry=exp_date,
                option_type="put",
                bid=max(0.0, put_price - 1.0),
                ask=put_price + 1.0,
                last=put_price,
                volume=120,
                open_interest=980,
                implied_vol=iv,
            )

    return chain
