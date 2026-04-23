"""
prices.py
---------
Fetch historical OHLCV data and compute daily log-returns for S&P 100 / S&P 500
constituents (or any valid Yahoo Finance ticker) using ``yfinance``.
"""

import pandas as pd
import numpy as np
import yfinance as yf

# S&P 100 tickers (as of 2025) — representative subset used as default universe
SP100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "JNJ", "WMT", "MA", "PG", "HD",
    "CVX", "MRK", "ABBV", "KO", "PEP", "COST", "AVGO", "TMO", "ACN",
    "MCD", "BAC", "CSCO", "ABT", "CRM", "ORCL", "NEE", "LIN", "DHR",
    "AMD", "INTC", "QCOM", "TXN", "HON", "IBM", "CAT", "DE", "GS",
    "MS", "BLK", "AXP", "SPG", "AMGN", "GILD", "ISRG", "MDLZ", "SLB",
    "GE", "MMM", "BA", "LMT", "RTX", "UPS", "FDX", "DIS", "NFLX",
    "ADBE", "NOW", "PYPL", "SQ", "SNAP", "UBER", "LYFT", "ABNB",
    "^GSPC",   # S&P 500 index
    "^OEX",    # S&P 100 index
]

INDEX_TICKERS = ["^GSPC", "^OEX"]


def get_prices(
    tickers: list[str] | None = None,
    start: str = "2023-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted closing prices.

    Parameters
    ----------
    tickers : list[str], optional
        List of Yahoo Finance tickers. Defaults to the two major indices.
    start : str
        Start date in ``'YYYY-MM-DD'`` format.
    end : str, optional
        End date. Defaults to today.
    interval : str
        Data frequency. ``'1d'`` for daily (default).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with tickers as columns and dates as index.
    """
    if tickers is None:
        tickers = INDEX_TICKERS

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    return prices


def generate_sample_prices(
    tickers: list[str] | None = None,
    start: str = "2024-01-01",
    n_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price data for testing (geometric Brownian motion).

    Parameters
    ----------
    tickers : list[str], optional
        Ticker names to simulate. Defaults to ``['^GSPC', '^OEX']``.
    start : str
        Start date.
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic adjusted close prices with the same schema as :func:`get_prices`.
    """
    if tickers is None:
        tickers = ["^GSPC", "^OEX"]

    np.random.seed(seed)
    dates = pd.bdate_range(start=start, periods=n_days)  # business days only

    prices = {}
    for ticker in tickers:
        s0 = 4800.0 if "GSPC" in ticker else 2200.0
        mu = 0.0003          # daily drift
        sigma = 0.012        # daily volatility
        shocks = np.random.normal(mu, sigma, n_days)
        path = s0 * np.exp(np.cumsum(shocks))
        prices[ticker] = path

    return pd.DataFrame(prices, index=dates)


def get_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Compute daily returns from a prices DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Output of :func:`get_prices`.
    log : bool
        If ``True`` (default), compute log-returns. Otherwise simple returns.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame aligned to ``prices``.
    """
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna(how="all")


def generate_sample_prices(
    tickers: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price data using geometric Brownian motion.

    Useful for testing when Yahoo Finance is unavailable.

    Parameters
    ----------
    tickers : list[str], optional
        Ticker names to simulate. Defaults to ``['^GSPC', '^OEX']``.
    start : str
        Start date.
    end : str
        End date.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of simulated adjusted close prices.
    """
    if tickers is None:
        tickers = ["^GSPC", "^OEX"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    prices_dict = {}
    start_prices = {"^GSPC": 4800.0, "^OEX": 2200.0}
    for ticker in tickers:
        s0 = start_prices.get(ticker, 100.0)
        mu, sigma = 0.0003, 0.012  # ~7.5% annual drift, ~19% vol
        shocks = rng.normal(mu, sigma, n)
        log_prices = np.log(s0) + np.cumsum(shocks)
        prices_dict[ticker] = np.exp(log_prices)

    df = pd.DataFrame(prices_dict, index=dates)
    df.index.name = "Date"
    return df
