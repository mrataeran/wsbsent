"""
wrangling.py
------------
Merge daily sentiment scores with equity return data and engineer
time-lagged features for downstream modeling.
"""

import pandas as pd
import numpy as np


def merge_sentiment_prices(
    sentiment: pd.DataFrame,
    returns: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Align daily sentiment scores with equity return data.

    Parameters
    ----------
    sentiment : pd.DataFrame
        Daily sentiment from :func:`~wsbsent.sentiment.aggregate_daily_sentiment`.
        Must have a DatetimeIndex.
    returns : pd.DataFrame
        Daily returns from :func:`~wsbsent.prices.get_returns`.
        Must have a DatetimeIndex.
    how : str
        Join type: ``'inner'`` (default) keeps only overlapping dates.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with sentiment and return columns, indexed by date.
    """
    # Normalize both indices to date-only (drop time component if present)
    sentiment.index = pd.to_datetime(sentiment.index).normalize()
    returns.index = pd.to_datetime(returns.index).normalize()

    merged = sentiment.join(returns, how=how)
    merged = merged.sort_index()
    return merged


def build_features(
    merged: pd.DataFrame,
    sentiment_cols: list[str] | None = None,
    return_cols: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    target_col: str = "^GSPC",
    horizon: int = 1,
) -> pd.DataFrame:
    """Engineer lagged sentiment and technical features for classification.

    Parameters
    ----------
    merged : pd.DataFrame
        Output of :func:`merge_sentiment_prices`.
    sentiment_cols : list[str], optional
        Sentiment columns to lag. Defaults to all ``vader_*`` and ``tb_*`` cols.
    return_cols : list[str], optional
        Return columns to include as features (lagged by ``horizon``).
    lags : list[int], optional
        Lag periods to engineer. Default: ``[1, 2, 3, 5]``.
    rolling_windows : list[int], optional
        Rolling mean windows for sentiment smoothing. Default: ``[3, 5]``.
    target_col : str
        Column name of the prediction target (equity ticker/index).
    horizon : int
        Number of days ahead to predict. Default is 1 (next-day direction).

    Returns
    -------
    pd.DataFrame
        Feature matrix with a binary ``target`` column:
        1 if next-day return > 0, else 0. Rows with NaN are dropped.
    """
    if lags is None:
        lags = [1, 2, 3, 5]
    if rolling_windows is None:
        rolling_windows = [3, 5]

    if sentiment_cols is None:
        sentiment_cols = [
            c for c in merged.columns
            if c.startswith("vader_") or c.startswith("tb_")
        ]

    if return_cols is None:
        return_cols = [c for c in merged.columns if c not in sentiment_cols
                       and c not in ("post_count", "total_score", "total_comments")]

    feat = merged.copy()

    # Lagged sentiment features
    for col in sentiment_cols:
        for lag in lags:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    # Rolling mean sentiment
    for col in sentiment_cols:
        for w in rolling_windows:
            feat[f"{col}_roll{w}"] = feat[col].shift(1).rolling(w).mean()

    # Lagged returns
    for col in return_cols:
        if col in feat.columns:
            feat[f"{col}_lag1"] = feat[col].shift(1)

    # Target: binary direction
    if target_col in feat.columns:
        feat["target"] = (feat[target_col].shift(-horizon) > 0).astype(int)
    else:
        raise ValueError(
            f"target_col '{target_col}' not found in merged DataFrame. "
            f"Available columns: {list(feat.columns)}"
        )

    feat = feat.dropna()
    return feat
