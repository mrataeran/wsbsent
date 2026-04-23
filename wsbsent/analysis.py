"""
analysis.py
-----------
Statistical analysis: time-lagged Pearson correlations between sentiment and
returns, and a logistic regression classifier for market direction prediction.
"""

import pandas as pd
import numpy as np
from scipy import stats


def lagged_correlation(
    merged: pd.DataFrame,
    sentiment_col: str = "vader_compound",
    return_col: str = "^GSPC",
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compute Pearson correlations between a sentiment series and returns at
    multiple lags (sentiment leads return by ``lag`` days).

    Parameters
    ----------
    merged : pd.DataFrame
        Output of :func:`~wsbsent.wrangling.merge_sentiment_prices`.
    sentiment_col : str
        Sentiment column to use.
    return_col : str
        Return column to correlate against.
    max_lag : int
        Maximum lag (in trading days) to test.

    Returns
    -------
    pd.DataFrame
        Columns: ``lag``, ``pearson_r``, ``p_value``, ``significant`` (bool,
        using two-tailed alpha=0.05).
    """
    empty = pd.DataFrame(columns=["lag", "pearson_r", "p_value", "significant"])

    if sentiment_col not in merged.columns or return_col not in merged.columns:
        return empty

    s = merged[sentiment_col].dropna()
    r = merged[return_col].dropna()
    shared = s.index.intersection(r.index)
    s, r = s.loc[shared], r.loc[shared]

    if len(shared) < max(max_lag * 2 + 5, 10):
        return empty

    records = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            s_shifted = s.iloc[:-lag]
            r_shifted = r.iloc[lag:]
        elif lag < 0:
            s_shifted = s.iloc[abs(lag):]
            r_shifted = r.iloc[:lag]
        else:
            s_shifted, r_shifted = s, r

        if len(s_shifted) < 10:
            continue

        corr, pval = stats.pearsonr(s_shifted.values, r_shifted.values)
        records.append(
            {
                "lag": lag,
                "pearson_r": round(corr, 4),
                "p_value": round(pval, 4),
                "significant": pval < 0.05,
            }
        )

    return pd.DataFrame(records)


def classify_direction(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a logistic regression to classify next-day market direction.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`~wsbsent.wrangling.build_features`.
    feature_cols : list[str], optional
        Columns to use as predictors. Defaults to all columns except
        ``target_col`` and raw sentiment/return columns.
    target_col : str
        Binary target column name.
    test_size : float
        Fraction of data to reserve for testing (chronological split).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``'model'``, ``'accuracy'``, ``'report'``, ``'confusion'``,
        ``'feature_importance'``, ``'X_test'``, ``'y_test'``, ``'y_pred'``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    if feature_cols is None:
        exclude = {target_col, "post_count", "total_score", "total_comments"}
        raw_sentiment = [c for c in features.columns
                         if (c.startswith("vader_") or c.startswith("tb_"))
                         and not ("lag" in c or "roll" in c)]
        raw_returns = [c for c in features.columns
                       if not c.startswith("vader_") and not c.startswith("tb_")
                       and "lag" not in c and "roll" not in c
                       and c not in exclude]
        exclude |= set(raw_sentiment) | set(raw_returns)
        feature_cols = [c for c in features.columns if c not in exclude]

    X = features[feature_cols].values
    y = features[target_col].values

    # Chronological train/test split (no shuffle — preserves time ordering)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importance = pd.Series(
        np.abs(model.coef_[0]), index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred),
        "feature_importance": importance,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
