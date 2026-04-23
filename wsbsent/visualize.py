"""
visualize.py
------------
Plotting utilities for sentiment vs. return time series, lag correlation
bar charts, confusion matrices, and feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PALETTE = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "price": "#2c3e50",
    "sentiment": "#3498db",
    "accent": "#9b59b6",
}


def plot_sentiment_returns(
    merged: pd.DataFrame,
    sentiment_col: str = "vader_compound",
    return_col: str = "^GSPC",
    title: str | None = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Dual-axis time series: sentiment vs. cumulative index returns.

    Parameters
    ----------
    merged : pd.DataFrame
        Must contain both a sentiment column and a return column.
    sentiment_col : str
        Daily sentiment score column.
    return_col : str
        Daily log-return column.
    title : str, optional
        Chart title.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    plt.Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    cumret = merged[return_col].cumsum() * 100
    ax1.plot(
        merged.index, cumret,
        color=PALETTE["price"], linewidth=1.8, label="Cumulative Return (%)"
    )
    ax1.set_ylabel("Cumulative Log-Return (%)", color=PALETTE["price"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=PALETTE["price"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2 = ax1.twinx()
    colors = [
        PALETTE["positive"] if v > 0 else PALETTE["negative"]
        for v in merged[sentiment_col]
    ]
    ax2.bar(
        merged.index, merged[sentiment_col],
        color=colors, alpha=0.4, width=1, label="Daily Sentiment"
    )
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Sentiment Score (VADER Compound)", color=PALETTE["sentiment"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=PALETTE["sentiment"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_title(
        title or f"WSB Sentiment vs {return_col} Cumulative Return",
        fontsize=14, fontweight="bold", pad=12
    )
    ax1.set_xlabel("Date", fontsize=12)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    lag_df: pd.DataFrame,
    figsize: tuple = (10, 4),
    title: str = "Lagged Pearson Correlation: WSB Sentiment → Equity Returns",
) -> plt.Figure:
    """Bar chart of lag correlations with significance markers.

    Parameters
    ----------
    lag_df : pd.DataFrame
        Output of :func:`~wsbsent.analysis.lagged_correlation`.
    figsize : tuple
        Figure dimensions.
    title : str
        Chart title.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if lag_df.empty or "pearson_r" not in lag_df.columns:
        ax.text(0.5, 0.5, "Not enough overlapping data to compute lag correlations.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    colors = [
        PALETTE["positive"] if r > 0 else PALETTE["negative"]
        for r in lag_df["pearson_r"]
    ]
    bars = ax.bar(lag_df["lag"], lag_df["pearson_r"], color=colors, alpha=0.75, width=0.7)

    # Mark significant bars
    for bar, sig in zip(bars, lag_df["significant"]):
        if sig:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003 * np.sign(bar.get_height() or 1),
                "*", ha="center", va="bottom", fontsize=12, color="#2c3e50"
            )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (days) — negative = returns lead sentiment", fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.annotate(
        "* p < 0.05", xy=(0.98, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=9, color="#666"
    )
    fig.tight_layout()
    return fig


def plot_confusion(
    confusion: np.ndarray,
    accuracy: float,
    figsize: tuple = (5, 4),
) -> plt.Figure:
    """Heatmap confusion matrix for the direction classifier.

    Parameters
    ----------
    confusion : np.ndarray
        2x2 confusion matrix from sklearn.
    accuracy : float
        Overall accuracy to display in title.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=["Bearish", "Bullish"],
        yticklabels=["Bearish", "Bullish"],
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.1%})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 15,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Horizontal bar chart of top feature importances (|coef| from LR).

    Parameters
    ----------
    importance : pd.Series
        Series with feature names as index and |coefficient| as values.
    top_n : int
        Number of features to display.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE["accent"]] * len(top)
    ax.barh(top.index[::-1], top.values[::-1], color=colors, alpha=0.8)
    ax.set_xlabel("|Logistic Regression Coefficient|", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
