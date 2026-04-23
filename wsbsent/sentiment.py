"""
sentiment.py
------------
Apply VADER and TextBlob sentiment scorers to WSB post text, then aggregate
scores to daily granularity (volume-weighted by post score).
"""

import pandas as pd
import numpy as np


def _get_vader():
    """Lazy-load VADER to avoid import overhead when unused."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is required: pip install vaderSentiment"
        ) from exc


def _get_textblob():
    try:
        from textblob import TextBlob
        return TextBlob
    except ImportError as exc:
        raise ImportError("textblob is required: pip install textblob") from exc


def score_posts(
    df: pd.DataFrame,
    text_col: str = "title",
    include_selftext: bool = True,
    method: str = "both",
) -> pd.DataFrame:
    """Score each post with VADER and/or TextBlob sentiment.

    Parameters
    ----------
    df : pd.DataFrame
        Posts dataframe from :func:`~wsbsent.collect.scrape_wsb`.
    text_col : str
        Primary text column to score. Default is ``'title'``.
    include_selftext : bool
        If ``True``, concatenate ``selftext`` to ``title`` before scoring.
    method : str
        Which scorer(s) to apply: ``'vader'``, ``'textblob'``, or ``'both'``.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional sentiment columns:
        ``vader_compound``, ``vader_pos``, ``vader_neg``, ``vader_neu``
        and/or ``tb_polarity``, ``tb_subjectivity``.
    """
    out = df.copy()

    if include_selftext and "selftext" in out.columns:
        text = (
            out[text_col].fillna("") + " " + out["selftext"].fillna("")
        ).str.strip()
    else:
        text = out[text_col].fillna("")

    if method in ("vader", "both"):
        analyzer = _get_vader()
        scores = text.apply(lambda t: analyzer.polarity_scores(t))
        out["vader_compound"] = scores.apply(lambda s: s["compound"])
        out["vader_pos"] = scores.apply(lambda s: s["pos"])
        out["vader_neg"] = scores.apply(lambda s: s["neg"])
        out["vader_neu"] = scores.apply(lambda s: s["neu"])

    if method in ("textblob", "both"):
        TextBlob = _get_textblob()
        tb = text.apply(lambda t: TextBlob(t).sentiment)
        out["tb_polarity"] = tb.apply(lambda s: s.polarity)
        out["tb_subjectivity"] = tb.apply(lambda s: s.subjectivity)

    return out


def aggregate_daily_sentiment(
    scored: pd.DataFrame,
    date_col: str = "date",
    weight_col: str | None = "score",
) -> pd.DataFrame:
    """Aggregate post-level sentiment scores to daily means.

    Parameters
    ----------
    scored : pd.DataFrame
        Output of :func:`score_posts`.
    date_col : str
        Column containing post dates.
    weight_col : str or None
        If provided, compute a weighted average using this column
        (e.g., upvote ``'score'``). Pass ``None`` for simple mean.

    Returns
    -------
    pd.DataFrame
        Daily sentiment DataFrame indexed by date with mean scores and
        post/comment volume columns.
    """
    sentiment_cols = [
        c for c in scored.columns
        if c.startswith("vader_") or c.startswith("tb_")
    ]

    groups = scored.groupby(date_col)

    if weight_col and weight_col in scored.columns:
        def wmean(grp):
            w = grp[weight_col].clip(lower=0) + 1  # avoid zero weights
            result = {}
            for col in sentiment_cols:
                result[col] = np.average(grp[col], weights=w)
            result["post_count"] = len(grp)
            result["total_score"] = grp[weight_col].sum()
            if "num_comments" in grp.columns:
                result["total_comments"] = grp["num_comments"].sum()
            return pd.Series(result)

        daily = groups.apply(wmean)
    else:
        agg = {col: "mean" for col in sentiment_cols}
        agg["post_id"] = "count"
        if "num_comments" in scored.columns:
            agg["num_comments"] = "sum"
        daily = groups.agg(agg).rename(
            columns={"post_id": "post_count", "num_comments": "total_comments"}
        )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily
