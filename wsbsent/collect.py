"""
collect.py
----------
Scrape posts from r/wallstreetbets using the Arctic Shift API
(https://arctic-shift.photon-reddit.com). No API credentials required.
Supports pagination via timestamp-based cursoring.
"""

import time
import datetime
import requests
import pandas as pd
import numpy as np

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
HEADERS = {"User-Agent": "wsbsent/0.1.0 (academic research project)"}


def scrape_wsb(
    sort: str = "new",
    pages: int = 10,
    sleep: float = 1.0,
    after: str | None = None,
    before: str | None = None,
) -> pd.DataFrame:
    """Scrape r/wallstreetbets posts via the Arctic Shift API.

    Parameters
    ----------
    sort : str
        Sort order. One of 'new', 'top', 'controversial'.
    pages : int
        Number of paginated requests (each returns up to 100 posts).
    sleep : float
        Seconds to wait between requests.
    after : str, optional
        ISO date string (e.g. '2024-01-01') to fetch posts after this date.
    before : str, optional
        ISO date string to fetch posts before this date.

    Returns
    -------
    pd.DataFrame
        Columns: post_id, title, selftext, score,
        num_comments, upvote_ratio, created_utc, date,
        flair, url.
    """
    records = []
    params = {
        "subreddit": "wallstreetbets",
        "limit": 100,
        "sort": "desc" if sort == "new" else sort,
    }
    if after:
        params["after"] = after
    if before:
        params["before"] = before

    oldest_utc = None  # used as cursor for pagination

    for page in range(pages):
        if oldest_utc is not None:
            params["before"] = oldest_utc

        try:
            resp = requests.get(
                ARCTIC_SHIFT_URL, headers=HEADERS, params=params, timeout=15
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[collect] Arctic Shift request failed on page {page}: {exc}")
            break

        posts = resp.json().get("data", [])
        if not posts:
            break

        for post in posts:
            created = post.get("created_utc", 0)
            try:
                created = int(created)
            except (ValueError, TypeError):
                created = 0
            date = datetime.datetime.utcfromtimestamp(created).date() if created else None
            records.append({
                "post_id": post.get("id", ""),
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0.5),
                "created_utc": created,
                "date": date,
                "flair": post.get("link_flair_text", ""),
                "url": post.get("url", ""),
            })

        # Cursor: oldest timestamp in this batch becomes the next before= value
        oldest_utc = min(p.get("created_utc", 0) for p in posts)
        time.sleep(sleep)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset="post_id").sort_values("date").reset_index(drop=True)
    return df

def load_posts(path: str) -> pd.DataFrame:
    """Load a previously saved posts CSV.

    Parameters
    ----------
    path : str
        Path to a CSV file produced by :func:`scrape_wsb` or compatible schema.

    Returns
    -------
    pd.DataFrame
        Posts dataframe with ``date`` parsed as datetime.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def generate_sample_data(
    n_posts: int = 500,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic WSB post data for testing and demos.

    Useful when Reddit API access is unavailable. The synthetic data
    reproduces realistic distributions of upvote scores, comment counts,
    sentiment patterns, and flair proportions observed on WSB.

    Parameters
    ----------
    n_posts : int
        Number of synthetic posts to generate.
    start : str
        Date range start (``'YYYY-MM-DD'``).
    end : str
        Date range end.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`scrape_wsb`.
    """
    import random
    rng = np.random.default_rng(seed)
    random.seed(seed)

    bull_titles = [
        "NVDA is going to the moon 🚀", "Loading up on SPY calls before CPI",
        "This rally is just getting started", "Bought the dip, see you at ATH",
        "GME short interest is insane right now", "Fed pivot = infinite money glitch",
        "TSLA breaking out of the wedge", "Why I'm all-in on tech this quarter",
        "S&P 500 hit new ATH, bears destroyed", "Just turned $5k into $50k on options",
        "Market is pricing in too much fear", "Rotation into growth is happening NOW",
        "AMD earnings gonna send us to Mars", "Bull thesis for MSFT, long read but worth it",
        "Options flow is screaming bullish", "My 5-bagger play for Q4",
    ]
    bear_titles = [
        "Bear case for SPY, change my mind", "This market is in a bubble",
        "Hedging with puts before FOMC", "GDP miss incoming, stay cautious",
        "Sold everything, going to cash", "Why I think we revisit October lows",
        "Credit spreads are flashing red", "Inverse ETFs printing today",
        "VIX spike incoming, watch out", "This rally has no volume behind it",
        "Recession fears are real this time", "Took profits, market looks toppy",
    ]
    neutral_titles = [
        "DD: deep dive into earnings preview", "What's your play for this week?",
        "WSB daily discussion thread", "Positions update, show me yours",
        "Reading the tape on sector rotation", "Options expiry week strategy",
        "My portfolio after 1 year on WSB", "Thoughts on after-hours moves?",
        "Anyone else watching the yield curve?", "Rate my portfolio",
    ]
    flairs = ["DD", "Gain Porn", "Loss Porn", "YOLO", "Discussion",
              "News", "Meme", "Technical Analysis", ""]
    flair_weights = [0.15, 0.18, 0.12, 0.20, 0.15, 0.08, 0.06, 0.04, 0.02]

    date_range = pd.date_range(start=start, end=end, freq="D")
    records = []
    for i in range(n_posts):
        date = rng.choice(date_range)
        sentiment_draw = rng.random()
        if sentiment_draw > 0.55:
            title = random.choice(bull_titles)
        elif sentiment_draw < 0.3:
            title = random.choice(bear_titles)
        else:
            title = random.choice(neutral_titles)

        score = int(rng.lognormal(mean=5.5, sigma=2.0))
        records.append({
            "post_id": f"syn_{i:05d}",
            "title": title,
            "selftext": "",
            "score": score,
            "num_comments": int(score * rng.uniform(0.1, 0.8)),
            "upvote_ratio": float(rng.uniform(0.55, 0.98)),
            "created_utc": int(date.astype("int64") // 10**9),
            "date": date,
            "flair": random.choices(flairs, weights=flair_weights, k=1)[0],
            "url": "",
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
