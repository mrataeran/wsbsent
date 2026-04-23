"""
collect.py
----------
Scrape posts and comments from r/wallstreetbets using Reddit's public JSON API.
No API credentials required — uses the public .json endpoint with a rate-limited
request loop.
"""

import time
import datetime
import requests
import pandas as pd
import numpy as np

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
BASE_URL = "https://www.reddit.com/r/wallstreetbets/{sort}.json"

_SAMPLE_TITLES = [
    "YOLO'd my life savings into TSLA calls — up 300% 🚀🚀🚀",
    "SPY puts printing, bears eating today",
    "Why NVDA is going to $1000 — my DD",
    "Lost $50k on GME this week, back to zero",
    "Fed pivot incoming? Loading up on QQQ calls",
    "AAPL earnings play — what's your thesis?",
    "This market is completely rigged, change my mind",
    "AMD going to crush earnings tomorrow, technical breakdown",
    "Closed my short position, taking small loss. Not worth it.",
    "PLTR to the moon, robots will rule the world",
    "Bear market incoming? Here's why I'm buying puts",
    "My MSFT call spread just doubled — staying in",
    "Sold everything Friday, cash gang until fed meeting",
    "Meme stocks back? AMC and GME spiking pre-market",
    "Rate cut bets crumbling — market overreaction or justified?",
    "Options expiration Friday — expect volatility",
    "First green day in 2 weeks, bulls finally showing up",
    "RIVN down 15%, loading more puts",
    "Bought the dip on COIN, already up 8%",
    "SPY hitting resistance at 520, watching for rejection",
]
_FLAIRS = ["DD", "Gain Porn", "Loss Porn", "Discussion", "Meme", "News", "YOLO", ""]


def scrape_wsb(
    sort: str = "new",
    pages: int = 10,
    sleep: float = 1.5,
) -> pd.DataFrame:
    """Scrape r/wallstreetbets posts via the public Reddit JSON API.

    Parameters
    ----------
    sort : str
        Feed to scrape. One of ``'new'``, ``'hot'``, ``'top'``.
    pages : int
        Number of paginated requests (each returns up to 100 posts).
        ``pages=10`` yields up to 1,000 posts.
    sleep : float
        Seconds to wait between requests to avoid rate-limiting.

    Returns
    -------
    pd.DataFrame
        Columns: ``post_id``, ``title``, ``selftext``, ``score``,
        ``num_comments``, ``upvote_ratio``, ``created_utc``, ``date``,
        ``flair``, ``url``.

    Examples
    --------
    >>> df = scrape_wsb(sort="new", pages=3)
    >>> df.shape
    (300, 10)
    """
    url = BASE_URL.format(sort=sort)
    params: dict = {"limit": 100}
    records = []
    after = None

    for page in range(pages):
        if after:
            params["after"] = after
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[collect] Request failed on page {page}: {exc}")
            break

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            created = post.get("created_utc", 0)
            records.append(
                {
                    "post_id": post.get("id", ""),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.5),
                    "created_utc": created,
                    "date": datetime.datetime.utcfromtimestamp(created).date()
                    if created
                    else None,
                    "flair": post.get("link_flair_text", ""),
                    "url": post.get("url", ""),
                }
            )

        after = data.get("after")
        if not after:
            break
        time.sleep(sleep)

    df = pd.DataFrame(records)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_sample_data(
    n_days: int = 90,
    posts_per_day: tuple = (40, 150),
    start: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic WSB post data for testing and demos.

    Produces a DataFrame with the same schema as :func:`scrape_wsb` so that
    all downstream modules work identically with synthetic or real data.

    Parameters
    ----------
    n_days : int
        Number of calendar days to simulate.
    posts_per_day : tuple
        ``(min, max)`` range for daily post counts.
    start : str
        Start date for the synthetic dataset.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic posts with realistic sentiment variation and volume patterns.
    """
    import random
    import string

    rng = random.Random(seed)
    np.random.seed(seed)

    start_dt = pd.Timestamp(start)
    records = []

    for day_offset in range(n_days):
        date = start_dt + pd.Timedelta(days=day_offset)
        n_posts = rng.randint(*posts_per_day)
        # Inject a sentiment trend: slight drift + random noise
        day_bias = np.sin(day_offset / 15) * 0.1  # gentle cyclic bias

        for _ in range(n_posts):
            title_idx = rng.randint(0, len(_SAMPLE_TITLES) - 1)
            base_title = _SAMPLE_TITLES[title_idx]
            # Add ticker noise
            tickers = ["TSLA", "AAPL", "NVDA", "SPY", "QQQ", "AMD", "MSFT", "GME"]
            title = base_title.replace(
                rng.choice(["TSLA", "NVDA", "AAPL"]),
                rng.choice(tickers),
            )
            score = max(0, int(np.random.lognormal(5, 1.5)))
            uid = "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))
            ts = date.timestamp() + rng.randint(0, 86399)
            records.append(
                {
                    "post_id": uid,
                    "title": title,
                    "selftext": "",
                    "score": score,
                    "num_comments": max(0, int(score * rng.uniform(0.1, 0.8))),
                    "upvote_ratio": round(rng.uniform(0.55, 0.98), 2),
                    "created_utc": ts,
                    "date": date.normalize(),
                    "flair": rng.choice(_FLAIRS),
                    "url": f"https://reddit.com/r/wallstreetbets/comments/{uid}",
                }
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


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
        "This rally is just getting started", "Bought the dip — see you at ATH",
        "GME short interest is insane right now", "Fed pivot = infinite money glitch",
        "TSLA breaking out of the wedge", "Why I'm all-in on tech this quarter",
        "S&P 500 hit new ATH — bears destroyed", "Just turned $5k into $50k on options",
        "Market is pricing in too much fear", "Rotation into growth is happening NOW",
        "AMD earnings gonna send us to Mars", "Bull thesis for MSFT — long read but worth it",
        "Options flow is screaming bullish", "My 5-bagger play for Q4",
    ]
    bear_titles = [
        "Bear case for SPY — change my mind", "This market is in a bubble",
        "Hedging with puts before FOMC", "GDP miss incoming — stay cautious",
        "Sold everything, going to cash", "Why I think we revisit October lows",
        "Credit spreads are flashing red", "Inverse ETFs printing today",
        "VIX spike incoming — watch out", "This rally has no volume behind it",
        "Recession fears are real this time", "Took profits, market looks toppy",
    ]
    neutral_titles = [
        "DD: deep dive into earnings preview", "What's your play for this week?",
        "WSB daily discussion thread", "Positions update — show me yours",
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
