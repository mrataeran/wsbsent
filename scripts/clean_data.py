#!/usr/bin/env python3
"""
scripts/clean_data.py
---------------------
Standalone script to scrape raw WSB posts, score sentiment, fetch price
data, and write cleaned output CSVs to the data/ directory.

Usage
-----
    python scripts/clean_data.py --pages 5 --start 2024-01-01 --out data/

"""

import argparse
import os
import sys
import pandas as pd

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wsbsent.collect import scrape_wsb
from wsbsent.sentiment import score_posts, aggregate_daily_sentiment
from wsbsent.prices import get_prices, get_returns


def parse_args():
    parser = argparse.ArgumentParser(description="WSBSent data pipeline")
    parser.add_argument("--sort", default="new", choices=["new", "hot", "top"])
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of Reddit API pages to fetch (100 posts/page)")
    parser.add_argument("--start", default="2024-01-01",
                        help="Price data start date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", default=["^GSPC", "^OEX"],
                        help="Yahoo Finance tickers to download")
    parser.add_argument("--out", default="data/",
                        help="Output directory for cleaned CSVs")
    parser.add_argument("--method", default="both",
                        choices=["vader", "textblob", "both"],
                        help="Sentiment method(s) to apply")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[1/4] Scraping r/wallstreetbets ({args.pages} pages, sort={args.sort})...")
    posts = scrape_wsb(sort=args.sort, pages=args.pages)
    raw_path = os.path.join(args.out, "wsb_raw.csv")
    posts.to_csv(raw_path, index=False)
    print(f"      Saved {len(posts)} posts → {raw_path}")

    if posts.empty:
        print("  Warning: no posts scraped. Run with --pages > 0 and valid network access.")
        print("  Tip: use wsbsent.collect.generate_sample_data() to generate test data.")
        return

    print(f"[2/4] Scoring sentiment (method={args.method})...")
    scored = score_posts(posts, method=args.method)
    scored_path = os.path.join(args.out, "wsb_scored.csv")
    scored.to_csv(scored_path, index=False)
    print(f"      Saved → {scored_path}")

    print("[3/4] Aggregating daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(scored)
    sent_path = os.path.join(args.out, "wsb_daily_sentiment.csv")
    daily_sentiment.to_csv(sent_path)
    print(f"      Saved {len(daily_sentiment)} days → {sent_path}")

    print(f"[4/4] Fetching price data for {args.tickers} from {args.start}...")
    prices = get_prices(tickers=args.tickers, start=args.start)
    returns = get_returns(prices)
    returns_path = os.path.join(args.out, "equity_returns.csv")
    returns.to_csv(returns_path)
    print(f"      Saved {len(returns)} rows → {returns_path}")

    print("\nDone. Files written to:", args.out)


if __name__ == "__main__":
    main()
