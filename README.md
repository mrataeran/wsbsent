# wsbsent 📈

**Sentiment analysis of r/wallstreetbets and equity return prediction.**

A Python package that scrapes WSB posts, scores them with VADER and TextBlob,
and investigates whether retail investor sentiment carries predictive signal
for S&P 100 and S&P 500 returns.

Built for STAT 386 Final Project, Winter 2026 · Brigham Young University
by @mrataeran & @vaoikun
---

## Installation

Install directly from GitHub (no credentials required):

```bash
pip install git+https://github.com/mrataeran/wsbsent.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/mrataeran/wsbsent.git
cd wsbsent
pip install -e ".[dev]"
```

---

## Quick Start

```python
from wsbsent import (
    scrape_wsb, score_posts, aggregate_daily_sentiment,
    get_prices, get_returns,
    merge_sentiment_prices, build_features,
    lagged_correlation, classify_direction,
    plot_sentiment_returns, plot_correlation_heatmap,
)

# 1. Collect WSB posts (no API key needed)
posts = scrape_wsb(sort="new", pages=5)

# 2. Score sentiment
scored = score_posts(posts, method="both")
daily = aggregate_daily_sentiment(scored)

# 3. Fetch equity data
prices = get_prices(tickers=["^GSPC", "^OEX"], start="2024-01-01")
returns = get_returns(prices)

# 4. Merge and analyze
merged = merge_sentiment_prices(daily, returns)
lag_df = lagged_correlation(merged, sentiment_col="vader_compound", return_col="^GSPC")

# 5. Classify direction
features = build_features(merged)
results = classify_direction(features)
print(f"Test accuracy: {results['accuracy']:.1%}")

# 6. Visualize
fig = plot_sentiment_returns(merged)
fig.savefig("sentiment_vs_returns.png", dpi=150)
```

---

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app provides:
- Date range and ticker selector
- VADER / TextBlob / Both model toggle
- Dual-axis sentiment vs. cumulative return chart
- Lag correlation bar chart with significance markers
- Confusion matrix and feature importance
- Raw data explorer
- Flair distribution breakdown

---

## Data Pipeline

```bash
python scripts/clean_data.py --pages 10 --start 2024-01-01 --out data/
```

Writes four CSVs to `data/`:
| File | Contents |
|---|---|
| `wsb_raw.csv` | Raw scraped posts |
| `wsb_scored.csv` | Posts with sentiment scores |
| `wsb_daily_sentiment.csv` | Daily aggregated sentiment |
| `equity_returns.csv` | Daily log-returns |

---

## Documentation

Full documentation, tutorial, and written report:
**[mrataeran.github.io/wsbsent](https://mrataeran.github.io/wsbsent)**

---

## Project Structure

```
wsbsent/
├── wsbsent/
│   ├── __init__.py
│   ├── collect.py       # Reddit public JSON API scraper
│   ├── prices.py        # yfinance price/return fetcher
│   ├── sentiment.py     # VADER + TextBlob scoring
│   ├── wrangling.py     # Merge + feature engineering
│   ├── analysis.py      # Lag correlation + LR classifier
│   └── visualize.py     # Matplotlib/seaborn plotting
├── app/
│   └── streamlit_app.py # Interactive dashboard
├── scripts/
│   └── clean_data.py    # CLI data pipeline
├── docs/                # Quarto documentation + GitHub Pages
│   ├── index.qmd
│   ├── tutorial.qmd
│   └── report.qmd
├── data/                # Output CSVs (gitignored)
├── pyproject.toml
└── README.md
```

---

## License

MIT License · Ata Raeran · 2026
