"""
app/streamlit_app.py
---------------------
Interactive dashboard: WSB sentiment vs. equity return prediction.

Run locally:
    streamlit run app/streamlit_app.py

Deploy to Streamlit Cloud:
    Point to this file from the repo root.
"""

import sys
import os
import datetime

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Allow import from repo root without full install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wsbsent.collect import scrape_wsb, generate_sample_data
from wsbsent.prices import get_prices, get_returns, generate_sample_prices, INDEX_TICKERS, SP100_TICKERS
from wsbsent.sentiment import score_posts, aggregate_daily_sentiment
from wsbsent.wrangling import merge_sentiment_prices, build_features
from wsbsent.analysis import lagged_correlation, classify_direction
from wsbsent.visualize import (
    plot_sentiment_returns,
    plot_correlation_heatmap,
    plot_confusion,
    plot_feature_importance,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="WSBSent, Sentiment & Equity Return Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 700; color: #2c3e50; }
    .sub-title  { font-size: 1rem; color: #7f8c8d; margin-top: -10px; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px 20px; margin: 4px 0;
        border-left: 4px solid #3498db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/wsbsent-v0.1.0-blue",
        use_column_width=False,
    )
    st.markdown("## ⚙️ Controls")

    st.markdown("### 📅 Date Range")
    today = datetime.date.today()
    start_date = st.date_input("Start", value=datetime.date(2024, 1, 1))
    end_date = st.date_input("End", value=today)

    st.markdown("### 📊 Equity Target")
    ticker_options = ["^GSPC (S&P 500)", "^OEX (S&P 100)"] + [
        t for t in SP100_TICKERS[:20] if t not in ("^GSPC", "^OEX")
    ]
    ticker_choice = st.selectbox("Ticker", ticker_options, index=0)
    ticker = ticker_choice.split(" ")[0]

    st.markdown("### 🧠 Sentiment Model")
    method = st.radio("Method", ["VADER", "TextBlob", "Both"], index=0)
    method_key = method.lower().replace(" ", "")

    sentiment_map = {
        "VADER": "vader_compound",
        "TextBlob": "tb_polarity",
        "Both": "vader_compound",
    }
    sent_col = sentiment_map[method]

    st.markdown("### 🔄 Data Source")
    n_pages = st.slider("Reddit pages (100 posts/page)", 1, 20, 10)
    st.caption("💡 10+ pages recommended for real data, more pages = wider date coverage.")
    sort_mode = st.selectbox("Sort", ["new", "hot", "top"], index=0)

    demo_mode = st.checkbox("🧪 Demo mode (synthetic data)", value=False,
                           help="Use generated data if Reddit API is unavailable")
    fetch_btn = st.button("🚀 Fetch & Analyze", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**wsbsent** · [GitHub](https://github.com/mrataeran/wsbsent) · "
        "STAT 386 Final Project"
    )

# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────
st.markdown('<p class="main-title">📈 WSBSent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Sentiment Analysis of r/wallstreetbets '
    "and Equity Return Prediction</p>",
    unsafe_allow_html=True,
)

st.info(
    "Configure the controls in the sidebar, then press **Fetch & Analyze** "
    "to run the full pipeline.",
    icon="ℹ️",
)

# ──────────────────────────────────────────────
# Session state cache
# ──────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None


@st.cache_data(show_spinner=False)
def run_pipeline(sort, pages, start, end, ticker, method, demo=False):
    """Full pipeline: scrape → sentiment → prices → analysis."""
    # 1. Posts
    if demo:
        posts = generate_sample_data(500, start=str(start), end=str(end))
    else:
        posts = scrape_wsb(sort=sort, pages=pages)

    # 2. Sentiment
    m = method.lower().replace(" ", "")
    if m == "both":
        m = "both"
    elif m == "vader":
        m = "vader"
    else:
        m = "textblob"

    if posts.empty or "title" not in posts.columns:
        raise RuntimeError(
            "Reddit scraping returned no posts. Reddit may be blocking this network. "
            "Enable Demo mode in the sidebar to test with synthetic data instead."
        )

    scored = score_posts(posts, method=m)
    daily_sent = aggregate_daily_sentiment(scored)

    # 3. Prices, align date range to actual post coverage for maximum overlap
    tickers_to_fetch = list({ticker, "^GSPC"})
    if not posts.empty and "date" in posts.columns:
        price_start = str(posts["date"].min().date())
        price_end   = str(posts["date"].max().date())
    else:
        price_start, price_end = str(start), str(end)

    if demo:
        prices = generate_sample_prices(tickers_to_fetch, start=price_start, end=price_end)
    else:
        prices = get_prices(tickers=tickers_to_fetch, start=price_start, end=price_end)
    returns = get_returns(prices)

    # 4. Merge (outer so we keep all sentiment days, NaN where market closed)
    merged = merge_sentiment_prices(daily_sent, returns, how="outer")
    # Forward-fill returns over weekends then drop days with no return data
    ret_cols = [c for c in merged.columns if c in tickers_to_fetch]
    merged[ret_cols] = merged[ret_cols].ffill()
    merged = merged.dropna(subset=ret_cols)

    # 5. Lag correlation, use adaptive max_lag so small datasets still work
    sent_col_map = {"vader": "vader_compound", "textblob": "tb_polarity", "both": "vader_compound"}
    sc = sent_col_map.get(m, "vader_compound")
    if sc not in merged.columns:
        sc = [c for c in merged.columns if c.startswith("vader_") or c.startswith("tb_")][0]

    target_col = ticker if ticker in merged.columns else "^GSPC"
    n_obs = len(merged)
    max_lag = min(10, max(1, n_obs // 5))   # adaptive: at least 5 obs per lag tested

    lag_df = None
    clf_results = None
    if target_col in merged.columns and sc in merged.columns:
        lag_df = lagged_correlation(merged, sentiment_col=sc, return_col=target_col, max_lag=max_lag)
        try:
            features = build_features(merged, target_col=target_col, lags=[1, 2, 3], rolling_windows=[3])
            if len(features) >= 20:
                clf_results = classify_direction(features, target_col="target")
        except Exception:
            pass

    return {
        "posts": posts,
        "scored": scored,
        "daily_sent": daily_sent,
        "merged": merged,
        "lag_df": lag_df,
        "clf_results": clf_results,
        "sent_col": sc,
        "target_col": target_col,
        "source": "live",
    }


if fetch_btn:
    with st.spinner("Fetching Reddit posts and market data…"):
        try:
            st.session_state.results = run_pipeline(
                sort_mode, n_pages, start_date, end_date, ticker, method, demo=demo_mode
            )
            st.success("Pipeline complete!", icon="✅")
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# ──────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────
res = st.session_state.results

if res is not None:
    merged = res["merged"]
    daily_sent = res["daily_sent"]
    lag_df = res["lag_df"]
    clf_results = res["clf_results"]
    sc = res["sent_col"]
    tc = res["target_col"]

    # ── Key metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Posts Collected", f"{len(res['posts']):,}")
    with col2:
        if sc in daily_sent.columns:
            avg_sent = daily_sent[sc].mean()
            st.metric("Avg. Daily Sentiment", f"{avg_sent:.3f}")
    with col3:
        if tc in merged.columns:
            total_ret = merged[tc].sum() * 100
            st.metric("Cumulative Return (%)", f"{total_ret:.2f}%")
    with col4:
        if clf_results:
            st.metric("Classifier Accuracy", f"{clf_results['accuracy']:.1%}")

    # Source banner
    if res.get("source") == "preloaded":
        st.warning("Arctic Shift API unavailable from this server. Showing preloaded dataset collected locally.", icon="📦")

    # Coverage info banner
    if not merged.empty:
        st.info(
            f"📅 **Data coverage:** {merged.index.min().date()} → {merged.index.max().date()}  "
            f"| **{len(merged)} overlapping days**  "
            f"| **{len(res['posts'])} posts scraped**",
            icon=None,
        )
    st.markdown("---")

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Sentiment vs Returns", "🔗 Lag Correlation",
         "🤖 Classification", "📋 Raw Data", "📄 Flair Breakdown"]
    )

    with tab1:
        if sc in merged.columns and tc in merged.columns:
            fig = plot_sentiment_returns(merged, sentiment_col=sc, return_col=tc)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Not enough overlapping data to plot.")

    with tab2:
        if lag_df is not None and not lag_df.empty:
            fig2 = plot_correlation_heatmap(lag_df)
            st.pyplot(fig2)
            plt.close(fig2)
            sig = lag_df[lag_df["significant"]]
            if not sig.empty:
                st.dataframe(sig, use_container_width=True)
            else:
                st.info("No statistically significant lags found (p < 0.05).")
        else:
            st.warning("Not enough overlapping data for lag correlation. Try a wider date range or more pages.")

    with tab3:
        if clf_results:
            c1, c2 = st.columns(2)
            with c1:
                fig3 = plot_confusion(clf_results["confusion"], clf_results["accuracy"])
                st.pyplot(fig3)
                plt.close(fig3)
            with c2:
                fig4 = plot_feature_importance(clf_results["feature_importance"])
                st.pyplot(fig4)
                plt.close(fig4)
            report_df = pd.DataFrame(clf_results["report"]).T
            st.dataframe(report_df.round(3), use_container_width=True)
        else:
            st.warning("Classification failed, insufficient data.")

    with tab4:
        st.subheader("Daily Sentiment")
        st.dataframe(daily_sent.round(4), use_container_width=True)
        st.subheader("Merged Dataset (Sentiment + Returns)")
        st.dataframe(merged.round(4), use_container_width=True)
        if lag_df is not None:
            st.subheader("Lag Correlation Table")
            st.dataframe(lag_df, use_container_width=True)

    with tab5:
        if "flair" in res["posts"].columns:
            flair_counts = (
                res["posts"]["flair"]
                .fillna("(none)")
                .value_counts()
                .rename_axis("flair")
                .reset_index(name="count")
            )
            st.bar_chart(flair_counts.set_index("flair"))
        else:
            st.info("No flair data available.")
