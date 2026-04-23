"""
wsbsent: Sentiment analysis of r/wallstreetbets and equity return prediction.
"""

from .collect import scrape_wsb, load_posts, generate_sample_data, generate_sample_data
from .prices import get_prices, get_returns, generate_sample_prices, generate_sample_prices
from .sentiment import score_posts, aggregate_daily_sentiment
from .wrangling import merge_sentiment_prices, build_features
from .analysis import lagged_correlation, classify_direction
from .visualize import plot_sentiment_returns, plot_correlation_heatmap, plot_confusion

__version__ = "0.1.0"
__author__ = "Ata Raeran"
