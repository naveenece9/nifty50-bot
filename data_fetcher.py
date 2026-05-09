"""
data_fetcher.py — Download OHLCV data and news sentiment for Nifty 50
"""
import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import (
    NIFTY50_TICKERS, NIFTY_INDEX_TICKER, NEWS_API_KEY,
    DATA_DIR, LOOKBACK_YEARS
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  OHLCV Data
# -----------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Download daily OHLCV for a single ticker."""
    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df["ticker"] = ticker
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_ohlcv(tickers: list = NIFTY50_TICKERS) -> dict:
    """Download OHLCV for all tickers. Returns dict {ticker: DataFrame}."""
    data = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching {ticker} ({i+1}/{len(tickers)})")
        df = fetch_ohlcv(ticker)
        if not df.empty:
            data[ticker] = df
        time.sleep(0.3)
    logger.info(f"Successfully fetched {len(data)}/{len(tickers)} tickers")
    return data


def fetch_nifty_index() -> pd.DataFrame:
    """Download Nifty 50 index data for market trend filter."""
    return fetch_ohlcv(NIFTY_INDEX_TICKER, period_years=1)


def save_ohlcv(data: dict):
    """Save each ticker's data as a parquet file."""
    for ticker, df in data.items():
        safe = ticker.replace(".", "_").replace("^", "")
        path = os.path.join(DATA_DIR, f"{safe}.parquet")
        df.to_parquet(path)
    logger.info(f"Saved {len(data)} ticker files to {DATA_DIR}")


def load_ohlcv(tickers: list = NIFTY50_TICKERS) -> dict:
    """Load previously saved parquet files."""
    data = {}
    for ticker in tickers:
        safe = ticker.replace(".", "_").replace("^", "")
        path = os.path.join(DATA_DIR, f"{safe}.parquet")
        if os.path.exists(path):
            data[ticker] = pd.read_parquet(path)
    return data


# -----------------------------------------------------------------------------
#  Ticker → Company name map (for news search)
# -----------------------------------------------------------------------------

TICKER_TO_NAME = {
    "ADANIENT.NS":   "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS":   "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BPCL.NS":       "BPCL",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS":  "Britannia",
    "CIPLA.NS":      "Cipla",
    "COALINDIA.NS":  "Coal India",
    "DIVISLAB.NS":   "Divi's Laboratories",
    "DRREDDY.NS":    "Dr Reddy",
    "EICHERMOT.NS":  "Eicher Motors",
    "GRASIM.NS":     "Grasim",
    "HCLTECH.NS":    "HCL Technologies",
    "HDFCBANK.NS":   "HDFC Bank",
    "HDFCLIFE.NS":   "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS":   "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS":  "ICICI Bank",
    "ITC.NS":        "ITC",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS":       "Infosys",
    "JSWSTEEL.NS":   "JSW Steel",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "LT.NS":         "Larsen Toubro",
    "M&M.NS":        "Mahindra",
    "MARUTI.NS":     "Maruti Suzuki",
    "NESTLEIND.NS":  "Nestle India",
    "NTPC.NS":       "NTPC",
    "ONGC.NS":       "ONGC",
    "POWERGRID.NS":  "Power Grid",
    "RELIANCE.NS":   "Reliance Industries",
    "SBILIFE.NS":    "SBI Life",
    "SBIN.NS":       "State Bank of India",
    "SUNPHARMA.NS":  "Sun Pharmaceutical",
    "TCS.NS":        "Tata Consultancy Services",
    "TATACONSUM.NS": "Tata Consumer",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS":  "Tata Steel",
    "TECHM.NS":      "Tech Mahindra",
    "TITAN.NS":      "Titan",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "UPL.NS":        "UPL",
    "WIPRO.NS":      "Wipro",
    "ZOMATO.NS":     "Zomato",
}

POSITIVE_WORDS = {
    "surge", "rally", "gain", "profit", "growth", "beat", "strong",
    "upgrade", "positive", "record", "bullish", "outperform", "rise",
    "jumped", "soared", "climbed", "higher", "buy", "excellent",
    "expansion", "wins", "award", "launches", "partnership", "dividend",
    "revenue", "acquisition", "boost", "breakthrough", "recover",
}

NEGATIVE_WORDS = {
    "fall", "drop", "loss", "decline", "miss", "weak", "downgrade",
    "negative", "bearish", "underperform", "crash", "concern", "risk",
    "slump", "tumble", "plunge", "lower", "sell", "fraud", "probe",
    "investigation", "penalty", "fine", "lawsuit", "debt", "defaults",
    "layoffs", "slowdown", "warning", "cut", "missed", "disappoints",
}


def _score_sentiment(articles: list) -> float:
    """Score sentiment from a list of article dicts."""
    pos, neg = 0, 0
    for article in articles:
        title = article.get("title") or ""
        desc  = article.get("description") or ""
        text  = (title + " " + desc).lower()
        words = set(text.split())
        pos  += len(words & POSITIVE_WORDS)
        neg  += len(words & NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.5
    return round(pos / (pos + neg), 4)


# -----------------------------------------------------------------------------
#  Primary: Yahoo Finance news (no API key needed)
# -----------------------------------------------------------------------------

def fetch_sentiment_from_yfinance(ticker: str) -> float | None:
    """
    Fetch news from Yahoo Finance (free, no API key).
    Returns sentiment score or None if no articles found.
    """
    try:
        t        = yf.Ticker(ticker)
        news     = t.news  # list of dicts with 'title', 'summary' etc.
        if not news:
            return None

        articles = [
            {"title": n.get("title", ""), "description": n.get("summary", "")}
            for n in news[:20]
        ]
        score = _score_sentiment(articles)
        logger.debug(f"{ticker} yfinance sentiment: {score} ({len(articles)} articles)")
        return score
    except Exception as e:
        logger.debug(f"yfinance news failed for {ticker}: {e}")
        return None


# -----------------------------------------------------------------------------
#  Fallback: NewsAPI
# -----------------------------------------------------------------------------

def fetch_sentiment_from_newsapi(ticker: str, days_back: int = 7) -> float | None:
    """
    Fetch sentiment from NewsAPI.
    Returns score or None if no articles / key not set.
    """
    if not NEWS_API_KEY:
        return None

    company   = TICKER_TO_NAME.get(ticker, ticker.replace(".NS", ""))
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Try multiple query terms to maximise article hits
    queries = [
        f'"{company}" stock NSE',
        f"{company} India shares",
        company,
    ]

    for query in queries:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        query,
                    "from":     from_date,
                    "language": "en",
                    "sortBy":   "relevancy",
                    "pageSize": 20,
                    "apiKey":   NEWS_API_KEY,
                },
                timeout=10,
            )
            data     = resp.json()
            articles = data.get("articles", [])

            # Skip if NewsAPI returned an error or no results
            if data.get("status") != "ok":
                logger.warning(f"NewsAPI error for {ticker}: {data.get('message')}")
                return None

            if articles:
                score = _score_sentiment(articles)
                logger.debug(f"{ticker} NewsAPI sentiment: {score} ({len(articles)} articles, query='{query}')")
                return score

        except Exception as e:
            logger.warning(f"NewsAPI request failed for {ticker}: {e}")
            return None

    return None  # All queries returned 0 articles


# -----------------------------------------------------------------------------
#  Public function: tries Yahoo first, falls back to NewsAPI
# -----------------------------------------------------------------------------

def fetch_news_sentiment(ticker: str, days_back: int = 7) -> float:
    """
    Get news sentiment for a ticker.
    1. Try Yahoo Finance news (free, no key needed)
    2. Fall back to NewsAPI if Yahoo returns nothing
    3. Default to 0.5 (neutral) if both fail
    """
    # Try Yahoo Finance first
    score = fetch_sentiment_from_yfinance(ticker)
    if score is not None:
        return score

    # Fall back to NewsAPI
    score = fetch_sentiment_from_newsapi(ticker, days_back)
    if score is not None:
        return score

    logger.info(f"{ticker}: no news found — using neutral sentiment 0.5")
    return 0.5


def fetch_sentiment_for_all(tickers: list = NIFTY50_TICKERS) -> dict:
    """Return dict {ticker: sentiment_score} for all tickers."""
    scores = {}
    for ticker in tickers:
        scores[ticker] = fetch_news_sentiment(ticker)
        time.sleep(0.2)
    return scores


# -----------------------------------------------------------------------------
#  Fundamentals (from yfinance)
# -----------------------------------------------------------------------------

def fetch_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental ratios from yfinance."""
    defaults = {"pe_ratio": 25.0, "pb_ratio": 3.0, "de_ratio": 1.0, "roe": 0.15}
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio": float(info.get("trailingPE")     or defaults["pe_ratio"]),
            "pb_ratio": float(info.get("priceToBook")    or defaults["pb_ratio"]),
            "de_ratio": float(info.get("debtToEquity")   or defaults["de_ratio"]),
            "roe":      float(info.get("returnOnEquity") or defaults["roe"]),
        }
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        return defaults