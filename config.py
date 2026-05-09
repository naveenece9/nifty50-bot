"""
config.py — Central configuration for Nifty 50 Signal Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── NewsAPI ───────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Signal settings ───────────────────────────────────────────────────────────
SIGNAL_THRESHOLD      = float(os.getenv("SIGNAL_THRESHOLD", 0.65))
SWING_TARGET_DAYS     = int(os.getenv("SWING_TARGET_DAYS", 5))
SWING_TARGET_PCT      = float(os.getenv("SWING_TARGET_PCT", 2.0))
MAX_SIGNALS_PER_RUN   = int(os.getenv("MAX_SIGNALS_PER_RUN", 5))
MARKET_CLOSE_HOUR     = int(os.getenv("MARKET_CLOSE_HOUR", 16))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
SIGNAL_DIR  = os.path.join(BASE_DIR, "signals")

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, SIGNAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── All 50 Nifty 50 tickers (Yahoo Finance format) ────────────────────────────
NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS",  "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS",      "BHARTIARTL.NS", "BRITANNIA.NS",  "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS",   "DRREDDY.NS",    "EICHERMOT.NS",
    "GRASIM.NS",    "HCLTECH.NS",    "HDFCBANK.NS",   "HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS",   "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS",       "INDUSINDBK.NS", "INFY.NS",       "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS",         "M&M.NS",        "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS",       "ONGC.NS",       "POWERGRID.NS",
    "RELIANCE.NS",  "SBILIFE.NS",    "SBIN.NS",       "SUNPHARMA.NS",
    "TCS.NS",       "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS",     "TITAN.NS",      "ULTRACEMCO.NS", "UPL.NS",
    "WIPRO.NS",     "ZOMATO.NS",
]

# ── Nifty 50 index (for market trend filter) ──────────────────────────────────
NIFTY_INDEX_TICKER = "^NSEI"

# ── Feature settings ──────────────────────────────────────────────────────────
LOOKBACK_YEARS    = 2       # Years of historical data for training
RSI_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_PERIOD         = 20
ATR_PERIOD        = 14
EMA_SHORT         = 20
EMA_LONG          = 50
EMA_200           = 200
LAG_DAYS          = [1, 3, 5, 10]      # Return lag windows
ROLLING_WINDOWS   = [5, 10, 20]        # Rolling stat windows
