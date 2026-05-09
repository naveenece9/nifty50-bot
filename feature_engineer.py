"""
feature_engineer.py — Compute technical indicators, lag features, and labels
"""
import logging
import numpy as np
import pandas as pd
import ta
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, ATR_PERIOD, EMA_SHORT, EMA_LONG, EMA_200,
    LAG_DAYS, ROLLING_WINDOWS, SWING_TARGET_DAYS, SWING_TARGET_PCT,
)

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to OHLCV dataframe."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema_20"]  = ta.trend.ema_indicator(close, window=EMA_SHORT)
    df["ema_50"]  = ta.trend.ema_indicator(close, window=EMA_LONG)
    df["ema_200"] = ta.trend.ema_indicator(close, window=EMA_200)
    df["adx"]     = ta.trend.adx(high, low, close, window=14)
    df["adx_pos"] = ta.trend.adx_pos(high, low, close, window=14)
    df["adx_neg"] = ta.trend.adx_neg(high, low, close, window=14)

    # Price vs EMAs (relative position)
    df["price_vs_ema20"]  = (close - df["ema_20"])  / df["ema_20"]
    df["price_vs_ema50"]  = (close - df["ema_50"])  / df["ema_50"]
    df["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]
    df["ema20_vs_ema50"]  = (df["ema_20"] - df["ema_50"]) / df["ema_50"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi"]           = ta.momentum.rsi(close, window=RSI_PERIOD)
    df["rsi_overbought"]= (df["rsi"] > 70).astype(int)
    df["rsi_oversold"]  = (df["rsi"] < 30).astype(int)

    macd_obj = ta.trend.MACD(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["macd"]           = macd_obj.macd()
    df["macd_signal"]    = macd_obj.macd_signal()
    df["macd_diff"]      = macd_obj.macd_diff()
    df["macd_cross_up"]  = (
        (df["macd_diff"] > 0) & (df["macd_diff"].shift(1) <= 0)
    ).astype(int)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["williams_r"] = ta.momentum.williams_r(high, low, close)
    df["cci"]        = ta.trend.cci(high, low, close)
    df["roc_5"]      = ta.momentum.roc(close, window=5)
    df["roc_10"]     = ta.momentum.roc(close, window=10)

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=BB_PERIOD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = bb.bollinger_pband()   # % position within bands

    df["atr"]      = ta.volatility.average_true_range(high, low, close, ATR_PERIOD)
    df["atr_pct"]  = df["atr"] / close      # Normalised ATR

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]      = ta.volume.on_balance_volume(close, vol)
    df["obv_ema"]  = df["obv"].ewm(span=20).mean()
    df["obv_ratio"]= df["obv"] / df["obv_ema"].replace(0, np.nan)
    df["vol_ema20"]= vol.ewm(span=20).mean()
    df["vol_ratio"]= vol / df["vol_ema20"].replace(0, np.nan)  # Volume spike

    # VWAP (rolling 20-day approximation)
    df["vwap"] = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
    df["price_vs_vwap"] = (close - df["vwap"]) / df["vwap"]

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-lag and rolling statistical features."""
    df = df.copy()
    close = df["close"]

    for lag in LAG_DAYS:
        df[f"return_{lag}d"] = close.pct_change(lag)

    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}d"]   = close.rolling(w).mean() / close - 1
        df[f"roll_std_{w}d"]    = close.rolling(w).std()  / close
        df[f"roll_max_{w}d"]    = close.rolling(w).max()  / close - 1
        df[f"roll_min_{w}d"]    = close.rolling(w).min()  / close - 1
        df[f"vol_roll_{w}d"]    = df["volume"].rolling(w).mean()

    # Candle pattern features
    df["candle_body"]  = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_shadow"] = (df["high"]  - df[["open","close"]].max(axis=1)) / df["open"]
    df["lower_shadow"] = (df[["open","close"]].min(axis=1) - df["low"]) / df["open"]
    df["is_bullish"]   = (df["close"] > df["open"]).astype(int)

    # Gap feature
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary label: 1 if price rises >= SWING_TARGET_PCT% within
    SWING_TARGET_DAYS days, else 0.
    Also creates a 'future_return' column for reference.
    """
    df = df.copy()
    close = df["close"]

    # Max return over next N days
    df["future_return"] = (
        close.shift(-SWING_TARGET_DAYS).rolling(SWING_TARGET_DAYS).max()
        / close - 1
    ) * 100

    df["label"] = (df["future_return"] >= SWING_TARGET_PCT).astype(int)
    return df


def add_sentiment_column(df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """Add a single sentiment score as a constant column (today's score)."""
    df = df.copy()
    df["news_sentiment"] = sentiment
    return df


def add_fundamental_columns(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """Add fundamental ratios as constant columns."""
    df = df.copy()
    for key, val in fundamentals.items():
        df[key] = val
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    sentiment: float = 0.5,
    fundamentals: dict = None,
) -> pd.DataFrame:
    """Full pipeline: OHLCV → feature-rich DataFrame with labels."""
    if fundamentals is None:
        fundamentals = {"pe_ratio": 25.0, "pb_ratio": 3.0, "de_ratio": 1.0, "roe": 0.15}

    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_labels(df)
    df = add_sentiment_column(df, sentiment)
    df = add_fundamental_columns(df, fundamentals)

    # Drop rows with NaN (warmup period for indicators)
    df.dropna(inplace=True)

    logger.debug(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes raw OHLCV and label cols)."""
    exclude = {"open", "high", "low", "close", "volume", "ticker",
               "label", "future_return", "vwap", "bb_upper", "bb_lower",
               "bb_mid", "ema_20", "ema_50", "ema_200", "obv_ema", "vol_ema20"}
    return [c for c in df.columns if c not in exclude]
