"""
signal_generator.py — Generate balanced buy/sell signals for Nifty 50
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime

from config import (
    NIFTY50_TICKERS, SIGNAL_THRESHOLD,
    MAX_SIGNALS_PER_RUN, SIGNAL_DIR,
)
from data_fetcher import (
    fetch_ohlcv, fetch_news_sentiment,
    fetch_fundamentals, fetch_nifty_index,
)
from feature_engineer import build_feature_matrix
from model_trainer import load_model

logger = logging.getLogger(__name__)

# Thresholds — adjusted for balance
BUY_THRESHOLD  = 0.60   # Lower than before to get more BUY signals
SELL_THRESHOLD = 0.45   # Higher than before to get more SELL signals


def is_market_bullish() -> bool:
    """Returns True unless Nifty is more than 1% below its 50-day EMA."""
    try:
        df = fetch_nifty_index()
        if df.empty or len(df) < 50:
            return True
        close      = df["close"]
        ema50      = close.ewm(span=50).mean()
        last_close = float(close.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        gap_pct    = (last_close - last_ema50) / last_ema50 * 100
        bullish    = gap_pct > -1.0
        logger.info(f"Market: {'BULLISH' if bullish else 'BEARISH'} "
                    f"(Nifty {last_close:.0f} vs EMA50 {last_ema50:.0f}, gap={gap_pct:+.2f}%)")
        return bullish
    except Exception as e:
        logger.warning(f"Market filter failed: {e}")
        return True


def is_overextended(df: pd.DataFrame, threshold_pct: float = 7.0) -> bool:
    """True if stock rose more than threshold% in last 3 days."""
    if len(df) < 4:
        return False
    recent_return = (df["close"].iloc[-1] / df["close"].iloc[-4] - 1) * 100
    return bool(recent_return > threshold_pct)


def make_serializable(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def calculate_targets(df: pd.DataFrame, action: str, close_price: float, atr: float) -> dict:
    """Calculate dynamic target and stop-loss using recent swing highs/lows."""
    recent     = df.tail(20)
    swing_high = float(recent["high"].max())
    swing_low  = float(recent["low"].min())

    if action == "BUY":
        stop_loss = round(max(swing_low - atr * 0.3, close_price - atr * 1.5), 2)
        target    = round(max(swing_high + atr * 0.3, close_price + atr * 2.0), 2)
    else:
        stop_loss = round(min(swing_high + atr * 0.3, close_price + atr * 1.5), 2)
        target    = round(min(swing_low - atr * 0.3, close_price - atr * 2.0), 2)

    risk        = abs(close_price - stop_loss)
    reward      = abs(target - close_price)
    risk_reward = round(reward / max(risk, 0.01), 2)

    return {
        "target":      target,
        "stop_loss":   stop_loss,
        "risk_reward": risk_reward,
        "swing_high":  round(swing_high, 2),
        "swing_low":   round(swing_low, 2),
        "atr":         round(atr, 2),
    }


def score_ticker(ticker, model, scaler, feature_cols) -> dict | None:
    """Score a single ticker and return signal dict."""
    df = fetch_ohlcv(ticker, period_years=1)
    if df.empty or len(df) < 220:
        logger.warning(f"{ticker}: insufficient data ({len(df)} rows)")
        return None

    sentiment    = fetch_news_sentiment(ticker)
    fundamentals = fetch_fundamentals(ticker)

    try:
        feat_df = build_feature_matrix(df, sentiment, fundamentals)
    except Exception as e:
        logger.error(f"{ticker}: feature build failed - {e}")
        return None

    for col in set(feature_cols) - set(feat_df.columns):
        feat_df[col] = 0.0

    X_today  = feat_df[feature_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X_today)
    proba    = float(model.predict_proba(X_scaled)[0, 1])

    last        = df.iloc[-1]
    close_price = float(last["close"])
    atr         = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else close_price * 0.015
    levels      = calculate_targets(df, "BUY", close_price, atr)

    return {
        "ticker":         ticker,
        "proba":          proba,
        "close":          round(close_price, 2),
        "volume":         int(last["volume"]),
        "news_sentiment": round(float(sentiment), 3),
        "date":           str(df.index[-1].date()),
        "generated_at":   datetime.now().isoformat(),
        "pe_ratio":       float(fundamentals.get("pe_ratio", 25.0)),
        "atr":            round(atr, 2),
        "swing_high":     levels["swing_high"],
        "swing_low":      levels["swing_low"],
        "df":             df,  # keep for target calc
    }


def run_daily_signals() -> list:
    """
    Score all tickers, then pick:
    - Top 3 highest proba as BUY (if proba >= BUY_THRESHOLD)
    - Top 2 lowest proba as SELL (if proba <= SELL_THRESHOLD)
    Always tries to return a mix of BUY and SELL signals.
    """
    logger.info("=" * 60)
    logger.info(f"Daily signal run - {date.today()}")
    logger.info("=" * 60)

    model, scaler, feature_cols = load_model()
    market_bullish = is_market_bullish()

    # Score all tickers
    all_scores = []
    for ticker in NIFTY50_TICKERS:
        logger.info(f"Scoring {ticker}...")
        result = score_ticker(ticker, model, scaler, feature_cols)
        if result:
            all_scores.append(result)

    if not all_scores:
        logger.error("No scores generated!")
        return []

    # Sort by probability
    all_scores.sort(key=lambda x: x["proba"], reverse=True)

    # Log distribution
    probas = [s["proba"] for s in all_scores]
    logger.info(f"Prob distribution - Max:{max(probas):.3f} Min:{min(probas):.3f} "
                f"Mean:{sum(probas)/len(probas):.3f}")

    # Pick top BUYs
    buy_candidates = [
        s for s in all_scores
        if s["proba"] >= BUY_THRESHOLD
        and not (market_bullish is False)  # skip BUYs if clearly bearish
        and not is_overextended(s["df"])
    ]

    # Pick top SELLs (lowest probability)
    sell_candidates = [
        s for s in reversed(all_scores)
        if s["proba"] <= SELL_THRESHOLD
    ]

    # If no BUYs found above threshold, take top 2 by probability anyway
    if not buy_candidates:
        logger.info("No BUYs above threshold - taking top 2 by probability")
        buy_candidates = all_scores[:2]

    # If no SELLs found below threshold, take bottom 2 by probability anyway
    if not sell_candidates:
        logger.info("No SELLs below threshold - taking bottom 2 by probability")
        sell_candidates = list(reversed(all_scores))[:2]

    # Build final signal list: 3 BUY + 2 SELL
    top_buys  = buy_candidates[:3]
    top_sells = sell_candidates[:2]

    signals = []

    for s in top_buys:
        df  = s.pop("df")
        atr = s["atr"]
        levels = calculate_targets(df, "BUY", s["close"], atr)
        signals.append({
            **{k: v for k, v in s.items() if k != "proba"},
            "action":         "BUY",
            "confidence_pct": round(s["proba"] * 100, 1),
            "probability":    round(s["proba"], 4),
            "market_bullish": bool(market_bullish),
            **levels,
        })

    for s in top_sells:
        df  = s.pop("df") if "df" in s else None
        atr = s["atr"]
        if df is not None:
            levels = calculate_targets(df, "SELL", s["close"], atr)
        else:
            levels = {"target": 0, "stop_loss": 0, "risk_reward": 0,
                      "swing_high": 0, "swing_low": 0, "atr": atr}
        signals.append({
            **{k: v for k, v in s.items() if k != "proba"},
            "action":         "SELL",
            "confidence_pct": round((1 - s["proba"]) * 100, 1),
            "probability":    round(s["proba"], 4),
            "market_bullish": bool(market_bullish),
            **levels,
        })

    _save_signals(signals)

    buy_count  = sum(1 for s in signals if s["action"] == "BUY")
    sell_count = sum(1 for s in signals if s["action"] == "SELL")
    logger.info(f"Final signals: {len(signals)} total | {buy_count} BUY | {sell_count} SELL")
    return signals


def _save_signals(signals: list):
    path = os.path.join(SIGNAL_DIR, f"signals_{date.today()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(signals), f, indent=2)
    logger.info(f"Signals saved: {path}")
