"""
signal_generator.py — Generate daily buy/sell signals for all Nifty 50 stocks
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime

from config import (
    NIFTY50_TICKERS, NIFTY_INDEX_TICKER, SIGNAL_THRESHOLD,
    MAX_SIGNALS_PER_RUN, SIGNAL_DIR,
)
from data_fetcher import (
    fetch_ohlcv, fetch_news_sentiment,
    fetch_fundamentals, fetch_nifty_index,
)
from feature_engineer import build_feature_matrix, get_feature_columns
from model_trainer import load_model

logger = logging.getLogger(__name__)

# SELL fires when prob is below this
SELL_THRESHOLD = 0.50


def is_market_bullish() -> bool:
    """
    Returns True unless Nifty is more than 1% below its 50-day EMA.
    Small gaps (< 1%) are treated as neutral/flat, not bearish.
    """
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
        if bullish:
            logger.info(f"Market: BULLISH (Nifty {last_close:.0f} vs EMA50 {last_ema50:.0f}, gap={gap_pct:+.2f}%)")
        else:
            logger.info(f"Market: BEARISH (Nifty {last_close:.0f} vs EMA50 {last_ema50:.0f}, gap={gap_pct:+.2f}%) - BUYs blocked")
        return bullish
    except Exception as e:
        logger.warning(f"Market filter failed: {e} - defaulting to allowed")
        return True


def is_overextended(df: pd.DataFrame, threshold_pct: float = 5.0) -> bool:
    """True if stock rose more than threshold% in last 3 days (avoid chasing)."""
    if len(df) < 4:
        return False
    recent_return = (df["close"].iloc[-1] / df["close"].iloc[-4] - 1) * 100
    return bool(recent_return > threshold_pct)


def make_serializable(obj):
    """Recursively convert numpy/bool types to native Python for JSON."""
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
    """
    Calculate dynamic target and stop-loss using recent swing highs/lows.
    Each stock gets unique levels based on its own price structure.
    """
    recent     = df.tail(20)
    swing_high = float(recent["high"].max())
    swing_low  = float(recent["low"].min())

    if action == "BUY":
        # Stop: below recent swing low (but not too far)
        stop_by_swing = swing_low - atr * 0.3
        stop_by_atr   = close_price - atr * 1.5
        stop_loss     = round(max(stop_by_swing, stop_by_atr), 2)

        # Target: above recent swing high (or at least 2x ATR away)
        target_by_swing = swing_high + atr * 0.3
        target_by_atr   = close_price + atr * 2.0
        target          = round(max(target_by_swing, target_by_atr), 2)

    else:  # SELL
        # Stop: above recent swing high
        stop_by_swing = swing_high + atr * 0.3
        stop_by_atr   = close_price + atr * 1.5
        stop_loss     = round(min(stop_by_swing, stop_by_atr), 2)

        # Target: below recent swing low
        target_by_swing = swing_low - atr * 0.3
        target_by_atr   = close_price - atr * 2.0
        target          = round(min(target_by_swing, target_by_atr), 2)

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


def generate_signal_for_ticker(ticker, model, scaler, feature_cols, market_bullish):
    """
    Generate a signal dict for a single ticker.
    Returns None if confidence is in uncertain zone or filters block it.
    """
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

    # Fill any missing features with 0
    for col in set(feature_cols) - set(feat_df.columns):
        feat_df[col] = 0.0

    X_today  = feat_df[feature_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X_today)
    proba    = float(model.predict_proba(X_scaled)[0, 1])

    # Determine action using separate BUY / SELL thresholds
    if proba >= SIGNAL_THRESHOLD:
        action = "BUY"
    elif proba < SELL_THRESHOLD:
        action = "SELL"
    else:
        # Uncertain middle zone — skip
        logger.info(f"{ticker}: skipped (prob={proba:.3f} in uncertain zone {SELL_THRESHOLD}-{SIGNAL_THRESHOLD})")
        return None

    # Apply market regime filter for BUYs
    if action == "BUY" and not market_bullish:
        logger.info(f"{ticker}: BUY blocked - market clearly bearish (>1% below EMA50)")
        return None

    # Avoid chasing overextended stocks
    if action == "BUY" and is_overextended(df):
        logger.info(f"{ticker}: BUY blocked - overextended (>5% in 3 days)")
        return None

    last        = df.iloc[-1]
    close_price = float(last["close"])
    atr         = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else close_price * 0.015

    # Calculate dynamic target/stop based on swing levels
    levels = calculate_targets(df, action, close_price, atr)

    signal = {
        "ticker":         ticker,
        "action":         action,
        "probability":    round(proba, 4),
        "confidence_pct": round(proba * 100, 1),
        "close":          round(close_price, 2),
        "volume":         int(last["volume"]),
        "news_sentiment": round(float(sentiment), 3),
        "date":           str(df.index[-1].date()),
        "generated_at":   datetime.now().isoformat(),
        "pe_ratio":       float(fundamentals.get("pe_ratio", 25.0)),
        "market_bullish": bool(market_bullish),
        **levels,
    }

    logger.info(
        f"{ticker}: {action} | prob={proba:.3f} | close={close_price:.2f} "
        f"| target={levels['target']} | sl={levels['stop_loss']} | RR={levels['risk_reward']}"
    )
    return signal


def run_daily_signals() -> list:
    """
    Main function: score all Nifty 50 stocks and return top signals.
    """
    logger.info("=" * 60)
    logger.info(f"Daily signal run - {date.today()}")
    logger.info("=" * 60)

    model, scaler, feature_cols = load_model()
    market_bullish = is_market_bullish()

    all_signals = []
    for ticker in NIFTY50_TICKERS:
        logger.info(f"Scoring {ticker}...")
        signal = generate_signal_for_ticker(
            ticker, model, scaler, feature_cols, market_bullish
        )
        if signal:
            all_signals.append(signal)

    # Sort: BUY by highest confidence, SELL by lowest prob (most bearish first)
    buy_signals  = sorted(
        [s for s in all_signals if s["action"] == "BUY"],
        key=lambda x: x["probability"], reverse=True
    )
    sell_signals = sorted(
        [s for s in all_signals if s["action"] == "SELL"],
        key=lambda x: x["probability"]
    )

    # Take top BUYs and top SELLs up to MAX_SIGNALS_PER_RUN
    half        = MAX_SIGNALS_PER_RUN // 2
    top_signals = buy_signals[:half + 1] + sell_signals[:half]
    top_signals = top_signals[:MAX_SIGNALS_PER_RUN]

    _save_signals(top_signals)

    logger.info(f"Done: {len(top_signals)} signals | {len(buy_signals)} BUY | {len(sell_signals)} SELL")
    return top_signals


def _save_signals(signals: list):
    """Persist today's signals as JSON."""
    path = os.path.join(SIGNAL_DIR, f"signals_{date.today()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(signals), f, indent=2)
    logger.info(f"Signals saved: {path}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run signal generation")
    args = parser.parse_args()

    if args.run:
        signals = run_daily_signals()
        for s in signals:
            print(f"\n{s['action']} {s['ticker']} | conf={s['confidence_pct']}% | "
                  f"close={s['close']} | target={s['target']} | sl={s['stop_loss']} | RR=1:{s['risk_reward']}")
    else:
        parser.print_help()