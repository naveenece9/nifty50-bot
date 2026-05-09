"""
backtester.py — Evaluate signal quality on held-out historical data
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from config import NIFTY50_TICKERS, MODEL_DIR, SIGNAL_THRESHOLD, SWING_TARGET_DAYS
from data_fetcher import fetch_ohlcv, fetch_fundamentals
from feature_engineer import build_feature_matrix, get_feature_columns
from model_trainer import load_model

logger = logging.getLogger(__name__)


def backtest_ticker(ticker: str, model, scaler, feature_cols: list) -> pd.DataFrame | None:
    """
    Run model over 6-month held-out window for a single ticker.
    Returns DataFrame with predicted proba, actual label, and P&L.
    """
    df_full = fetch_ohlcv(ticker, period_years=2)
    if df_full.empty or len(df_full) < 300:
        return None

    fundamentals = fetch_fundamentals(ticker)

    try:
        feat_df = build_feature_matrix(df_full, sentiment=0.5, fundamentals=fundamentals)
    except Exception as e:
        logger.error(f"Feature build failed for {ticker}: {e}")
        return None

    # Use last 6 months as test window
    test_start = int(len(feat_df) * 0.75)
    test_df    = feat_df.iloc[test_start:].copy()

    avail = [c for c in feature_cols if c in test_df.columns]
    for col in set(feature_cols) - set(avail):
        test_df[col] = 0.0

    X_test = scaler.transform(test_df[feature_cols].values)
    probas  = model.predict_proba(X_test)[:, 1]

    test_df["proba"]     = probas
    test_df["predicted"] = (probas >= SIGNAL_THRESHOLD).astype(int)
    test_df["ticker"]    = ticker

    # Simulate trade P&L: if BUY signal, hold for SWING_TARGET_DAYS days
    returns = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if row["predicted"] == 1:  # BUY signal
            entry = row["close"]
            future_idx = i + SWING_TARGET_DAYS
            if future_idx < len(test_df):
                exit_price = test_df["close"].iloc[future_idx]
                pnl_pct    = (exit_price - entry) / entry * 100
            else:
                pnl_pct = 0.0
            returns.append({"date": idx, "ticker": ticker,
                            "entry": entry, "pnl_pct": pnl_pct,
                            "proba": row["proba"]})

    test_df["pnl_pct"] = 0.0
    return test_df, returns


def run_full_backtest(tickers: list = None):
    """Run backtest across all tickers and print a performance report."""
    if tickers is None:
        tickers = NIFTY50_TICKERS[:20]   # Limit for speed; increase as needed

    model, scaler, feature_cols = load_model()

    all_results = []
    all_trades  = []

    for ticker in tickers:
        logger.info(f"Backtesting {ticker}...")
        result = backtest_ticker(ticker, model, scaler, feature_cols)
        if result is None:
            continue
        test_df, trades = result
        all_results.append(test_df)
        all_trades.extend(trades)

    if not all_results:
        logger.error("No backtest results — check data availability")
        return

    combined = pd.concat(all_results)
    trades_df = pd.DataFrame(all_trades)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = combined["label"].values
    y_pred = combined["predicted"].values
    probas = combined["proba"].values

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, probas)

    print("\n" + "="*50)
    print("  BACKTEST RESULTS")
    print("="*50)
    print(f"  Tickers tested   : {len(tickers)}")
    print(f"  Total signals    : {y_pred.sum()}")
    print(f"  Precision@{SIGNAL_THRESHOLD:.2f}   : {prec:.3f}  ({prec*100:.1f}% correct)")
    print(f"  Recall           : {rec:.3f}")
    print(f"  ROC-AUC          : {auc:.3f}")

    if not trades_df.empty:
        wins     = (trades_df["pnl_pct"] > 0).sum()
        total_tr = len(trades_df)
        avg_pnl  = trades_df["pnl_pct"].mean()
        hit_rate = wins / total_tr * 100 if total_tr else 0

        print(f"\n  Simulated Trades : {total_tr}")
        print(f"  Win Rate         : {hit_rate:.1f}%")
        print(f"  Avg P&L/trade    : {avg_pnl:.2f}%")
        print(f"  Total P&L        : {trades_df['pnl_pct'].sum():.2f}%")

    print("="*50 + "\n")

    # ── Plot cumulative P&L ───────────────────────────────────────────────────
    if not trades_df.empty:
        trades_df = trades_df.sort_values("date")
        trades_df["cumulative_pnl"] = trades_df["pnl_pct"].cumsum()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(trades_df["date"], trades_df["cumulative_pnl"], color="#2563eb", linewidth=2)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.fill_between(trades_df["date"], trades_df["cumulative_pnl"], 0,
                         where=trades_df["cumulative_pnl"] >= 0, alpha=0.2, color="#16a34a")
        ax1.fill_between(trades_df["date"], trades_df["cumulative_pnl"], 0,
                         where=trades_df["cumulative_pnl"] < 0,  alpha=0.2, color="#dc2626")
        ax1.set_title("Cumulative P&L (%) — BUY Signals")
        ax1.set_ylabel("Cumulative Return %")
        ax1.grid(True, alpha=0.3)

        ax2.hist(trades_df["pnl_pct"], bins=40, color="#2563eb", alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_title("P&L Distribution per Trade")
        ax2.set_xlabel("Return %")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(MODEL_DIR, "backtest_results.png")
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info(f"Backtest chart saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_full_backtest()
