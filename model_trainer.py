"""
model_trainer.py — Train and evaluate the XGBoost signal model
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)

from config import MODEL_DIR
from data_fetcher import (
    fetch_all_ohlcv, fetch_sentiment_for_all,
    fetch_fundamentals, fetch_nifty_index,
)
from feature_engineer import build_feature_matrix, get_feature_columns

logger = logging.getLogger(__name__)

MODEL_PATH   = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH= os.path.join(MODEL_DIR, "feature_cols.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  Build training dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(tickers: list = None) -> pd.DataFrame:
    """
    Fetch data for all tickers and combine into one big training DataFrame.
    Each row = one stock × one day.
    """
    from config import NIFTY50_TICKERS
    if tickers is None:
        tickers = NIFTY50_TICKERS

    logger.info("Fetching OHLCV data...")
    all_data   = fetch_all_ohlcv(tickers)
    sentiments = fetch_sentiment_for_all(tickers)

    frames = []
    for ticker, df in all_data.items():
        logger.info(f"Building features for {ticker}")
        sentiment    = sentiments.get(ticker, 0.5)
        fundamentals = fetch_fundamentals(ticker)
        try:
            feat_df = build_feature_matrix(df, sentiment, fundamentals)
            feat_df["ticker"] = ticker
            frames.append(feat_df)
        except Exception as e:
            logger.error(f"Feature build failed for {ticker}: {e}")

    if not frames:
        raise ValueError("No feature data built — check data fetcher logs")

    combined = pd.concat(frames, axis=0)
    combined.sort_index(inplace=True)
    logger.info(f"Training dataset: {combined.shape}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Train model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame = None):
    """
    Train XGBoost on the combined dataset.
    Uses walk-forward time-series cross-validation.
    Saves model, scaler, and feature list to disk.
    """
    if df is None:
        df = build_training_dataset()

    feature_cols = get_feature_columns(df)

    # Remove last SWING_TARGET_DAYS rows per ticker (no valid label yet)
    from config import SWING_TARGET_DAYS
    df = df.groupby("ticker").apply(
        lambda g: g.iloc[:-SWING_TARGET_DAYS]
    ).reset_index(drop=True)

    X = df[feature_cols].values
    y = df["label"].values

    logger.info(f"Samples: {len(y)}  |  BUY: {y.sum()}  |  SELL: {(y==0).sum()}")
    logger.info(f"Features: {len(feature_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series split (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)

    # Compute class weight
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        scale_pos_weight  = pos_weight,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx],         y[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba    = model.predict_proba(X_val)[:, 1]
        auc      = roc_auc_score(y_val, proba)
        prec     = precision_score(y_val, (proba >= 0.65).astype(int), zero_division=0)
        cv_scores.append({"fold": fold+1, "auc": auc, "precision@0.65": prec})
        logger.info(f"  Fold {fold+1} — AUC: {auc:.3f}  Precision@0.65: {prec:.3f}")

    # Final fit on all data
    model.fit(X_scaled, y, verbose=False)

    # Save artifacts
    joblib.dump(model,        MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Print summary
    scores_df = pd.DataFrame(cv_scores)
    logger.info(f"\nCV Summary:\n{scores_df.to_string(index=False)}")
    logger.info(f"Mean AUC: {scores_df['auc'].mean():.3f}")

    _save_feature_importance(model, feature_cols)
    return model, scaler, feature_cols


def _save_feature_importance(model, feature_cols):
    """Save a feature importance bar chart."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Feature importance chart saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Load saved model
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load saved model, scaler, and feature list from disk."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        raise FileNotFoundError(
            "Model not found. Run: python model_trainer.py --train"
        )
    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    logger.info("Model loaded from disk")
    return model, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("logs", "training.log")),
        ],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        logger.info("Starting model training...")
        train_model()
        logger.info("Training complete!")
    else:
        parser.print_help()
