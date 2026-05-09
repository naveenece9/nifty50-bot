"""
startup.py — Train model if needed, run signals once and exit.
Designed for Railway Cron Jobs — spins up, sends signals, shuts down.
SESSION_NAME env variable must be set (Morning or Closing).
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Safety check — prevent accidental infinite loop ───────────────────────────
session = os.getenv("SESSION_NAME", "").strip()
if not session:
    logger.error("SESSION_NAME environment variable not set.")
    logger.error("This script should only run via Railway Cron Job.")
    logger.error("Set SESSION_NAME=Morning or SESSION_NAME=Closing in your cron job variables.")
    sys.exit(0)

logger.info(f"Starting {session} signal run...")

# ── Create required folders ───────────────────────────────────────────────────
for folder in ["models", "data", "logs", "signals"]:
    os.makedirs(folder, exist_ok=True)

# ── Train model if not found ──────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "xgb_model.pkl")
if not os.path.exists(MODEL_PATH):
    logger.info("No model found - training now (takes 20-40 mins)...")
    from model_trainer import train_model
    train_model()
    logger.info("Training complete!")
else:
    logger.info("Model found - skipping training")

# ── Generate and send signals, then exit ─────────────────────────────────────
from signal_generator import run_daily_signals
from telegram_bot import send_signals

try:
    signals = run_daily_signals()
    send_signals(signals, session=session)
    logger.info(f"Done - {len(signals)} signal(s) sent for {session} session. Exiting.")
    sys.exit(0)
except Exception as e:
    logger.error(f"Signal run failed: {e}", exc_info=True)
    sys.exit(1)