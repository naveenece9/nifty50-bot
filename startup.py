"""
startup.py — Train model if needed, then run signals once and exit.
Used by Railway Cron Jobs — spins up, sends signals, shuts down.
"""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Create required folders
for folder in ["models", "data", "logs", "signals"]:
    os.makedirs(folder, exist_ok=True)

# Train model if not found
MODEL_PATH = os.path.join("models", "xgb_model.pkl")
if not os.path.exists(MODEL_PATH):
    logger.info("No model found - training now (takes 20-40 mins)...")
    from model_trainer import train_model
    train_model()
    logger.info("Training complete!")
else:
    logger.info("Model found - skipping training")

# Get session name from environment variable
session = os.getenv("SESSION_NAME", "Signal")
logger.info(f"Running {session} signals...")

# Generate and send signals, then exit
from signal_generator import run_daily_signals
from telegram_bot import send_signals

signals = run_daily_signals()
send_signals(signals, session=session)
logger.info(f"Done - {len(signals)} signal(s) sent. Exiting.")