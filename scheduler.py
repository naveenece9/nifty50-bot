"""
scheduler.py — APScheduler job to run signals daily after market close
"""
import logging
import os
import sys
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import MARKET_CLOSE_HOUR, LOG_DIR
from signal_generator import run_daily_signals
from telegram_bot import send_signals

# Fix Windows console encoding for unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Logging setup
log_file = os.path.join(LOG_DIR, f"bot_{datetime.today().strftime('%Y%m')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=open(os.devnull, "w")),  # suppress console emoji errors
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)

# Add a safe console handler that replaces unencodable chars
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)


def daily_signal_job():
    """Main daily job: generate signals and push to Telegram."""
    logger.info(">> Daily signal job started")
    try:
        signals = run_daily_signals()
        send_signals(signals)
        logger.info(f">> Job complete - {len(signals)} signal(s) sent")
    except Exception as e:
        logger.error(f">> Job failed: {e}", exc_info=True)


def start_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        daily_signal_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=MARKET_CLOSE_HOUR,
            minute=15,
            timezone="Asia/Kolkata",
        ),
        id="daily_signals",
        name="Nifty 50 Daily Signal",
        misfire_grace_time=300,
    )
    logger.info(f"Scheduler started - will run Mon-Fri at {MARKET_CLOSE_HOUR}:15 IST")
    logger.info("Press Ctrl+C to stop")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nifty 50 Signal Bot Scheduler")
    parser.add_argument("--now",      action="store_true", help="Run signal job immediately")
    parser.add_argument("--schedule", action="store_true", help="Start the daily scheduler")
    args = parser.parse_args()

    if args.now:
        logger.info("Running job immediately (--now flag)")
        daily_signal_job()
    elif args.schedule:
        start_scheduler()
    else:
        parser.print_help()