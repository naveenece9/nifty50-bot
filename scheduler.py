"""
scheduler.py — Runs signals twice daily: 9:30 AM and 3:45 PM IST (Mon-Fri)
"""
import logging
import os
import sys
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import LOG_DIR
from signal_generator import run_daily_signals
from telegram_bot import send_signals

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Logging setup
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"bot_{datetime.today().strftime('%Y%m')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def daily_signal_job(session: str = ""):
    """Generate signals and send to Telegram."""
    logger.info(f"[{session}] Signal job started")
    try:
        signals = run_daily_signals()
        send_signals(signals, session=session)
        logger.info(f"[{session}] Done - {len(signals)} signal(s) sent")
    except Exception as e:
        logger.error(f"[{session}] Job failed: {e}", exc_info=True)


def start_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # 9:30 AM IST — Morning signals
    scheduler.add_job(
        daily_signal_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=9,
            minute=30,
            timezone="Asia/Kolkata",
        ),
        kwargs={"session": "Morning"},
        id="morning_signals",
        name="Morning Signal 9:30 AM",
        misfire_grace_time=300,
    )

    # 3:45 PM IST — Closing signals
    scheduler.add_job(
        daily_signal_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=15,
            minute=45,
            timezone="Asia/Kolkata",
        ),
        kwargs={"session": "Closing"},
        id="closing_signals",
        name="Closing Signal 3:45 PM",
        misfire_grace_time=300,
    )

    logger.info("Scheduler started - running Mon-Fri at 9:30 AM and 3:45 PM IST")
    logger.info("Next signals: Monday 9:30 AM IST")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nifty 50 Signal Bot")
    parser.add_argument("--now",      action="store_true", help="Run immediately")
    parser.add_argument("--schedule", action="store_true", help="Start scheduler")
    parser.add_argument("--session",  type=str, default="Test", help="Session label")
    args = parser.parse_args()

    if args.now:
        logger.info("Running immediately...")
        daily_signal_job(session=args.session)
    elif args.schedule:
        start_scheduler()
    else:
        parser.print_help()