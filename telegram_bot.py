"""
telegram_bot.py — Send Nifty 50 signals to Telegram
"""
import logging
import asyncio
from datetime import date
from telegram import Bot
from telegram.constants import ParseMode
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

SESSION_EMOJI = {
    "Morning": "Morning",
    "Midday":  "Midday",
    "Closing": "Closing",
    "Test":    "Test",
    "":        "Signal",
}


def _format_signal_message(signal: dict) -> str:
    action   = signal["action"]
    ticker   = signal["ticker"].replace(".NS", "")
    conf     = signal["confidence_pct"]
    close    = signal["close"]
    target   = signal["target"]
    sl       = signal["stop_loss"]
    rr       = signal["risk_reward"]
    pe       = signal.get("pe_ratio", "N/A")
    sent     = signal.get("news_sentiment", 0.5)
    atr      = signal.get("atr", "N/A")
    sh       = signal.get("swing_high", "N/A")
    slo      = signal.get("swing_low", "N/A")
    sig_date = signal.get("date", str(date.today()))

    sent_label = "Positive" if sent > 0.55 else ("Negative" if sent < 0.45 else "Neutral")
    sent_icon  = "+" if sent > 0.55 else ("-" if sent < 0.45 else "~")
    action_tag = "[BUY]" if action == "BUY" else "[SELL]"

    msg = (
        f"{action_tag} *{ticker}*\n"
        f"------------------------\n"
        f"Date: `{sig_date}`\n"
        f"CMP: `Rs {close}`\n"
        f"Target: `Rs {target}`\n"
        f"Stop Loss: `Rs {sl}`\n"
        f"Risk:Reward: `1 : {rr}`\n"
        f"------------------------\n"
        f"Confidence: `{conf}%`\n"
        f"ATR: `{atr}`\n"
        f"20d High: `{sh}` | Low: `{slo}`\n"
        f"PE Ratio: `{pe}`\n"
        f"News: `{sent_icon} {sent_label}`\n"
        f"------------------------\n"
        f"_Not financial advice. Trade at your own risk._"
    )
    return msg


def _format_summary_message(signals: list, session: str = "") -> str:
    today      = date.today().strftime("%d %b %Y")
    buy_count  = sum(1 for s in signals if s["action"] == "BUY")
    sell_count = sum(1 for s in signals if s["action"] == "SELL")
    mkt_state  = "Bullish" if signals and signals[0].get("market_bullish") else "Bearish"
    sess_label = SESSION_EMOJI.get(session, session) if session else "Daily"

    return (
        f"*Nifty 50 Signals - {sess_label} Session*\n"
        f"Date: {today}\n"
        f"Market: {mkt_state}\n"
        f"BUY: {buy_count}  |  SELL: {sell_count}\n"
        f"------------------------\n"
        f"Top picks below"
    )


async def _send_messages(signals: list, session: str = ""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    if signals:
        summary = _format_summary_message(signals, session)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=summary,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("Sent summary message")

    for signal in signals:
        msg = _format_signal_message(signal)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"Sent: {signal['action']} {signal['ticker']}")
        await asyncio.sleep(0.5)

    if not signals:
        sess_label = f"{session} " if session else ""
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"*No high-confidence signals - {sess_label}{date.today()}*\nNo setups above threshold.",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("No signals - sent empty message")


def send_signals(signals: list, session: str = ""):
    """Sync wrapper - call this from the scheduler."""
    asyncio.run(_send_messages(signals, session))


async def _test_connection():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            "*Nifty 50 Bot connected!*\n"
            "Schedule:\n"
            "Morning - 9:30 AM IST\n"
            "Closing - 3:45 PM IST\n"
            "(Mon-Fri only)"
        ),
        parse_mode=ParseMode.MARKDOWN,
    )
    logger.info("Test message sent")


def test_bot():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first!")
        return
    asyncio.run(_test_connection())
    print("Test message sent! Check your Telegram.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()
    if args.test:
        test_bot()
    else:
        parser.print_help()
