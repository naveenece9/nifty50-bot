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


def _format_signal_message(signal: dict) -> str:
    """Format a single signal into a clean Telegram message."""
    action  = signal["action"]
    ticker  = signal["ticker"].replace(".NS", "")
    conf    = signal["confidence_pct"]
    close   = signal["close"]
    target  = signal["target"]
    sl      = signal["stop_loss"]
    rr      = signal["risk_reward"]
    pe      = signal.get("pe_ratio", "N/A")
    sent    = signal.get("news_sentiment", 0.5)
    sig_date= signal.get("date", str(date.today()))

    emoji   = "🟢" if action == "BUY" else "🔴"
    sent_label = "📰 Positive" if sent > 0.55 else ("📰 Negative" if sent < 0.45 else "📰 Neutral")

    msg = (
        f"{emoji} *{action} Signal — {ticker}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Date: `{sig_date}`\n"
        f"💰 CMP: `₹{close}`\n"
        f"🎯 Target: `₹{target}`\n"
        f"🛡 Stop Loss: `₹{sl}`\n"
        f"⚖️ Risk:Reward: `1:{rr}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Model Confidence: `{conf}%`\n"
        f"📊 PE Ratio: `{pe}`\n"
        f"{sent_label} news sentiment\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _This is not financial advice. Trade at your own risk._"
    )
    return msg


def _format_summary_message(signals: list) -> str:
    """Format a summary header before sending individual signals."""
    today       = date.today().strftime("%d %b %Y")
    buy_count   = sum(1 for s in signals if s["action"] == "BUY")
    sell_count  = sum(1 for s in signals if s["action"] == "SELL")
    market_state= "🟢 Bullish" if signals[0].get("market_bullish") else "🔴 Bearish"

    return (
        f"📊 *Nifty 50 Daily Signals — {today}*\n"
        f"Market Regime: {market_state}\n"
        f"Signals: `{buy_count}` BUY  |  `{sell_count}` SELL\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Top picks below 👇"
    )


async def _send_messages(signals: list):
    """Async: send summary + one message per signal."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    # Send summary header
    if signals:
        summary = _format_summary_message(signals)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=summary,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("Sent summary message")

    # Send each signal
    for signal in signals:
        msg = _format_signal_message(signal)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"Sent signal: {signal['action']} {signal['ticker']}")
        await asyncio.sleep(0.5)  # Avoid hitting Telegram rate limits

    if not signals:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"📊 *No high-confidence signals today* ({date.today()})\nModel found no setups above the threshold.",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("No signals to send — sent 'no signals' message")


def send_signals(signals: list):
    """Sync wrapper — call this from the scheduler."""
    asyncio.run(_send_messages(signals))


async def _test_connection():
    """Quick test: send a test message to verify bot is working."""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="✅ *Nifty 50 Bot connected!*\nDaily signals will arrive after market close.",
        parse_mode=ParseMode.MARKDOWN,
    )
    logger.info("Test message sent successfully")


def test_bot():
    """Send a test message to verify setup."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first!")
        return
    asyncio.run(_test_connection())
    print("✅ Test message sent! Check your Telegram.")


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
