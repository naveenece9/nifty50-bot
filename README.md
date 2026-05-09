# 📊 Nifty 50 Signal Bot

An ML-powered swing trading signal bot that analyses all 50 Nifty 50 stocks daily
and sends **BUY / SELL** alerts directly to your Telegram channel.

**Model**: XGBoost trained on technical indicators, price patterns, news sentiment,
and fundamental data. Signals fire only when model confidence ≥ 65%.

---

## Architecture

```
OHLCV Data (yfinance)
Tech Indicators (ta)      ──→  Feature Matrix  ──→  XGBoost  ──→  Signal Filter  ──→  Telegram
News Sentiment (NewsAPI)                                              (confidence + market regime)
Fundamentals (yfinance)
```

---

## Quick Start (5 steps)

### Step 1 — Clone and install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install packages
pip install -r requirements.txt
```

> **Note**: `torch` is a large download (~2GB). If you don't need FinBERT sentiment,
> remove `transformers` and `torch` from requirements.txt — the bot uses lightweight
> keyword sentiment by default.

---

### Step 2 — Create your Telegram Bot

1. Open Telegram and message **@BotFather**
2. Send: `/newbot`
3. Follow prompts → you'll get a **Bot Token** (looks like `7123456789:AAHxxx...`)
4. Message your new bot once (just say "hi")
5. Visit this URL to get your **Chat ID**:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   Look for `"chat": {"id": 123456789}` in the response

---

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
TELEGRAM_BOT_TOKEN=7123456789:AAHxxx...your_token_here
TELEGRAM_CHAT_ID=123456789

# Optional — get free key at newsapi.org (500 requests/day free)
NEWS_API_KEY=your_newsapi_key

# Signal settings (defaults work well)
SIGNAL_THRESHOLD=0.65
SWING_TARGET_DAYS=5
SWING_TARGET_PCT=2.0
MAX_SIGNALS_PER_RUN=5
```

**Test your bot connection:**
```bash
python telegram_bot.py --test
```
You should receive a test message on Telegram instantly.

---

### Step 4 — Train the model

This downloads 2 years of data for all 50 Nifty stocks and trains XGBoost.
**Takes 15–30 minutes** depending on your internet speed.

```bash
python model_trainer.py --train
```

You'll see output like:
```
Fold 1 — AUC: 0.712  Precision@0.65: 0.641
Fold 2 — AUC: 0.724  Precision@0.65: 0.658
...
Model saved to models/xgb_model.pkl
```

Saved files:
- `models/xgb_model.pkl` — trained model
- `models/scaler.pkl` — feature scaler
- `models/feature_cols.pkl` — feature list
- `models/feature_importance.png` — which features matter most

**Retrain monthly** to keep the model current:
```bash
python model_trainer.py --train
```

---

### Step 5 — Run the bot

**Test run (generates signals immediately):**
```bash
python scheduler.py --now
```

**Start the daily scheduler (runs every weekday at 4:15 PM IST):**
```bash
python scheduler.py --schedule
```

Keep this running on a server or use a cloud service (see Deployment section below).

---

## File Structure

```
nifty50_bot/
├── config.py              # All settings and ticker list
├── data_fetcher.py        # Download OHLCV, news, fundamentals
├── feature_engineer.py    # Technical indicators + labels
├── model_trainer.py       # Train and save XGBoost model
├── signal_generator.py    # Daily inference + filters
├── telegram_bot.py        # Format and send Telegram messages
├── scheduler.py           # APScheduler daily job
├── backtester.py          # Evaluate model on historical data
├── requirements.txt
├── .env.example
├── data/                  # Cached OHLCV parquet files
├── models/                # Saved model artifacts
├── logs/                  # Daily log files
└── signals/               # JSON signal files per day
```

---

## Features Used by the Model

| Category | Features |
|---|---|
| Trend | EMA 20/50/200, ADX, price vs EMA ratios |
| Momentum | RSI, MACD, Stochastics, Williams %R, CCI, ROC |
| Volatility | Bollinger Bands width/%, ATR, ATR % |
| Volume | OBV ratio, volume spike, VWAP deviation |
| Price patterns | Candle body, upper/lower shadow, gap, is_bullish |
| Lag returns | 1d, 3d, 5d, 10d returns |
| Rolling stats | 5/10/20d mean, std, max, min |
| Sentiment | News keyword score (0=negative, 1=positive) |
| Fundamentals | PE ratio, PB ratio, D/E ratio, ROE |

---

## Signal Filters

Signals are blocked if:
- **Model confidence < 65%** (configurable via `SIGNAL_THRESHOLD`)
- **Market is bearish** — Nifty 50 index is below its 50-day EMA
- **Stock overextended** — rose >5% in last 3 days (avoid chasing)

---

## Backtesting

Evaluate model performance on the last 6 months of data:

```bash
python backtester.py
```

Output:
```
==================================================
  BACKTEST RESULTS
==================================================
  Tickers tested   : 20
  Total signals    : 143
  Precision@0.65   : 0.641  (64.1% correct)
  Recall           : 0.312
  ROC-AUC          : 0.718

  Simulated Trades : 89
  Win Rate         : 61.8%
  Avg P&L/trade    : 1.42%
==================================================
```

Charts saved to `models/backtest_results.png`

---

## Deployment (Free Options)

### Option A — Railway (easiest)

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables in the Railway dashboard
4. It runs 24/7 for free (500 hours/month on free tier)

### Option B — Render

1. Push to GitHub
2. [render.com](https://render.com) → New → Background Worker
3. Set start command: `python scheduler.py --schedule`
4. Add environment variables

### Option C — VPS (DigitalOcean / Linode ~$5/month)

```bash
# Run in background with nohup
nohup python scheduler.py --schedule > logs/nohup.log 2>&1 &
```

### Option D — Local machine + cron

If you have a PC that stays on during market hours:
```bash
# Add to crontab (runs at 4:15 PM Mon-Fri)
15 16 * * 1-5 cd /path/to/nifty50_bot && python scheduler.py --now
```

---

## Telegram Message Format

```
🟢 BUY Signal — RELIANCE
━━━━━━━━━━━━━━━━━━━━━━
📅 Date: 2025-01-15
💰 CMP: ₹2847.50
🎯 Target: ₹2934.20
🛡 Stop Loss: ₹2788.10
⚖️ Risk:Reward: 1:1.45
━━━━━━━━━━━━━━━━━━━━━━
🤖 Model Confidence: 71.3%
📊 PE Ratio: 28.4
📰 Positive news sentiment
━━━━━━━━━━━━━━━━━━━━━━
⚠️ This is not financial advice. Trade at your own risk.
```

---

## Tips to Improve Accuracy

1. **Retrain monthly** — markets change, models drift
2. **Paper trade first** — run for 1 month without real money to validate
3. **Increase threshold** — set `SIGNAL_THRESHOLD=0.70` for fewer but higher-confidence signals
4. **Add sector filter** — only trade sectors with strong momentum
5. **Add earnings blackout** — avoid signals 2 days before/after quarterly results

---

## Disclaimer

This bot is for **educational and research purposes only**.
It does not constitute financial advice. Always consult a SEBI-registered advisor
before making investment decisions. Past model performance does not guarantee future results.
