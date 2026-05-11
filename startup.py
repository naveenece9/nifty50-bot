import os
import sys
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for folder in ["models", "data", "logs", "signals"]:
    os.makedirs(folder, exist_ok=True)

MODEL_PATH = os.path.join("models", "xgb_model.pkl")
if not os.path.exists(MODEL_PATH):
    logger.info("No model found - training now...")
    from model_trainer import train_model
    train_model()
    logger.info("Training complete!")
else:
    logger.info("Model found - skipping training")

logger.info("Starting scheduler...")
from scheduler import start_scheduler
start_scheduler()
