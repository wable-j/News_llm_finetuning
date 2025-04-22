"""
Project configuration parameters for news summarization model fine-tuning
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
EVALUATION_DIR = PROJECT_ROOT / "src" / "evaluation"
DEPLOYMENT_DIR = PROJECT_ROOT / "src" / "deployment"

# Raw data settings
RSS_FEEDS = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "https://feeds.nbcnews.com/nbcnews/public/news",
    "https://www.npr.org/rss/rss.php?id=1001",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
]
MAX_ARTICLES_PER_FEED = 100
DATA_CACHE_FILE = DATA_DIR / "news_articles.json"
PROCESSED_DATA_FILE = DATA_DIR / "processed_articles.json"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Dataset settings
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
RANDOM_SEED = 42

# Model settings
MODEL_NAME = "google/pegasus-cnn_dailymail"  # Base model for fine-tuning
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 4

# Evaluation settings
METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
SAVE_EVERY_N_STEPS = 100

# Deployment settings
API_PORT = 5000
API_HOST = "0.0.0.0"