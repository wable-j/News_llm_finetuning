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
    # Reduced number of feeds to speed up data collection
]
MAX_ARTICLES_PER_FEED = 50  # Reduced from 100 to speed up data processing
DATA_CACHE_FILE = DATA_DIR / "news_articles.json"
PROCESSED_DATA_FILE = DATA_DIR / "processed_articles.json"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Dataset settings
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_INPUT_LENGTH = 512  # Reduced from 1024 to fit in 8GB VRAM
MAX_TARGET_LENGTH = 64   # Reduced from 128 for faster training
RANDOM_SEED = 42

# Model settings
MODEL_NAME = "facebook/bart-large-cnn"  # Alternative model that works well for summarization
BATCH_SIZE = 4  # Maintained batch size of 4 which should work with 8GB VRAM
LEARNING_RATE = 1e-4  # Increased from 5e-5 for faster convergence
NUM_EPOCHS = 1  # Reduced from 3 to complete training in 30 minutes
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced from 4 for faster updates

# Evaluation settings
METRICS = ["rouge1", "rouge2", "rougeL"]  # Reduced metrics for faster evaluation
SAVE_EVERY_N_STEPS = 100

# Deployment settings
API_PORT = 5000
API_HOST = "0.0.0.0"