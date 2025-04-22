"""
Module for fetching news articles from RSS feeds and preprocessing them for model training
"""

import json
import os
import re
import time
from datetime import datetime
import random
import feedparser
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSS_FEEDS, MAX_ARTICLES_PER_FEED, DATA_CACHE_FILE, PROCESSED_DATA_FILE,
    TRAIN_FILE, VAL_FILE, TEST_FILE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

# Download NLTK resources
nltk.download('punkt', quiet=True)

def fetch_rss_feeds():
    """
    Fetch news articles from RSS feeds and save to a local cache file
    """
    print(f"Fetching articles from {len(RSS_FEEDS)} RSS feeds...")
    all_articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.title
            
            print(f"Processing {source}: Found {len(feed.entries)} articles")
            
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                # Extract article data
                article = {
                    'source': source,
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'content': ''
                }
                
                # Try to get full content if available
                if 'content' in entry:
                    article['content'] = entry.content[0].value
                elif 'description' in entry:
                    article['content'] = entry.description
                else:
                    article['content'] = article['summary']
                
                all_articles.append(article)
            
            # Pause to avoid hammering the servers
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching {feed_url}: {e}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DATA_CACHE_FILE), exist_ok=True)
    
    # Save to cache
    with open(DATA_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(all_articles)} articles to {DATA_CACHE_FILE}")
    return all_articles

def clean_html(text):
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def preprocess_articles(articles=None):
    """
    Preprocess articles for summarization:
    - Clean HTML
    - Extract full text
    - Generate extractive summary
    """
    if articles is None:
        # Load from cache if not provided
        try:
            with open(DATA_CACHE_FILE, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except FileNotFoundError:
            print(f"Cache file {DATA_CACHE_FILE} not found. Fetching new articles...")
            articles = fetch_rss_feeds()
    
    processed_articles = []
    
    for article in articles:
        # Clean content
        content = clean_html(article['content'])
        
        # Get sentences and create an extractive summary (first 3 sentences)
        sentences = sent_tokenize(content)
        if len(sentences) > 3:
            extractive_summary = ' '.join(sentences[:3])
        else:
            extractive_summary = content
            
        processed_article = {
            'title': article['title'],
            'source': article['source'],
            'url': article['link'],
            'published_date': article.get('published', ''),
            'text': content,
            'summary': extractive_summary
        }
        
        processed_articles.append(processed_article)
    
    # Save processed articles
    with open(PROCESSED_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(processed_articles)} processed articles to {PROCESSED_DATA_FILE}")
    return processed_articles

def split_and_save_data(articles=None):
    """
    Split processed articles into train, validation, and test sets
    """
    if articles is None:
        # Load from processed file if not provided
        try:
            with open(PROCESSED_DATA_FILE, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except FileNotFoundError:
            print(f"Processed file {PROCESSED_DATA_FILE} not found. Processing articles...")
            articles = preprocess_articles()
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(articles)
    
    # Calculate split indices
    total = len(articles)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    # Split data
    train_data = articles[:train_end]
    val_data = articles[train_end:val_end]
    test_data = articles[val_end:]
    
    print(f"Splitting data: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    
    # Save as jsonl files
    for data, file_path in [
        (train_data, TRAIN_FILE),
        (val_data, VAL_FILE),
        (test_data, TEST_FILE)
    ]:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} records to {file_path}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def create_dataset_stats(split_data=None):
    """
    Create statistics about the dataset
    """
    if split_data is None:
        # Load split data
        split_data = {}
        for split, file_path in [
            ('train', TRAIN_FILE),
            ('validation', VAL_FILE),
            ('test', TEST_FILE)
        ]:
            data = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                split_data[split] = data
            except FileNotFoundError:
                print(f"File {file_path} not found.")
    
    stats = {}
    
    for split, data in split_data.items():
        # Skip if data is empty
        if not data:
            continue
            
        # Calculate text and summary lengths
        text_lengths = [len(item['text'].split()) for item in data]
        summary_lengths = [len(item['summary'].split()) for item in data]
        
        # Count sources
        sources = {}
        for item in data:
            source = item['source']
            sources[source] = sources.get(source, 0) + 1
        
        stats[split] = {
            'count': len(data),
            'text_length': {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'avg': sum(text_lengths) / len(text_lengths)
            },
            'summary_length': {
                'min': min(summary_lengths),
                'max': max(summary_lengths),
                'avg': sum(summary_lengths) / len(summary_lengths)
            },
            'sources': sources
        }
    
    # Print statistics
    for split, split_stats in stats.items():
        print(f"\n{split.capitalize()} set statistics:")
        print(f"  Count: {split_stats['count']}")
        print(f"  Text length: min={split_stats['text_length']['min']}, "
              f"max={split_stats['text_length']['max']}, "
              f"avg={split_stats['text_length']['avg']:.1f}")
        print(f"  Summary length: min={split_stats['summary_length']['min']}, "
              f"max={split_stats['summary_length']['max']}, "
              f"avg={split_stats['summary_length']['avg']:.1f}")
        print("  Sources:")
        for source, count in split_stats['sources'].items():
            print(f"    {source}: {count}")
    
    return stats

def main():
    """Main function to execute the data preprocessing pipeline"""
    print("Starting data preprocessing pipeline...")
    
    # Step 1: Fetch articles from RSS feeds
    articles = fetch_rss_feeds()
    
    # Step 2: Preprocess articles
    processed_articles = preprocess_articles(articles)
    
    # Step 3: Split into train, validation, and test sets
    split_data = split_and_save_data(processed_articles)
    
    # Step 4: Create dataset statistics
    stats = create_dataset_stats(split_data)
    
    print("Data preprocessing pipeline completed successfully!")
    
if __name__ == "__main__":
    main()