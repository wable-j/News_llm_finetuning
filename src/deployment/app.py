"""
Module for deploying a summarization model as a web API with fallback to pre-trained model
and topic-based article search functionality
"""

import os
import json
import time
import requests
from datetime import datetime
import torch
import feedparser
import re
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, API_PORT, API_HOST, MODEL_NAME, RSS_FEEDS
)

# Create Flask app
app = Flask(__name__)

# Global model and tokenizer objects
MODEL = None
TOKENIZER = None
USING_PRETRAINED = False
ARTICLE_CACHE = {}  # Cache for storing fetched articles

def load_model():
    """
    Load the fine-tuned model and tokenizer, with fallback to pre-trained model
    """
    global MODEL, TOKENIZER, USING_PRETRAINED
    
    model_dir = os.path.join(MODELS_DIR, "fine-tuned-news-summarizer")
    
    try:
        if os.path.exists(model_dir):
            print(f"Loading fine-tuned model from {model_dir}")
            MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
            USING_PRETRAINED = False
        else:
            print(f"Fine-tuned model not found. Loading pre-trained model {MODEL_NAME} instead.")
            MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            USING_PRETRAINED = True
            
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL.to(device)
        MODEL.eval()
        
        print(f"Model loaded and ready on {device}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_summary(text):
    """
    Generate a summary for the given text
    """
    global MODEL, TOKENIZER
    
    # Check if model is loaded
    if MODEL is None or TOKENIZER is None:
        load_model()
    
    # Determine device
    device = next(MODEL.parameters()).device
    
    # Tokenize
    inputs = TOKENIZER(
        text,
        max_length=MAX_INPUT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    # Generate summary
    with torch.no_grad():
        generated_ids = MODEL.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode summary
    summary = TOKENIZER.decode(generated_ids[0], skip_special_tokens=True)
    
    return summary

def fetch_article_from_url(url):
    """
    Fetch an article from a URL using feedparser or requests for regular web pages
    """
    try:
        # Check if article is already in cache
        if url in ARTICLE_CACHE:
            return ARTICLE_CACHE[url]
        
        # Parse the feed or URL
        if url.startswith('http'):
            # Try as RSS feed first
            feed = feedparser.parse(url)
            
            # Check if it's a valid feed
            if feed.get('entries') and len(feed.entries) > 0:
                # Get the first entry
                entry = feed.entries[0]
                
                # Extract article data
                article = {
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
                
                # Cache the article
                ARTICLE_CACHE[url] = article
                return article
            else:
                # If not a feed, try to fetch as web page
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get title
                title = soup.title.text if soup.title else "Untitled Article"
                
                # Try to extract content (this is a simplified approach)
                # In a real system, you'd use a more sophisticated content extractor
                content = ""
                article_content = soup.find('article') or soup.find('main') or soup.find('div', {'class': re.compile('(content|article).*', re.I)})
                
                if article_content:
                    # Remove script, style elements
                    for element in article_content.select('script, style, nav, header, footer, aside'):
                        element.extract()
                    
                    # Get text
                    content = article_content.get_text(separator='\n').strip()
                else:
                    # Fallback: get all paragraphs
                    paragraphs = soup.find_all('p')
                    content = '\n'.join([p.get_text() for p in paragraphs])
                
                article = {
                    'title': title,
                    'link': url,
                    'content': content if content else "Unable to extract content automatically."
                }
                
                # Cache the article
                ARTICLE_CACHE[url] = article
                return article
        
        # If we get here, something went wrong
        return {
            'title': 'Article',
            'link': url,
            'content': 'Unable to extract content automatically. Please paste the article text directly.'
        }
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Failed to fetch article. Please paste the article text directly.'
        }

def fetch_all_feed_articles():
    """
    Fetch all articles from configured RSS feeds
    """
    all_articles = []
    
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            
            if feed.get('entries'):
                # Extract feed name from feed title or URL
                feed_name = feed.get('feed', {}).get('title', feed_url.split('/')[-1])
                
                for entry in feed.entries:
                    # Create article object
                    article = {
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'source': feed_name,
                        'summary': entry.get('summary', '')
                    }
                    
                    # Try to get full content if available
                    if 'content' in entry:
                        article['content'] = entry.content[0].value
                    elif 'description' in entry:
                        article['content'] = entry.description
                    else:
                        article['content'] = article['summary']
                    
                    # Add to list of articles
                    all_articles.append(article)
                    
                    # Cache the article
                    ARTICLE_CACHE[entry.link] = article
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {str(e)}")
    
    return all_articles

def search_articles(query, all_articles=None):
    """
    Search for articles matching a query
    """
    if all_articles is None:
        all_articles = fetch_all_feed_articles()
    
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Filter articles based on query
    matching_articles = []
    for article in all_articles:
        # Check if query appears in title or content
        if (query in article.get('title', '').lower() or 
            query in article.get('content', '').lower() or
            query in article.get('summary', '').lower()):
            
            # Create a simplified version for the search results
            matching_articles.append({
                'title': article.get('title', 'Untitled'),
                'link': article.get('link', ''),
                'source': article.get('source', 'Unknown'),
                'published': article.get('published', ''),
                'snippet': get_context_snippet(article.get('content', ''), query) or article.get('summary', '')[:150] + '...'
            })
    
    return matching_articles

def get_context_snippet(content, query, context_size=75):
    """
    Extract a snippet of text around the query term for context
    """
    if not content or not query:
        return ""
    
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Find position of query in content
    pos = content_lower.find(query_lower)
    if pos == -1:
        return ""
    
    # Calculate start and end positions for the snippet
    start = max(0, pos - context_size)
    end = min(len(content), pos + len(query) + context_size)
    
    # Extract the snippet
    snippet = content[start:end].strip()
    
    # Add ellipsis if needed
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet

# API Routes
@app.route('/')
def home():
    """Home page with a simple UI for the summarization service"""
    global USING_PRETRAINED
    
    model_status = "Using pre-trained model (fine-tuned model not found)" if USING_PRETRAINED else "Using fine-tuned model"
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Summarization Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            textarea {
                width: 100%;
                min-height: 200px;
                padding: 10px;
                font-family: inherit;
            }
            .input-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            label {
                font-weight: bold;
            }
            .btn {
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            .url-input, .search-input {
                width: 100%;
                padding: 10px;
            }
            .result {
                border: 1px solid #ddd;
                padding: 15px;
                background-color: #f9f9f9;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 10px;
                font-style: italic;
            }
            .tabs {
                display: flex;
                margin-bottom: 10px;
            }
            .tab {
                padding: 10px 15px;
                cursor: pointer;
                background-color: #eee;
            }
            .tab.active {
                background-color: #ddd;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .model-status {
                background-color: #f8f9fa;
                padding: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #17a2b8;
            }
            .article-item {
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 10px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .article-item:hover {
                background-color: #f0f0f0;
            }
            .article-title {
                font-weight: bold;
                margin-bottom: 5px;
                font-size: 1.1em;
            }
            .article-source {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .article-snippet {
                font-size: 0.9em;
                color: #333;
            }
            .highlight {
                background-color: yellow;
                font-weight: bold;
            }
            .hidden {
                display: none;
            }
            .selected-article {
                border: 2px solid #4CAF50;
                padding: 15px;
                margin-bottom: 15px;
            }
            .selected-article-title {
                font-weight: bold;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            .selected-article-meta {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }
            .selected-article-content {
                margin-bottom: 15px;
                max-height: 300px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #eee;
                background-color: #fafafa;
            }
            .search-results-container {
                margin-top: 20px;
            }
            .error-message {
                color: #d9534f;
                padding: 10px;
                background-color: #f9f2f2;
                border-left: 4px solid #d9534f;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <h1>News Summarization Service</h1>
        <p>Search for news by topic, select an article, and generate summaries using an AI model.</p>
        
        <div class="model-status">
            <strong>Status:</strong> """ + model_status + """
        </div>
        
        <div class="container">
            <div class="tabs">
                <div class="tab active" data-tab="search">Topic Search</div>
                <div class="tab" data-tab="text">Text Input</div>
                <div class="tab" data-tab="url">URL Input</div>
            </div>
            
            <div id="search-input" class="tab-content active">
                <div class="input-group">
                    <label for="topic-search">Enter a topic to search for news articles:</label>
                    <input type="text" id="topic-search" class="search-input" 
                           placeholder="Enter a topic (e.g., 'technology', 'climate', 'sports')">
                </div>
                <button id="search-btn" class="btn">Search Articles</button>
                
                <div class="loading" id="search-loading">Searching for articles...</div>
                
                <div id="search-results-container" class="search-results-container hidden">
                    <h2>Search Results</h2>
                    <div id="search-results"></div>
                </div>
                
                <div id="selected-article-container" class="hidden">
                    <h2>Selected Article</h2>
                    <div id="selected-article" class="selected-article">
                        <div id="selected-article-title" class="selected-article-title"></div>
                        <div id="selected-article-meta" class="selected-article-meta"></div>
                        <div id="selected-article-content" class="selected-article-content"></div>
                        <button id="summarize-selected" class="btn">Generate Summary</button>
                    </div>
                </div>
            </div>
            
            <div id="text-input" class="tab-content">
                <div class="input-group">
                    <label for="article-text">Paste your article text here:</label>
                    <textarea id="article-text" placeholder="Paste the full text of a news article here..."></textarea>
                </div>
                <button id="summarize-text" class="btn">Generate Summary</button>
            </div>
            
            <div id="url-input" class="tab-content">
                <div class="input-group">
                    <label for="article-url">Enter RSS feed or article URL:</label>
                    <input type="text" id="article-url" class="url-input" 
                           placeholder="https://example.com/article.html">
                </div>
                <button id="fetch-url" class="btn">Fetch & Summarize</button>
            </div>
            
            <div class="loading" id="summary-loading">Generating summary...</div>
            
            <div class="result-container hidden" id="result-container">
                <h2>Summary Result</h2>
                <div class="result" id="summary-result"></div>
            </div>
        </div>
        
        <script>
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and content
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    
                    // Show corresponding content
                    const tabName = tab.getAttribute('data-tab');
                    document.getElementById(tabName + '-input').classList.add('active');
                    
                    // Hide results when switching tabs
                    document.getElementById('result-container').classList.add('hidden');
                });
            });
            
            // Topic search and article selection
            document.getElementById('search-btn').addEventListener('click', async () => {
                const query = document.getElementById('topic-search').value.trim();
                if (!query) {
                    alert('Please enter a topic to search for.');
                    return;
                }
                
                // Show loading message and hide previous results
                document.getElementById('search-loading').style.display = 'block';
                document.getElementById('search-results-container').classList.add('hidden');
                document.getElementById('selected-article-container').classList.add('hidden');
                document.getElementById('result-container').classList.add('hidden');
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Create search results HTML
                    const resultsContainer = document.getElementById('search-results');
                    
                    if (data.articles.length === 0) {
                        resultsContainer.innerHTML = '<div class="error-message">No articles found matching your search term. Try a different keyword.</div>';
                    } else {
                        let resultsHTML = '';
                        
                        data.articles.forEach((article, index) => {
                            resultsHTML += `
                                <div class="article-item" data-index="${index}" data-url="${article.link}">
                                    <div class="article-title">${article.title}</div>
                                    <div class="article-source">${article.source} ${article.published ? '• ' + article.published : ''}</div>
                                    <div class="article-snippet">${article.snippet}</div>
                                </div>
                            `;
                        });
                        
                        resultsContainer.innerHTML = resultsHTML;
                        
                        // Add click event to article items
                        document.querySelectorAll('.article-item').forEach(item => {
                            item.addEventListener('click', async () => {
                                // Show loading
                                document.getElementById('search-loading').style.display = 'block';
                                
                                // Get article URL
                                const articleUrl = item.getAttribute('data-url');
                                
                                try {
                                    // Fetch full article
                                    const response = await fetch('/api/get-article', {
                                        method: 'POST',
                                        headers: {
                                            'Content-Type': 'application/json',
                                        },
                                        body: JSON.stringify({ url: articleUrl })
                                    });
                                    
                                    const articleData = await response.json();
                                    
                                    if (articleData.error) {
                                        throw new Error(articleData.error);
                                    }
                                    
                                    // Display selected article
                                    document.getElementById('selected-article-title').textContent = articleData.title;
                                    document.getElementById('selected-article-meta').textContent = `Source: ${articleData.source || 'Unknown'} ${articleData.published ? '• Published: ' + articleData.published : ''}`;
                                    document.getElementById('selected-article-content').textContent = articleData.content;
                                    
                                    // Show selected article container
                                    document.getElementById('selected-article-container').classList.remove('hidden');
                                    
                                    // Store URL for summarization
                                    document.getElementById('summarize-selected').setAttribute('data-url', articleUrl);
                                } catch (error) {
                                    alert('Error fetching article: ' + error.message);
                                } finally {
                                    // Hide loading
                                    document.getElementById('search-loading').style.display = 'none';
                                }
                            });
                        });
                    }
                    
                    // Show results container
                    document.getElementById('search-results-container').classList.remove('hidden');
                } catch (error) {
                    alert('Error searching for articles: ' + error.message);
                } finally {
                    // Hide loading message
                    document.getElementById('search-loading').style.display = 'none';
                }
            });
            
            // Summarize selected article
            document.getElementById('summarize-selected').addEventListener('click', async () => {
                const articleUrl = document.getElementById('summarize-selected').getAttribute('data-url');
                
                if (!articleUrl) {
                    alert('No article selected.');
                    return;
                }
                
                // Show loading
                document.getElementById('summary-loading').style.display = 'block';
                document.getElementById('result-container').classList.add('hidden');
                
                try {
                    const response = await fetch('/api/summarize-url', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url: articleUrl })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Display summary
                    document.getElementById('summary-result').textContent = data.summary;
                    document.getElementById('result-container').classList.remove('hidden');
                } catch (error) {
                    alert('Error generating summary: ' + error.message);
                } finally {
                    // Hide loading
                    document.getElementById('summary-loading').style.display = 'none';
                }
            });
            
            // Text summarization
            document.getElementById('summarize-text').addEventListener('click', async () => {
                const text = document.getElementById('article-text').value.trim();
                if (!text) {
                    alert('Please enter some text to summarize.');
                    return;
                }
                
                document.getElementById('summary-loading').style.display = 'block';
                document.getElementById('result-container').classList.add('hidden');
                
                try {
                    const response = await fetch('/api/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    document.getElementById('summary-result').textContent = data.summary;
                    document.getElementById('result-container').classList.remove('hidden');
                } catch (error) {
                    alert('Error generating summary: ' + error.message);
                } finally {
                    document.getElementById('summary-loading').style.display = 'none';
                }
            });
            
            // URL fetching and summarization
            document.getElementById('fetch-url').addEventListener('click', async () => {
                const url = document.getElementById('article-url').value.trim();
                if (!url) {
                    alert('Please enter a URL to fetch.');
                    return;
                }
                
                document.getElementById('summary-loading').style.display = 'block';
                document.getElementById('result-container').classList.add('hidden');
                
                try {
                    const response = await fetch('/api/summarize-url', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url }),
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.message || data.error);
                    }
                    
                    document.getElementById('summary-result').textContent = data.summary;
                    document.getElementById('result-container').classList.remove('hidden');
                } catch (error) {
                    alert('Error fetching or summarizing the article: ' + error.message);
                } finally {
                    document.getElementById('summary-loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for searching articles by topic"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No search query provided'}), 400
    
    query = data['query']
    
    # Search for articles
    try:
        articles = search_articles(query)
        return jsonify({
            'query': query,
            'articles': articles,
            'count': len(articles),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-article', methods=['POST'])
def api_get_article():
    """API endpoint for fetching a specific article"""
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    # Fetch article
    try:
        article = fetch_article_from_url(url)
        
        if 'error' in article:
            return jsonify({'error': article['message']}), 400
        
        return jsonify(article)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """API endpoint for text summarization"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided for summarization'}), 400
    
    text = data['text']
    
    # Generate summary
    try:
        summary = generate_summary(text)
        return jsonify({
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'using_pretrained': USING_PRETRAINED
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize-url', methods=['POST'])
def api_summarize_url():
    """API endpoint for URL summarization"""
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    # Fetch article
    article = fetch_article_from_url(url)
    
    if 'error' in article:
        return jsonify(article), 400
    
    # Generate summary from article content
    try:
        summary = generate_summary(article['content'])
        return jsonify({
            'title': article.get('title', 'Article'),
            'url': article.get('link', url),
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'using_pretrained': USING_PRETRAINED
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_server():
    """Run the Flask server"""
    print("Loading model before starting server...")
    load_model()
    print(f"Starting server on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)

if __name__ == "__main__":
    run_server()