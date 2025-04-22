"""
Module for deploying the fine-tuned summarization model as a web API
"""

import os
import json
import time
from datetime import datetime
import torch
import feedparser
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, API_PORT, API_HOST
)

# Create Flask app
app = Flask(__name__)

# Global model and tokenizer objects
MODEL = None
TOKENIZER = None

def load_model():
    """
    Load the fine-tuned model and tokenizer
    """
    global MODEL, TOKENIZER
    
    model_dir = os.path.join(MODELS_DIR, "fine-tuned-news-summarizer")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Please fine-tune the model first.")
    
    print(f"Loading model and tokenizer from {model_dir}")
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL.to(device)
    MODEL.eval()
    
    print(f"Model loaded and ready on {device}")

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
    Fetch an article from a URL using feedparser
    """
    try:
        # Parse the feed
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
            
            return article
        else:
            # If not a feed, treat as direct article URL
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

# API Routes
@app.route('/')
def home():
    """Home page with a simple UI for the summarization service"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Summarization Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
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
            .url-input {
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
        </style>
    </head>
    <body>
        <h1>News Summarization Service</h1>
        <p>Automatically generate concise summaries of news articles using our fine-tuned AI model.</p>
        
        <div class="container">
            <div class="tabs">
                <div class="tab active" data-tab="text">Text Input</div>
                <div class="tab" data-tab="url">URL Input</div>
            </div>
            
            <div id="text-input" class="tab-content active">
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
            
            <div class="loading">Generating summary...</div>
            
            <div class="result-container" style="display: none;">
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
                });
            });
            
            // Text summarization
            document.getElementById('summarize-text').addEventListener('click', async () => {
                const text = document.getElementById('article-text').value.trim();
                if (!text) {
                    alert('Please enter some text to summarize.');
                    return;
                }
                
                document.querySelector('.loading').style.display = 'block';
                document.querySelector('.result-container').style.display = 'none';
                
                try {
                    const response = await fetch('/api/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    const data = await response.json();
                    document.getElementById('summary-result').textContent = data.summary;
                    document.querySelector('.result-container').style.display = 'block';
                } catch (error) {
                    alert('Error generating summary. Please try again.');
                    console.error('Error:', error);
                } finally {
                    document.querySelector('.loading').style.display = 'none';
                }
            });
            
            // URL fetching and summarization
            document.getElementById('fetch-url').addEventListener('click', async () => {
                const url = document.getElementById('article-url').value.trim();
                if (!url) {
                    alert('Please enter a URL to fetch.');
                    return;
                }
                
                document.querySelector('.loading').style.display = 'block';
                document.querySelector('.result-container').style.display = 'none';
                
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
                        alert(data.message || 'Error fetching article.');
                    } else {
                        document.getElementById('summary-result').textContent = data.summary;
                        document.querySelector('.result-container').style.display = 'block';
                    }
                } catch (error) {
                    alert('Error fetching or summarizing the article. Please try again.');
                    console.error('Error:', error);
                } finally {
                    document.querySelector('.loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

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
            'timestamp': datetime.now().isoformat()
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
            'timestamp': datetime.now().isoformat()
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