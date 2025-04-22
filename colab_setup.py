"""
Setup script for running the news summarization project on Google Colab
"""

import os
import sys
import subprocess
import time

def install_dependencies():
    """Install required dependencies with version control"""
    print("Installing dependencies...")
    
    # First ensure numpy<2.0.0 is installed to avoid compatibility issues
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2.0.0", "--quiet"])
    
    # Install specific transformers version known to be compatible
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.30.2", "--quiet"])
    
    # Install other dependencies
    critical_packages = [
        "torch",
        "datasets",
        "feedparser",
        "nltk",
        "pandas",
        "matplotlib",
        "seaborn",
        "sentencepiece",
        "flask",
        "rouge-score"
    ]
    
    # Verify critical packages
    print("Verifying critical packages...")
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package} successfully installed")
        except ImportError:
            print(f"❌ {package} not installed properly, attempting individual install...")
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"])
    
    print("Dependencies installation completed")

def setup_colab_environment():
    """Set up the Colab environment for the project"""
    print("Setting up Colab environment...")
    
    # Create necessary directories if they don't exist
    for dir_path in ["src/data", "src/models", "src/evaluation", "src/deployment"]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")
    
    # Download NLTK data for tokenization
    import nltk
    nltk.download('punkt', quiet=True)
    print("Downloaded NLTK punkt tokenizer")
    
    # Check for GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("GPU is not available. Using CPU for computations (this will be slow).")
    
    print("Colab environment setup completed")

def setup_flask_for_colab(port=5000):
    """Set up Flask to run on Colab"""
    from google.colab.output import eval_js
    
    # Generate a public URL
    url = eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"Access Flask app at: {url}")
    return url

def main():
    """Main function to set up Colab environment"""
    print("=" * 80)
    print("Setting up News Summarization project on Colab")
    print("=" * 80)
    
    # Install dependencies
    install_dependencies()
    
    # Set up environment
    setup_colab_environment()
    
    print("\nSetup completed successfully! You can now run the project.")
    print("To run the full pipeline: !python main.py")
    print("To run specific steps: !python main.py --step [data|train|evaluate|deploy]")
    print("\nFor deployment with public access:")
    print("from colab_setup import setup_flask_for_colab")
    print("setup_flask_for_colab()")
    
if __name__ == "__main__":
    main()