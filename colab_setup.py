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
    
    # Then install the rest of the dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    
    # Verify critical packages
    print("Verifying critical packages...")
    for package in ["torch", "transformers", "feedparser", "flask", "pyngrok"]:
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

def setup_ngrok_for_deployment(port=5000, auth_token=None):
    """Set up ngrok for exposing the Flask deployment on Colab"""
    from pyngrok import ngrok, conf
    
    # Set ngrok auth token if provided
    if auth_token:
        conf.get_default().auth_token = auth_token
    
    # Get public URL and return
    public_url = ngrok.connect(port)
    print(f"Public URL for Flask app: {public_url}")
    return public_url

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
    print("\nFor deployment with public access, add this to your notebook:")
    print("from colab_setup import setup_ngrok_for_deployment")
    print("public_url = setup_ngrok_for_deployment()")
    
if __name__ == "__main__":
    main()