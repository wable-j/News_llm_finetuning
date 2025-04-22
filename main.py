"""
Main script for running the entire news summarization workflow
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_data_processing():
    """Run the data processing pipeline"""
    print_section("DATA PROCESSING")
    from src.data.data_processing import main as process_data
    return process_data()

def run_model_training():
    """Run the model training pipeline"""
    print_section("MODEL TRAINING")
    from src.models.model_training import main as train_model
    return train_model()

def run_model_evaluation():
    """Run the model evaluation pipeline"""
    print_section("MODEL EVALUATION")
    from src.evaluation.model_evaluation import main as evaluate_model
    return evaluate_model()

def run_deployment():
    """Run the model deployment server"""
    print_section("MODEL DEPLOYMENT")
    from src.deployment.app import run_server
    return run_server()

def main():
    """Main function to run the entire workflow or specific components"""
    parser = argparse.ArgumentParser(description="News Summarization Model Fine-tuning Workflow")
    
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["all", "data", "train", "evaluate", "deploy"],
        default="all",
        help="Specify which step of the workflow to run"
    )
    
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    print(f"Started workflow at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.step == "all":
            print("Running complete workflow: data → train → evaluate → deploy")
            run_data_processing()
            run_model_training()
            run_model_evaluation()
            run_deployment()
        elif args.step == "data":
            run_data_processing()
        elif args.step == "train":
            run_model_training()
        elif args.step == "evaluate":
            run_model_evaluation()
        elif args.step == "deploy":
            run_deployment()
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nError in workflow: {str(e)}")
        raise
    finally:
        # Calculate and display total run time
        end_time = datetime.now()
        run_time = end_time - start_time
        print(f"\nWorkflow completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total run time: {run_time}")

if __name__ == "__main__":
    main()