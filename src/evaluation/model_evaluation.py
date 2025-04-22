"""
Module for evaluating the fine-tuned summarization model
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, TEST_FILE, EVALUATION_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    METRICS, BATCH_SIZE
)

# Create evaluation directory if it doesn't exist
os.makedirs(EVALUATION_DIR, exist_ok=True)

def load_model_and_tokenizer():
    """
    Load the fine-tuned model and tokenizer
    """
    model_dir = os.path.join(MODELS_DIR, "fine-tuned-news-summarizer")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Please fine-tune the model first.")
    
    print(f"Loading model and tokenizer from {model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def load_test_data():
    """
    Load the test dataset
    """
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(test_data)} test examples from {TEST_FILE}")
    return test_data

def generate_summaries(model, tokenizer, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate summaries for the test data
    """
    model.to(device)
    model.eval()
    
    summaries = []
    
    # Process in batches
    batch_size = BATCH_SIZE
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating summaries"):
        batch = test_data[i:i+batch_size]
        texts = [item['text'] for item in batch]
        
        # Tokenize
        inputs = tokenizer(
            texts, 
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        
        # Generate summaries
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            
        # Decode the generated summaries
        generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Store results
        for j, gen_summary in enumerate(generated_summaries):
            summaries.append({
                'article': batch[j]['text'],
                'reference_summary': batch[j]['summary'],
                'generated_summary': gen_summary,
                'title': batch[j]['title'],
                'source': batch[j]['source']
            })
    
    return summaries

def evaluate_summaries(summaries):
    """
    Evaluate the generated summaries using various metrics
    """
    # Load metrics
    rouge = load_metric("rouge")
    
    # Prepare references and predictions
    references = [item['reference_summary'] for item in summaries]
    predictions = [item['generated_summary'] for item in summaries]
    
    # Calculate ROUGE scores
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    # Format results
    formatted_results = {
        'rouge1': {
            'precision': rouge_results['rouge1'].precision,
            'recall': rouge_results['rouge1'].recall,
            'f1': rouge_results['rouge1'].fmeasure
        },
        'rouge2': {
            'precision': rouge_results['rouge2'].precision,
            'recall': rouge_results['rouge2'].recall,
            'f1': rouge_results['rouge2'].fmeasure
        },
        'rougeL': {
            'precision': rouge_results['rougeL'].precision,
            'recall': rouge_results['rougeL'].recall,
            'f1': rouge_results['rougeL'].fmeasure
        }
    }
    
    # Calculate summary length statistics
    ref_lengths = [len(ref.split()) for ref in references]
    pred_lengths = [len(pred.split()) for pred in predictions]
    
    length_stats = {
        'reference_length': {
            'min': min(ref_lengths),
            'max': max(ref_lengths),
            'avg': sum(ref_lengths) / len(ref_lengths),
            'std': np.std(ref_lengths)
        },
        'generated_length': {
            'min': min(pred_lengths),
            'max': max(pred_lengths),
            'avg': sum(pred_lengths) / len(pred_lengths),
            'std': np.std(pred_lengths)
        }
    }
    
    # Combine results
    evaluation_results = {
        'metrics': formatted_results,
        'length_stats': length_stats
    }
    
    return evaluation_results

def visualize_results(summaries, evaluation_results):
    """
    Create visualizations for the evaluation results
    """
    # Create output directory
    viz_dir = os.path.join(EVALUATION_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Plot ROUGE scores
    plt.figure(figsize=(10, 6))
    metrics = ['rouge1', 'rouge2', 'rougeL']
    metrics_data = []
    
    for metric in metrics:
        metrics_data.append(evaluation_results['metrics'][metric]['precision'])
        metrics_data.append(evaluation_results['metrics'][metric]['recall'])
        metrics_data.append(evaluation_results['metrics'][metric]['f1'])
    
    labels = []
    for m in metrics:
        for t in ['Precision', 'Recall', 'F1']:
            labels.append(f"{m} {t}")
    
    plt.bar(labels, metrics_data, color='skyblue')
    plt.title('ROUGE Scores')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'rouge_scores.png'))
    
    # 2. Summary length comparison
    plt.figure(figsize=(10, 6))
    ref_lengths = [len(item['reference_summary'].split()) for item in summaries]
    gen_lengths = [len(item['generated_summary'].split()) for item in summaries]
    
    plt.hist(ref_lengths, alpha=0.6, label='Reference', bins=20, color='blue')
    plt.hist(gen_lengths, alpha=0.6, label='Generated', bins=20, color='orange')
    plt.title('Summary Length Distribution')
    plt.xlabel('Length (words)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'length_distribution.png'))
    
    # 3. Performance by source
    plt.figure(figsize=(12, 7))
    sources = {}
    
    for item in summaries:
        source = item['source']
        if source not in sources:
            sources[source] = {'ref_lengths': [], 'gen_lengths': [], 'ref_texts': [], 'gen_texts': []}
        
        sources[source]['ref_lengths'].append(len(item['reference_summary'].split()))
        sources[source]['gen_lengths'].append(len(item['generated_summary'].split()))
        sources[source]['ref_texts'].append(item['reference_summary'])
        sources[source]['gen_texts'].append(item['generated_summary'])
    
    source_names = list(sources.keys())
    avg_ref_lengths = [np.mean(sources[s]['ref_lengths']) for s in source_names]
    avg_gen_lengths = [np.mean(sources[s]['gen_lengths']) for s in source_names]
    
    x = np.arange(len(source_names))
    width = 0.35
    
    plt.bar(x - width/2, avg_ref_lengths, width, label='Reference', color='blue')
    plt.bar(x + width/2, avg_gen_lengths, width, label='Generated', color='orange')
    plt.title('Average Summary Length by Source')
    plt.xlabel('Source')
    plt.ylabel('Average Length (words)')
    plt.xticks(x, source_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'length_by_source.png'))
    
    # Save sample summaries
    with open(os.path.join(EVALUATION_DIR, 'sample_summaries.txt'), 'w', encoding='utf-8') as f:
        f.write("Sample Generated Summaries vs. References\n")
        f.write("=" * 80 + "\n\n")
        
        # Pick 10 random samples
        indices = np.random.choice(len(summaries), min(10, len(summaries)), replace=False)
        
        for idx in indices:
            item = summaries[idx]
            f.write(f"Title: {item['title']}\n")
            f.write(f"Source: {item['source']}\n")
            f.write(f"Reference Summary: {item['reference_summary']}\n")
            f.write(f"Generated Summary: {item['generated_summary']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Saved visualizations to {viz_dir}")

def save_evaluation_results(summaries, evaluation_results):
    """
    Save the evaluation results to files
    """
    # Save JSON results
    results_file = os.path.join(EVALUATION_DIR, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save all summaries
    summaries_file = os.path.join(EVALUATION_DIR, 'generated_summaries.jsonl')
    with open(summaries_file, 'w', encoding='utf-8') as f:
        for item in summaries:
            f.write(json.dumps(item) + '\n')
    
    # Generate a text report
    report_file = os.path.join(EVALUATION_DIR, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("News Summarization Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # ROUGE scores
        f.write("ROUGE Scores:\n")
        for metric, values in evaluation_results['metrics'].items():
            f.write(f"  {metric}:\n")
            for score_type, score in values.items():
                f.write(f"    {score_type}: {score:.4f}\n")
        f.write("\n")
        
        # Length statistics
        f.write("Summary Length Statistics:\n")
        for summary_type, stats in evaluation_results['length_stats'].items():
            f.write(f"  {summary_type}:\n")
            for stat_name, value in stats.items():
                if stat_name in ['min', 'max']:
                    f.write(f"    {stat_name}: {int(value)}\n")
                else:
                    f.write(f"    {stat_name}: {value:.2f}\n")
        
        f.write("\n")
        f.write("Sample summaries can be found in 'sample_summaries.txt'")
    
    print(f"Saved evaluation results to {EVALUATION_DIR}")

def main():
    """
    Main function to execute the model evaluation pipeline
    """
    print("Starting model evaluation pipeline...")
    
    # Step 1: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 2: Load test data
    test_data = load_test_data()
    
    # Step 3: Generate summaries
    print("Generating summaries for test data...")
    summaries = generate_summaries(model, tokenizer, test_data)
    
    # Step 4: Evaluate the generated summaries
    print("Evaluating generated summaries...")
    evaluation_results = evaluate_summaries(summaries)
    
    # Step 5: Visualize results
    print("Creating visualizations...")
    visualize_results(summaries, evaluation_results)
    
    # Step 6: Save evaluation results
    save_evaluation_results(summaries, evaluation_results)
    
    print("Model evaluation pipeline completed successfully!")
    
    return summaries, evaluation_results

if __name__ == "__main__":
    main()