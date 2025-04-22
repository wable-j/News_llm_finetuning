"""
Module for fine-tuning a Hugging Face model for news summarization
with compatibility for Colab environment
"""

import os
import json
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, WARMUP_RATIO,
    GRADIENT_ACCUMULATION_STEPS, TRAIN_FILE, VAL_FILE, TEST_FILE,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, MODELS_DIR, RANDOM_SEED
)

def load_datasets():
    """
    Load the train, validation, and test datasets
    """
    # Load data from JSONL files
    dataset_dict = {}
    for split, file_path in [
        ('train', TRAIN_FILE),
        ('validation', VAL_FILE),
        ('test', TEST_FILE)
    ]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            # Convert to datasets format
            dataset = Dataset.from_dict({
                'text': [item['text'] for item in data],
                'summary': [item['summary'] for item in data],
                'title': [item['title'] for item in data],
                'source': [item['source'] for item in data]
            })
            
            dataset_dict[split] = dataset
            print(f"Loaded {len(dataset)} examples for {split} split")
            
        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")
    
    return DatasetDict(dataset_dict)

def prepare_dataset_for_training(dataset_dict, tokenizer):
    """
    Tokenize datasets for training
    """
    def preprocess_function(examples):
        inputs = examples['text']
        targets = examples['summary']
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        model_inputs['labels'] = labels['input_ids']
        
        # Replace padding token id with -100 so they are ignored in computing loss
        labels_with_ignore_index = []
        for label in labels['input_ids']:
            labels_with_ignore_index.append([
                -100 if token == tokenizer.pad_token_id else token for token in label
            ])
        
        model_inputs['labels'] = labels_with_ignore_index
        
        return model_inputs
    
    # Apply preprocessing to all splits
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=['text', 'summary', 'title', 'source']
    )
    
    return tokenized_datasets

# Define a custom ROUGE metric for evaluation
class RougeMetric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            print("Installing rouge-score package...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "rouge-score", "--quiet"])
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        
        # Decode generated summaries
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        rouge1 = rouge2 = rougeL = 0.0
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = self.scorer.score(label, pred)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
        
        num_samples = len(decoded_preds)
        rouge1 /= num_samples
        rouge2 /= num_samples
        rougeL /= num_samples
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL
        }

# Custom generation function for summarization
def generate_summaries(model, tokenizer, batch, device):
    inputs = tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    
    summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return summaries

def fine_tune_model(tokenized_datasets):
    """
    Fine-tune the model on our dataset
    """
    # Load pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Set up data collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=MODEL_NAME
    )
    
    # Setup metric for evaluation
    rouge_metric = RougeMetric(tokenizer)
    
    # Define training arguments
    os.makedirs(MODELS_DIR, exist_ok=True)
    output_dir = os.path.join(MODELS_DIR, "fine-tuned-news-summarizer")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        seed=RANDOM_SEED,
        report_to="none",  # Set to "wandb" to use Weights & Biases
    )
    
    # Define compute_metrics function for the Trainer
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Generate summaries for the validation set
        if isinstance(predictions, tuple):
            # In some versions of transformers, predictions might be a tuple
            predictions = predictions[0]
        
        # Compute ROUGE metrics
        return rouge_metric.compute_metrics((predictions, labels))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting model fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model fine-tuning completed. Model saved to {output_dir}")
    
    return model, tokenizer, trainer

def main():
    """
    Main function to execute the model fine-tuning pipeline
    """
    print("Starting model fine-tuning pipeline...")
    
    # Install required packages for training
    print("Ensuring required packages are installed...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.30.2", "datasets", "rouge-score", "--quiet"])
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Step 1: Load datasets
    dataset_dict = load_datasets()
    
    # Step 2: Prepare datasets for training
    tokenized_datasets = prepare_dataset_for_training(dataset_dict, tokenizer)
    
    # Step 3: Fine-tune the model
    model, tokenizer, trainer = fine_tune_model(tokenized_datasets)
    
    print("Model fine-tuning pipeline completed successfully!")
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    main()