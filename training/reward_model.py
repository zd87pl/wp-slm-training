#!/usr/bin/env python3
"""
WordPress Reward Model for RLAIF Training
Trains a reward model to predict quality scores for WordPress responses
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, AutoConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import track
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

@dataclass
class RewardModelConfig:
    """Configuration for WordPress Reward Model"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 1024
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    output_dir: str = "./models/wp-reward-model"

class WordPressRewardDataset(Dataset):
    """Dataset for training WordPress reward model"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 1024,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process data
        console.print(f"[cyan]📁 Loading reward dataset: {data_path}[/cyan]")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Filter and validate data
        self.data = []
        for item in raw_data:
            if all(key in item for key in ["prompt", "response", "reward_score"]):
                if 0.0 <= item["reward_score"] <= 1.0:  # Valid score range
                    self.data.append(item)
        
        # Train/validation split (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        split_idx = int(0.8 * len(self.data))
        
        if split == "train":
            self.data = [self.data[i] for i in indices[:split_idx]]
        else:
            self.data = [self.data[i] for i in indices[split_idx:]]
        
        console.print(f"[green]✅ Loaded {len(self.data)} samples for {split} set[/green]")
        
        # Calculate statistics
        scores = [item["reward_score"] for item in self.data]
        console.print(f"[blue]📊 Score stats - Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}[/blue]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input text (prompt + response)
        input_text = self._format_input(item["prompt"], item["response"])
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["reward_score"], dtype=torch.float32),
            "criteria_scores": torch.tensor([
                item.get("criteria_scores", {}).get(criterion, 5.0) / 10.0  # Normalize to 0-1
                for criterion in ["code_quality", "wp_accuracy", "security", "clarity", "completeness"]
            ], dtype=torch.float32)
        }
    
    def _format_input(self, prompt: str, response: str) -> str:
        """Format prompt-response pair for reward model input"""
        return f"""<|system|>
You are evaluating a WordPress development response for quality and accuracy.

<|user|>
{prompt}

<|assistant|>
{response}

<|reward|>
"""

class WordPressRewardModel(nn.Module):
    """WordPress-specific reward model with multi-criteria scoring"""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # Load base model configuration
        model_config = AutoConfig.from_pretrained(config.base_model)
        
        # Modify for regression (single output)
        model_config.num_labels = 1
        model_config.problem_type = "regression"
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            config=model_config,
            torch_dtype=torch.float16
        )
        
        # Add multi-criteria head
        hidden_size = self.base_model.config.hidden_size
        
        # Overall reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output 0-1 range
        )
        
        # Multi-criteria head (optional, for analysis)
        self.criteria_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5),  # 5 criteria
            nn.Sigmoid()  # Output 0-1 range for each criterion
        )
        
        # Replace the classifier head
        self.base_model.classifier = self.reward_head
        
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Get base model outputs (without classifier)
        outputs = self.base_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use mean pooling if no pooler
            last_hidden_states = outputs.last_hidden_state
            pooled_output = torch.mean(last_hidden_states, dim=1)
        
        # Get reward score
        reward_score = self.reward_head(pooled_output).squeeze(-1)
        
        # Get criteria scores
        criteria_scores = self.criteria_head(pooled_output)
        
        loss = None
        if labels is not None:
            # Regression loss for overall reward
            loss_fct = nn.MSELoss()
            loss = loss_fct(reward_score, labels.float())
        
        return {
            "loss": loss,
            "logits": reward_score,
            "reward_scores": reward_score,
            "criteria_scores": criteria_scores,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

class RewardModelTrainer:
    """Trainer for WordPress Reward Model"""
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        console.print(f"[cyan]🔧 Initialized reward model trainer on {self.device}[/cyan]")
    
    def load_datasets(self, train_data_path: str, eval_data_path: str = None):
        """Load training and evaluation datasets"""
        self.train_dataset = WordPressRewardDataset(
            train_data_path, self.tokenizer, self.config.max_length, "train"
        )
        
        if eval_data_path:
            self.eval_dataset = WordPressRewardDataset(
                eval_data_path, self.tokenizer, self.config.max_length, "eval"
            )
        else:
            self.eval_dataset = WordPressRewardDataset(
                train_data_path, self.tokenizer, self.config.max_length, "eval"
            )
    
    def initialize_model(self):
        """Initialize the reward model"""
        console.print("[cyan]🏗️ Initializing WordPress reward model[/cyan]")
        
        self.model = WordPressRewardModel(self.config)
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            console.print("[cyan]🔧 Applying LoRA to reward model[/cyan]")
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA config
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="SEQUENCE_CLASSIFICATION"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.to(self.device)
        
    def train(self):
        """Train the reward model"""
        console.print("[cyan]🚀 Starting reward model training[/cyan]")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mse",
            greater_is_better=False,
            report_to=None,  # Disable wandb
            remove_unused_columns=False,
        )
        
        # Custom trainer with regression metrics
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        console.print(f"[green]✅ Reward model training completed![/green]")
        console.print(f"[cyan]📁 Model saved to: {self.config.output_dir}[/cyan]")
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics for regression"""
        predictions, labels = eval_pred
        predictions = predictions.squeeze() if predictions.ndim > 1 else predictions
        
        # Clip predictions to valid range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        # Accuracy within tolerance
        tolerance = 0.1
        accuracy = np.mean(np.abs(labels - predictions) <= tolerance)
        
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "accuracy_10": accuracy,
        }
    
    def evaluate_model(self, test_data_path: str = None):
        """Evaluate trained model performance"""
        if test_data_path:
            test_dataset = WordPressRewardDataset(
                test_data_path, self.tokenizer, self.config.max_length, "eval"
            )
        else:
            test_dataset = self.eval_dataset
        
        # Load best model
        model_path = Path(self.config.output_dir)
        if not model_path.exists():
            console.print(f"[red]❌ Model not found at {model_path}[/red]")
            return
        
        console.print("[cyan]📊 Evaluating reward model performance[/cyan]")
        
        # Run inference
        predictions = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(track(DataLoader(test_dataset, batch_size=8), description="Evaluating")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_predictions = outputs["reward_scores"].cpu().numpy()
                batch_labels = batch["labels"].numpy()
                
                predictions.extend(batch_predictions)
                labels.extend(batch_labels)
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        console.print(f"[green]📊 Evaluation Results:[/green]")
        console.print(f"[blue]MSE: {mse:.4f}[/blue]")
        console.print(f"[blue]RMSE: {np.sqrt(mse):.4f}[/blue]")
        console.print(f"[blue]MAE: {mae:.4f}[/blue]")
        console.print(f"[blue]R²: {r2:.4f}[/blue]")
        
        # Generate evaluation plots
        self._generate_evaluation_plots(predictions, labels)
        
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "predictions": predictions.tolist(),
            "labels": labels.tolist()
        }
    
    def _generate_evaluation_plots(self, predictions: np.ndarray, labels: np.ndarray):
        """Generate evaluation plots"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            pass
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(labels, predictions, alpha=0.6, color='blue')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Scores')
        axes[0, 0].set_ylabel('Predicted Scores')
        axes[0, 0].set_title('Predicted vs Actual Scores')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = predictions - labels
        axes[0, 1].scatter(labels, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Scores')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual Scores')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of predictions
        axes[1, 0].hist(predictions, bins=20, alpha=0.7, color='blue', label='Predicted')
        axes[1, 0].hist(labels, bins=20, alpha=0.7, color='red', label='Actual')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config.output_dir) / "evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]📈 Evaluation plots saved: {plot_path}[/green]")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train WordPress Reward Model")
    parser.add_argument("--data", required=True, help="Path to reward dataset JSON file")
    parser.add_argument("--output", default="./models/wp-reward-model", help="Output directory")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing model")
    
    args = parser.parse_args()
    
    # Create config
    config = RewardModelConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora,
        output_dir=args.output
    )
    
    # Initialize trainer
    trainer = RewardModelTrainer(config)
    
    if args.evaluate_only:
        # Only evaluate
        trainer.initialize_model()
        results = trainer.evaluate_model(args.data)
        
        # Save results
        results_path = Path(args.output) / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]📊 Results saved: {results_path}[/green]")
    else:
        # Train model
        trainer.load_datasets(args.data)
        trainer.initialize_model()
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate_model()
        
        # Save results
        results_path = Path(args.output) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()