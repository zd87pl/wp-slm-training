#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training script for WordPress SLM.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from trl import DPOTrainer
from datasets import load_dataset, Dataset
import tyro
from rich.console import Console

console = Console()

@dataclass
class ScriptArguments:
    """Script arguments for DPO training."""
    config: str = field(
        default="training/config/dpo.yaml",
        metadata={"help": "Path to training configuration YAML file"}
    )
    policy_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to policy model (defaults to config value)"}
    )
    train_prefs: str = field(
        default="data/prefs/train.jsonl",
        metadata={"help": "Path to training preferences JSONL file"}
    )
    eval_prefs: str = field(
        default="data/prefs/val.jsonl",
        metadata={"help": "Path to evaluation preferences JSONL file"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Resume training from checkpoint"}
    )


class WPDPOTrainer:
    def __init__(self, config_path: str):
        """Initialize DPO trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(f"[cyan]Loaded DPO configuration from {config_path}[/cyan]")
        console.print(f"[cyan]Using device: {self.device}[/cyan]")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_models_and_tokenizer(self, policy_model_path: Optional[str] = None):
        """Set up policy and reference models with tokenizer."""
        console.print("[yellow]Loading models and tokenizer...[/yellow]")
        
        # Use provided path or config default
        model_path = policy_model_path or self.config['model_name_or_path']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Determine if using quantization
        use_4bit = self.config.get('peft', {}).get('use_peft', True)
        
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None
            
        # Check if the model is already a PEFT model
        peft_config_path = Path(model_path) / "adapter_config.json"
        is_peft_model = peft_config_path.exists()
        
        if is_peft_model:
            console.print("[cyan]Loading PEFT model as policy...[/cyan]")
            
            # Load the PEFT config to get base model
            with open(peft_config_path, 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config['base_model_name_or_path']
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.config['training'].get('bf16') else torch.float16
            )
            
            # Load as policy model
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # For DPO, we need the reference model (original base)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.config['training'].get('bf16') else torch.float16
            )
            
        else:
            console.print("[cyan]Loading full model...[/cyan]")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.config['training'].get('bf16') else torch.float16
            )
            
            # Reference model is the same initially
            self.ref_model = self.model
            
        # Apply new LoRA for DPO if configured
        if self.config.get('peft', {}).get('use_peft', False) and not is_peft_model:
            console.print("[yellow]Applying new LoRA configuration for DPO...[/yellow]")
            
            peft_config = self.config['peft']
            lora_config = LoraConfig(
                r=peft_config['lora_r'],
                lora_alpha=peft_config['lora_alpha'],
                lora_dropout=peft_config['lora_dropout'],
                target_modules=peft_config['target_modules'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        console.print("[green]Models and tokenizer loaded successfully[/green]")
        
    def load_preference_dataset(self, train_file: str, eval_file: str):
        """Load preference datasets."""
        console.print("[yellow]Loading preference datasets...[/yellow]")
        
        # Load raw data
        train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))
                    
        eval_data = []
        with open(eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
                    
        # Format for DPO
        def format_preferences(data: List[Dict]) -> Dict[str, List]:
            formatted = {
                'prompt': [],
                'chosen': [],
                'rejected': []
            }
            
            prompt_template = self.config.get('prompt_template', '{prompt}')
            
            for item in data:
                # Format prompt
                prompt = prompt_template.format(prompt=item['prompt'])
                
                # Add to formatted data
                formatted['prompt'].append(prompt)
                formatted['chosen'].append(item['chosen'])
                formatted['rejected'].append(item['rejected'])
                
            return formatted
            
        # Create datasets
        train_formatted = format_preferences(train_data)
        eval_formatted = format_preferences(eval_data)
        
        self.train_dataset = Dataset.from_dict(train_formatted)
        self.eval_dataset = Dataset.from_dict(eval_formatted)
        
        console.print(f"[green]Loaded {len(self.train_dataset)} training preferences[/green]")
        console.print(f"[green]Loaded {len(self.eval_dataset)} evaluation preferences[/green]")
        
    def setup_training_args(self):
        """Set up training arguments for DPO."""
        training_config = self.config['training']
        
        # Standard training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_ratio=training_config['warmup_ratio'],
            optim=training_config['optim'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_strategy=training_config['save_strategy'],
            evaluation_strategy=training_config['evaluation_strategy'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            bf16=training_config.get('bf16', False),
            tf32=training_config.get('tf32', True),
            seed=training_config['seed'],
            report_to=training_config.get('report_to', ['tensorboard']),
            remove_unused_columns=training_config.get('remove_unused_columns', False),
        )
        
        return training_args
        
    def train(self, train_file: str, eval_file: str, 
             policy_model_path: Optional[str] = None,
             resume_from_checkpoint: Optional[str] = None):
        """Run DPO training."""
        # Set up models and tokenizer
        self.setup_models_and_tokenizer(policy_model_path)
        
        # Load datasets
        self.load_preference_dataset(train_file, eval_file)
        
        # Set up training arguments
        training_args = self.setup_training_args()
        
        # DPO specific config
        dpo_config = self.config.get('dpo', {})
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model if not dpo_config.get('reference_free', False) else None,
            args=training_args,
            beta=dpo_config.get('beta', 0.1),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_seq_length'],
            max_prompt_length=self.config['training'].get('max_prompt_length', 512),
            label_smoothing=dpo_config.get('label_smoothing', 0.0),
            loss_type=dpo_config.get('loss_type', 'sigmoid'),
        )
        
        # Start training
        console.print("[bold green]Starting DPO training...[/bold green]")
        
        dpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        console.print("[yellow]Saving final model...[/yellow]")
        dpo_trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Save training stats
        self._save_training_stats(dpo_trainer)
        
        console.print("[bold green]DPO training completed successfully![/bold green]")
        
    def _save_training_stats(self, trainer):
        """Save training statistics and metrics."""
        stats = {
            "final_train_loss": trainer.state.log_history[-1].get('train_loss', None),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', None),
            "total_steps": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "dpo_config": self.config.get('dpo', {})
        }
        
        stats_file = Path(self.config['output_dir']) / "dpo_training_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        
        console.print(f"[green]Training stats saved to {stats_file}[/green]")
        if stats['final_eval_loss']:
            console.print(f"Final eval loss: {stats['final_eval_loss']:.4f}")


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Initialize trainer
    trainer = WPDPOTrainer(script_args.config)
    
    # Run training
    trainer.train(
        script_args.train_prefs,
        script_args.eval_prefs,
        script_args.policy_model,
        script_args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()