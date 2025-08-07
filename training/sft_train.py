#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) training script for WordPress SLM.
Supports QLoRA and full fine-tuning.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
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
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
    TRL_COMPLETION_COLLATOR_AVAILABLE = True
except ImportError:
    # TRL DataCollatorForCompletionOnlyLM not available
    DataCollatorForCompletionOnlyLM = None
    TRL_COMPLETION_COLLATOR_AVAILABLE = False
from datasets import load_dataset
import tyro
from rich.console import Console

console = Console()

@dataclass
class ScriptArguments:
    """Script arguments for SFT training."""
    config: str = field(
        metadata={"help": "Path to training configuration YAML file"}
    )
    train_file: str = field(
        metadata={"help": "Path to training JSONL file"}
    )
    eval_file: str = field(
        metadata={"help": "Path to evaluation JSONL file"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Resume training from checkpoint"}
    )


class WPSFTTrainer:
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(f"[cyan]Loaded configuration from {config_path}[/cyan]")
        console.print(f"[cyan]Using device: {self.device}[/cyan]")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_model_and_tokenizer(self):
        """Set up model and tokenizer with quantization if specified."""
        console.print("[yellow]Loading model and tokenizer...[/yellow]")
        
        # Set up quantization config if using 4-bit
        bnb_config = None
        if self.config.get('quantization', {}).get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    self.config['quantization'].get('bnb_4bit_compute_dtype', 'float16')
                ),
                bnb_4bit_use_double_quant=self.config['quantization'].get(
                    'bnb_4bit_use_double_quant', True
                ),
                bnb_4bit_quant_type=self.config['quantization'].get(
                    'bnb_4bit_quant_type', 'nf4'
                )
            )
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config['training'].get('bf16', False) else torch.float16
        )
        
        # Prepare model for training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=self.config['training'].get(
                    'gradient_checkpointing', True
                )
            )
            
        console.print("[green]Model and tokenizer loaded successfully[/green]")
        
    def setup_peft(self):
        """Set up PEFT (LoRA) if configured."""
        if 'lora' not in self.config:
            console.print("[yellow]No LoRA configuration found, using full fine-tuning[/yellow]")
            return
            
        console.print("[yellow]Setting up LoRA...[/yellow]")
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora'].get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def load_datasets(self, train_file: str, eval_file: str):
        """Load training and evaluation datasets."""
        console.print("[yellow]Loading datasets...[/yellow]")
        
        # Load datasets
        self.train_dataset = load_dataset(
            'json', 
            data_files=train_file,
            split='train'
        )
        
        self.eval_dataset = load_dataset(
            'json',
            data_files=eval_file,
            split='train'  # datasets library quirk
        )
        
        console.print(f"[green]Loaded {len(self.train_dataset)} training examples[/green]")
        console.print(f"[green]Loaded {len(self.eval_dataset)} evaluation examples[/green]")
        
    def formatting_func(self, example):
        """Format training examples for SFT training with robust key handling"""
        try:
            # Debug: Print example structure for first few samples
            if not hasattr(self, '_debug_printed'):
                console.print(f"[blue]Dataset example keys: {list(example.keys())}[/blue]")
                console.print(f"[blue]Dataset example structure: {example}[/blue]")
                self._debug_printed = True
            
            # Handle different key variations commonly found in datasets
            prompt_keys = ['prompt', 'input', 'question', 'instruction', 'text']
            response_keys = ['response', 'output', 'answer', 'completion', 'target']
            
            prompt = None
            response = None
            
            # Find prompt
            for key in prompt_keys:
                if key in example and example[key]:
                    prompt = example[key]
                    break
            
            # Find response
            for key in response_keys:
                if key in example and example[key]:
                    response = example[key]
                    break
            
            # Handle conversation format (messages)
            if 'messages' in example:
                messages = example['messages']
                if isinstance(messages, list):
                    formatted_parts = []
                    for msg in messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if role == 'user':
                            formatted_parts.append(f"Human: {content}")
                        elif role == 'assistant':
                            formatted_parts.append(f"Assistant: {content}")
                        else:
                            formatted_parts.append(f"{role.title()}: {content}")
                    return "\n\n".join(formatted_parts)
            
            # Handle direct text field
            if 'text' in example:
                return example['text']
            
            # Handle prompt-response format
            if prompt and response:
                return f"Human: {prompt}\n\nAssistant: {response}"
            elif prompt:
                return f"Human: {prompt}\n\nAssistant: "
            else:
                # Fallback: use any available text content
                available_text = []
                for key, value in example.items():
                    if isinstance(value, str) and value.strip():
                        available_text.append(f"{key}: {value}")
                
                if available_text:
                    return "\n".join(available_text)
                else:
                    console.print(f"[red]Warning: No usable text found in example: {list(example.keys())}[/red]")
                    return "Human: No content available\n\nAssistant: I don't have enough information to respond."
                
        except Exception as e:
            console.print(f"[red]Error formatting example: {e}[/red]")
            console.print(f"[yellow]Example type: {type(example)}[/yellow]")
            console.print(f"[yellow]Example keys: {list(example.keys()) if hasattr(example, 'keys') else 'No keys method'}[/yellow]")
            return "Human: Error processing example\n\nAssistant: I encountered an error processing this example."
            
    def setup_training_args(self):
        """Set up training arguments."""
        training_config = self.config['training']
        
        # Build arguments dynamically to handle version differences
        args = {
            'output_dir': self.config['output_dir'],
            'num_train_epochs': training_config['num_train_epochs'],
            'per_device_train_batch_size': training_config['per_device_train_batch_size'],
            'per_device_eval_batch_size': training_config['per_device_eval_batch_size'],
            'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],
            'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
            'learning_rate': training_config['learning_rate'],
            'lr_scheduler_type': training_config['lr_scheduler_type'],
            'warmup_ratio': training_config['warmup_ratio'],
            'optim': training_config['optim'],
            'weight_decay': training_config['weight_decay'],
            'max_grad_norm': training_config['max_grad_norm'],
            'logging_steps': training_config['logging_steps'],
            'save_steps': training_config['save_steps'],
            'save_strategy': training_config['save_strategy'],
            'save_total_limit': training_config['save_total_limit'],
            'load_best_model_at_end': training_config['load_best_model_at_end'],
            'metric_for_best_model': training_config['metric_for_best_model'],
            'greater_is_better': training_config['greater_is_better'],
            'seed': training_config['seed'],
            'push_to_hub': training_config.get('push_to_hub', False),
            'remove_unused_columns': False,
            'report_to': [],  # Disable all reporting (TensorBoard, wandb, etc) to avoid dependency issues
        }
        
        # Handle evaluation strategy with version compatibility
        eval_strategy_value = training_config.get('eval_strategy', 'steps')
        try:
            # Try the newer parameter name first
            args['eval_strategy'] = eval_strategy_value
        except:
            try:
                # Fall back to older parameter name
                args['evaluation_strategy'] = eval_strategy_value
            except:
                # Skip evaluation strategy if neither works
                console.print("[yellow]Warning: Could not set evaluation strategy[/yellow]")
        
        # Add eval_steps only if we have evaluation
        if 'eval_strategy' in args or 'evaluation_strategy' in args:
            args['eval_steps'] = training_config['eval_steps']
        
        # Handle precision settings
        if training_config.get('bf16', False):
            args['bf16'] = True
        else:
            args['fp16'] = training_config.get('fp16', True)
            
        if training_config.get('tf32', True):
            args['tf32'] = True
            
        # Handle reporting
        report_to = training_config.get('report_to', ['tensorboard'])
        if report_to:
            args['report_to'] = report_to
            
        return TrainingArguments(**args)
        
    def train(self, train_file: str, eval_file: str, resume_from_checkpoint: Optional[str] = None):
        """Run the training process."""
        # Set up model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Set up PEFT if configured
        self.setup_peft()
        
        # Load datasets
        self.load_datasets(train_file, eval_file)
        
        # Set up training arguments
        training_args = self.setup_training_args()
        
        # Set up data collator for completion-only training
        if TRL_COMPLETION_COLLATOR_AVAILABLE:
            response_template = "\nASSISTANT:"
            try:
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    tokenizer=self.tokenizer,
                    mlm=False
                )
                console.print("[green]Using TRL DataCollatorForCompletionOnlyLM[/green]")
            except TypeError:
                # Handle different TRL versions with different parameters
                try:
                    data_collator = DataCollatorForCompletionOnlyLM(
                        response_template=response_template,
                        tokenizer=self.tokenizer
                    )
                    console.print("[green]Using TRL DataCollatorForCompletionOnlyLM (simplified)[/green]")
                except Exception as e:
                    console.print(f"[yellow]TRL DataCollator failed: {e}[/yellow]")
                    console.print("[yellow]Falling back to standard DataCollatorForSeq2Seq[/yellow]")
                    from transformers import DataCollatorForSeq2Seq
                    data_collator = DataCollatorForSeq2Seq(
                        tokenizer=self.tokenizer,
                        model=self.model,
                        label_pad_token_id=-100,
                        pad_to_multiple_of=8
                    )
        else:
            # Use standard data collator if TRL DataCollatorForCompletionOnlyLM not available
            console.print("[yellow]TRL DataCollatorForCompletionOnlyLM not available, using DataCollatorForSeq2Seq[/yellow]")
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8
            )
        
        # Initialize trainer with maximum version compatibility
        base_kwargs = {
            'model': self.model,
            'args': training_args,
            'train_dataset': self.train_dataset,
            'eval_dataset': self.eval_dataset,
        }
        
        # Optional parameters that may not be supported in all TRL versions
        optional_kwargs = {
            'tokenizer': self.tokenizer,
            'formatting_func': self.formatting_func,
            'data_collator': data_collator,
            'max_seq_length': self.config['training']['max_seq_length'],
        }
        
        # Try different combinations of parameters for maximum compatibility
        # Note: Removed 'packing' parameter as it's not supported in many TRL versions
        attempts = [
            # Try all parameters
            {**base_kwargs, **optional_kwargs},
            # Without tokenizer
            {**base_kwargs, **{k: v for k, v in optional_kwargs.items() if k != 'tokenizer'}},
            # Without tokenizer and max_seq_length
            {**base_kwargs, **{k: v for k, v in optional_kwargs.items() if k not in ['tokenizer', 'max_seq_length']}},
            # Without tokenizer and formatting_func (use default)
            {**base_kwargs, **{k: v for k, v in optional_kwargs.items() if k not in ['tokenizer', 'formatting_func']}},
            # Minimal parameters only
            base_kwargs
        ]
        
        trainer = None
        for i, kwargs in enumerate(attempts):
            try:
                trainer = SFTTrainer(**kwargs)
                console.print(f"[green]SFTTrainer initialized successfully (attempt {i+1})[/green]")
                break
            except TypeError as e:
                console.print(f"[yellow]SFTTrainer attempt {i+1} failed: {e}[/yellow]")
                continue
        
        if trainer is None:
            raise RuntimeError("Could not initialize SFTTrainer with any parameter combination")
        
        # Start training
        console.print("[bold green]Starting training...[/bold green]")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        console.print("[yellow]Saving final model...[/yellow]")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Save training stats
        self._save_training_stats(trainer)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        
    def _save_training_stats(self, trainer):
        """Save training statistics and metrics."""
        # Extract stats safely from log history
        final_train_loss = None
        final_eval_loss = None
        
        if trainer.state.log_history:
            # Look for the last entry with training loss
            for entry in reversed(trainer.state.log_history):
                if 'train_loss' in entry and final_train_loss is None:
                    final_train_loss = entry['train_loss']
                if 'eval_loss' in entry and final_eval_loss is None:
                    final_eval_loss = entry['eval_loss']
                if final_train_loss is not None and final_eval_loss is not None:
                    break
        
        stats = {
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "total_steps": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
        
        stats_file = Path(self.config['output_dir']) / "training_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        
        console.print(f"[green]Training stats saved to {stats_file}[/green]")
        
        # Display metrics safely
        if final_train_loss is not None:
            console.print(f"[cyan]Final training loss: {final_train_loss:.4f}[/cyan]")
        else:
            console.print("[yellow]No training loss recorded[/yellow]")
            
        if final_eval_loss is not None:
            console.print(f"[cyan]Final evaluation loss: {final_eval_loss:.4f}[/cyan]")
        else:
            console.print("[yellow]No evaluation loss recorded (evaluation may not have run)[/yellow]")
        
        if trainer.state.best_metric is not None:
            console.print(f"[cyan]Best metric: {trainer.state.best_metric:.4f}[/cyan]")


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Initialize trainer
    trainer = WPSFTTrainer(script_args.config)
    
    # Run training
    trainer.train(
        script_args.train_file,
        script_args.eval_file,
        script_args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()