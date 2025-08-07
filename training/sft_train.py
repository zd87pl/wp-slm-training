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

# Disable TensorBoard integration at environment level to prevent callback registration
os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'TRUE'
os.environ['WANDB_DISABLED'] = 'true'
# Force disable TensorBoard by setting an invalid path
os.environ['TENSORBOARD_LOG_DIR'] = '/tmp/disabled_tensorboard'

# Set Hugging Face token if available
hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser
)
# Import callback handler to manually control callbacks
from transformers.integrations import is_tensorboard_available
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
            
        # Load tokenizer with HF token support
        hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        tokenizer_kwargs = {
            'trust_remote_code': True
        }
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
            
        console.print(f"[yellow]Loading tokenizer from {self.config['base_model']}...[/yellow]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            **tokenizer_kwargs
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            console.print("[yellow]Set pad_token to eos_token[/yellow]")
            
        # Load model with HF token support
        model_kwargs = {
            'quantization_config': bnb_config,
            'device_map': "auto",
            'trust_remote_code': True,
            'torch_dtype': torch.bfloat16 if self.config['training'].get('bf16', False) else torch.float16
        }
        if hf_token:
            model_kwargs['token'] = hf_token
            
        console.print(f"[yellow]Loading model from {self.config['base_model']}...[/yellow]")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            **model_kwargs
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
        
    def preprocess_datasets(self):
        """Preprocess datasets with proper tokenization and formatting."""
        console.print("[yellow]Preprocessing datasets...[/yellow]")
        
        def tokenize_function(examples):
            """Tokenize and format examples for training."""
            # Apply formatting function to get text
            if isinstance(examples, dict) and len(examples) > 0:
                # Handle batch processing
                first_key = next(iter(examples.keys()))
                batch_size = len(examples[first_key])
                
                texts = []
                for i in range(batch_size):
                    example = {key: examples[key][i] for key in examples.keys()}
                    text = self.formatting_func(example)
                    texts.append(text)
            else:
                # Handle single example
                texts = [self.formatting_func(examples)]
            
            # Tokenize the texts
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,  # We'll pad in the data collator
                max_length=self.config['training']['max_seq_length'],
                return_tensors=None  # Return as lists, not tensors
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        console.print("[yellow]Applying tokenization to train dataset...[/yellow]")
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        console.print("[yellow]Applying tokenization to eval dataset...[/yellow]")
        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
        
        console.print("[green]Dataset preprocessing completed[/green]")
        
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
            
        # Force disable all reporting to avoid dependency issues (TensorBoard, wandb, etc)
        # This overrides any config setting to ensure compatibility
        args['report_to'] = []
            
        return TrainingArguments(**args)
        
    def train(self, train_file: str, eval_file: str, resume_from_checkpoint: Optional[str] = None):
        """Run the training process."""
        try:
            # Set up model and tokenizer
            console.print("[bold blue]Step 1: Setting up model and tokenizer...[/bold blue]")
            self.setup_model_and_tokenizer()
            console.print("[green]✓ Model and tokenizer setup completed[/green]")
            
            # Set up PEFT if configured
            console.print("[bold blue]Step 2: Setting up PEFT (LoRA)...[/bold blue]")
            self.setup_peft()
            console.print("[green]✓ PEFT setup completed[/green]")
            
            # Load datasets
            console.print("[bold blue]Step 3: Loading datasets...[/bold blue]")
            self.load_datasets(train_file, eval_file)
            console.print("[green]✓ Dataset loading completed[/green]")
            
            # Preprocess datasets with proper tokenization
            console.print("[bold blue]Step 4: Preprocessing datasets...[/bold blue]")
            self.preprocess_datasets()
            console.print("[green]✓ Dataset preprocessing completed[/green]")
            
            # Set up training arguments
            console.print("[bold blue]Step 5: Setting up training arguments...[/bold blue]")
            training_args = self.setup_training_args()
            console.print("[green]✓ Training arguments setup completed[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Training setup failed during initialization: {e}[/red]")
            console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
        
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
        
        # Monkey patch to prevent TensorBoard callback registration
        original_is_tensorboard_available = None
        try:
            import transformers.integrations
            original_is_tensorboard_available = transformers.integrations.is_tensorboard_available
            # Force TensorBoard to appear unavailable
            transformers.integrations.is_tensorboard_available = lambda: False
        except Exception as e:
            console.print(f"[yellow]Warning: Could not disable TensorBoard integration: {e}[/yellow]")

        try:
            # Initialize trainer with maximum version compatibility
            console.print("[bold blue]Step 6: Initializing SFTTrainer...[/bold blue]")
            base_kwargs = {
                'model': self.model,
                'args': training_args,
                'train_dataset': self.train_dataset,
                'eval_dataset': self.eval_dataset,
            }
            
            # Since we've preprocessed the data, we don't need formatting_func, tokenizer, or max_seq_length
            # These would conflict with our preprocessed tokenized data
            optional_kwargs = {
                'data_collator': data_collator,
            }
            
            # Try different combinations of parameters for preprocessed data compatibility
            attempts = [
                # Try with data collator
                {**base_kwargs, **optional_kwargs},
                # Minimal parameters only (no data collator)
                base_kwargs
            ]
            
            trainer = None
            for i, kwargs in enumerate(attempts):
                try:
                    console.print(f"[yellow]Attempting SFTTrainer initialization (attempt {i+1})...[/yellow]")
                    trainer = SFTTrainer(**kwargs)
                    console.print(f"[green]✓ SFTTrainer initialized successfully (attempt {i+1})[/green]")
                    break
                except TypeError as e:
                    console.print(f"[yellow]SFTTrainer attempt {i+1} failed: {e}[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[red]SFTTrainer attempt {i+1} failed with unexpected error: {e}[/red]")
                    continue
            
            if trainer is None:
                raise RuntimeError("Could not initialize SFTTrainer with any parameter combination")
                
        except Exception as e:
            console.print(f"[red]❌ SFTTrainer initialization failed: {e}[/red]")
            console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
        
        # Restore original TensorBoard availability function
        try:
            if original_is_tensorboard_available is not None:
                transformers.integrations.is_tensorboard_available = original_is_tensorboard_available
        except Exception:
            pass  # Ignore restoration errors
        
        try:
            # Start training with enhanced monitoring
            console.print("[bold green]Step 7: Starting training...[/bold green]")
            console.print("[yellow]Training configuration:[/yellow]")
            console.print(f"[yellow]  - Epochs: {self.config['training']['num_train_epochs']}[/yellow]")
            console.print(f"[yellow]  - Train batch size: {self.config['training']['per_device_train_batch_size']}[/yellow]")
            console.print(f"[yellow]  - Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}[/yellow]")
            console.print(f"[yellow]  - Learning rate: {self.config['training']['learning_rate']}[/yellow]")
            console.print(f"[yellow]  - Max sequence length: {self.config['training']['max_seq_length']}[/yellow]")
            console.print(f"[yellow]  - Output directory: {self.config['output_dir']}[/yellow]")
            
            # Check GPU memory before training
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                console.print(f"[yellow]  - GPU Memory Total: {gpu_memory:.1f} GB[/yellow]")
                torch.cuda.empty_cache()  # Clear cache before training
                
            # Start the actual training with detailed error capture
            console.print("[yellow]Initiating training loop...[/yellow]")
            training_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            console.print("[green]✓ Training loop completed successfully[/green]")
            console.print(f"[cyan]Training result: {training_result}[/cyan]")
            
            # Verify training actually happened by checking global step
            if trainer.state.global_step == 0:
                raise RuntimeError("Training completed but no steps were executed (global_step=0)")
            else:
                console.print(f"[green]✓ Training executed {trainer.state.global_step} steps[/green]")
            
        except torch.cuda.OutOfMemoryError as e:
            console.print("[red]❌ GPU Out of Memory Error during training[/red]")
            console.print(f"[yellow]CUDA OOM: {e}[/yellow]")
            console.print("[yellow]Suggestions:[/yellow]")
            console.print("[yellow]  1. Reduce per_device_train_batch_size[/yellow]")
            console.print("[yellow]  2. Increase gradient_accumulation_steps[/yellow]")
            console.print("[yellow]  3. Reduce max_seq_length[/yellow]")
            console.print("[yellow]  4. Enable gradient_checkpointing[/yellow]")
            raise
        except RuntimeError as e:
            console.print(f"[red]❌ Runtime error during training: {e}[/red]")
            console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
            # Check for common runtime errors
            if "CUDA" in str(e):
                console.print("[yellow]This appears to be a CUDA-related error[/yellow]")
            elif "memory" in str(e).lower():
                console.print("[yellow]This appears to be a memory-related error[/yellow]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
        except Exception as e:
            console.print(f"[red]❌ Unexpected error during training: {e}[/red]")
            console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
            console.print(f"[yellow]Error details: {str(e)}[/yellow]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
        
        try:
            # Save final model with verification
            console.print("[bold blue]Step 8: Saving final model...[/bold blue]")
            
            # Ensure output directory exists
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model()
            
            # Verify model was actually saved - check for various model file patterns
            model_patterns = [
                "*.bin", "*.safetensors",  # Model weights
                "adapter_*.json", "adapter_config.json",  # LoRA adapter files
                "pytorch_model.bin", "model.safetensors",  # Standard model files
                "config.json",  # Model configuration
                "training_args.bin"  # Training arguments
            ]
            
            model_files = []
            for pattern in model_patterns:
                model_files.extend(list(output_dir.glob(pattern)))
            
            # Also check for any .bin or .safetensors files in subdirectories
            model_files.extend(list(output_dir.rglob("*.bin")))
            model_files.extend(list(output_dir.rglob("*.safetensors")))
            
            # Remove duplicates
            model_files = list(set(model_files))
            
            if model_files:
                console.print(f"[green]✓ Model saved ({len(model_files)} files)[/green]")
                for f in sorted(model_files)[:5]:  # Show first 5 files
                    console.print(f"[cyan]  - {f.name} ({f.stat().st_size} bytes)[/cyan]")
                if len(model_files) > 5:
                    console.print(f"[cyan]  ... and {len(model_files) - 5} more files[/cyan]")
            else:
                console.print(f"[yellow]⚠ Warning: No standard model files detected in {output_dir}[/yellow]")
                # List all files in the directory for debugging
                all_files = list(output_dir.iterdir())
                if all_files:
                    console.print(f"[yellow]Files found in output directory:[/yellow]")
                    for f in all_files[:10]:  # Show first 10 files
                        if f.is_file():
                            console.print(f"[yellow]  - {f.name} ({f.stat().st_size} bytes)[/yellow]")
                        else:
                            console.print(f"[yellow]  - {f.name}/ (directory)[/yellow]")
                    if len(all_files) > 10:
                        console.print(f"[yellow]  ... and {len(all_files) - 10} more items[/yellow]")
                else:
                    console.print(f"[red]❌ No files found in {output_dir}[/red]")
                    raise RuntimeError(f"Model saving failed - output directory is empty: {output_dir}")
                
        except Exception as e:
            console.print(f"[red]❌ Model saving failed: {e}[/red]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
            
        try:
            # Save tokenizer with verification
            console.print("[bold blue]Step 9: Saving tokenizer...[/bold blue]")
            self.tokenizer.save_pretrained(self.config['output_dir'])
            
            # Verify tokenizer was saved
            tokenizer_files = list(output_dir.glob("tokenizer*")) + list(output_dir.glob("special_tokens_map.json"))
            if tokenizer_files:
                console.print(f"[green]✓ Tokenizer saved ({len(tokenizer_files)} files)[/green]")
            else:
                console.print("[yellow]⚠ Warning: No tokenizer files detected after saving[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Tokenizer saving failed: {e}[/red]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            raise
            
        try:
            # Save training stats with verification
            console.print("[bold blue]Step 10: Saving training stats...[/bold blue]")
            self._save_training_stats(trainer)
            
            # Verify stats file was created
            stats_file = output_dir / "training_stats.json"
            if stats_file.exists():
                console.print("[green]✓ Training stats saved[/green]")
            else:
                console.print("[yellow]⚠ Warning: Training stats file not found[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Training stats saving failed: {e}[/red]")
            import traceback
            console.print(f"[yellow]Full traceback:\n{traceback.format_exc()}[/yellow]")
            # Don't raise here - stats saving is not critical
            
        console.print("[bold green]🎉 Training pipeline completed successfully![/bold green]")
        console.print(f"[cyan]Final model location: {self.config['output_dir']}[/cyan]")
        console.print(f"[cyan]Total training steps: {trainer.state.global_step}[/cyan]")
        
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