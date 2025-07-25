#!/usr/bin/env python3
"""
Generate candidate responses from the SFT model for preference data creation.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel
from tqdm import tqdm
from rich.console import Console
import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference.prompt_templates import format_prompt, extract_response

console = Console()

class CandidateGenerator:
    def __init__(self, model_path: str, use_4bit: bool = True):
        """Initialize generator with model."""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
        self._load_model(use_4bit)
        
    def _load_model(self, use_4bit: bool):
        """Load model and tokenizer."""
        # Check if this is a PEFT model
        peft_config_path = Path(self.model_path) / "adapter_config.json"
        is_peft = peft_config_path.exists()
        
        if is_peft:
            # Load PEFT config to get base model
            with open(peft_config_path, 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config.get('base_model_name_or_path')
            
            # Load base model
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            
            # Load PEFT model
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            console.print("[green]Loaded PEFT model[/green]")
        else:
            # Load regular model
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            console.print("[green]Loaded full model[/green]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set model to eval mode
        self.model.eval()
        
    def generate_candidates(self, prompts: List[str], 
                          num_return_sequences: int = 2,
                          temperature: float = 0.8,
                          top_p: float = 0.9,
                          max_new_tokens: int = 1024) -> List[List[str]]:
        """Generate candidate responses for prompts."""
        all_candidates = []
        
        for prompt in tqdm(prompts, desc="Generating candidates"):
            # Format prompt
            formatted_prompt = format_prompt(prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode candidates
            candidates = []
            for output in outputs:
                # Decode full sequence
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Extract just the response
                response = extract_response(full_text)
                candidates.append(response)
                
            all_candidates.append(candidates)
            
        return all_candidates


def main():
    parser = argparse.ArgumentParser(description="Generate candidate responses for preference data")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SFT model")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts JSONL file")
    parser.add_argument("--output", type=str, default="data/prefs/candidates.jsonl",
                        help="Output path for candidates")
    parser.add_argument("--num-return-sequences", type=int, default=2,
                        help="Number of candidates per prompt")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load prompts
    console.print(f"[cyan]Loading prompts from {args.prompts}...[/cyan]")
    prompts = []
    with open(args.prompts, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append(data['prompt'])
                
    console.print(f"[green]Loaded {len(prompts)} prompts[/green]")
    
    # Initialize generator
    generator = CandidateGenerator(args.model, use_4bit=not args.no_4bit)
    
    # Generate candidates
    all_candidates = generator.generate_candidates(
        prompts,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save candidates
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        for prompt, candidates in zip(prompts, all_candidates):
            result = {
                'prompt': prompt,
                'candidates': candidates
            }
            f.write(json.dumps(result) + '\n')
            
    console.print(f"[green]Saved candidates to {output_path}[/green]")
    
    # Print statistics
    total_candidates = sum(len(c) for c in all_candidates)
    console.print(f"[cyan]Generated {total_candidates} total candidates[/cyan]")
    console.print(f"[cyan]Average candidates per prompt: {total_candidates / len(prompts):.1f}[/cyan]")


if __name__ == "__main__":
    main()