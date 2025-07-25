#!/usr/bin/env python3
"""
RunPod Handler for WordPress SLM
Supports both serverless inference and training jobs
"""

import os
import sys
import json
import torch
import runpod
from typing import Dict, Any

# Add project to path
sys.path.append('/workspace/wp-slm')

# Lazy imports to reduce cold start time
model = None
tokenizer = None


def load_model():
    """Load model on first request."""
    global model, tokenizer
    
    if model is None:
        print("Loading WordPress SLM model...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        model_path = os.environ.get('MODEL_PATH', '/workspace/models/wp-slm')
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}. Please mount a model or train one first.")
        
        # Load with quantization for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if it's a PEFT model
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(peft_config_path):
            # Load PEFT model
            with open(peft_config_path, 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config['base_model_name_or_path']
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        
        model.eval()
        print("Model loaded successfully!")


def generate_response(prompt: str, **kwargs) -> str:
    """Generate response from the model."""
    load_model()
    
    # Format prompt
    formatted_prompt = f"""You are a helpful WordPress expert assistant. Answer the following question accurately and provide code examples when relevant.

USER: {prompt}
ASSISTANT: """
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "ASSISTANT:" in generated:
        response = generated.split("ASSISTANT:")[-1].strip()
    else:
        response = generated[len(formatted_prompt):].strip()
    
    return response


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    
    Supports two modes:
    1. Inference: {"input": {"prompt": "...", "temperature": 0.7}}
    2. Training: {"input": {"mode": "train", "config": "sft_qlora.yaml"}}
    """
    job_input = job.get('input', {})
    
    # Check mode
    mode = job_input.get('mode', 'inference')
    
    if mode == 'inference':
        # Inference mode
        prompt = job_input.get('prompt')
        if not prompt:
            return {"error": "No prompt provided"}
        
        try:
            response = generate_response(
                prompt,
                temperature=job_input.get('temperature', 0.7),
                top_p=job_input.get('top_p', 0.9),
                max_new_tokens=job_input.get('max_new_tokens', 512)
            )
            
            return {
                "response": response,
                "prompt": prompt,
                "model": os.environ.get('MODEL_PATH', 'wp-slm')
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    elif mode == 'train':
        # Training mode
        config = job_input.get('config', 'sft_qlora.yaml')
        
        try:
            import subprocess
            
            # Run training script
            cmd = [
                "accelerate", "launch",
                "/workspace/wp-slm/training/sft_train.py",
                "--config", f"/workspace/wp-slm/training/config/{config}",
                "--train_file", "/workspace/data/sft/train.jsonl",
                "--eval_file", "/workspace/data/sft/val.jsonl"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else None,
                "return_code": result.returncode
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    else:
        return {"error": f"Unknown mode: {mode}"}


# RunPod serverless entry point
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})