#!/usr/bin/env python3
"""
vLLM inference server for WordPress SLM with OpenAI-compatible API.
"""

import argparse
import json
import time
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

from prompt_templates import format_prompt, extract_response, PromptTemplates

# API Models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "wp-slm"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=1024, ge=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]


class CompletionRequest(BaseModel):
    model: str = "wp-slm"
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=1024, ge=1)
    stream: bool = False
    stop: Optional[List[str]] = None


# FastAPI app
app = FastAPI(title="WordPress SLM API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
llm_engine: Optional[AsyncLLMEngine] = None
model_name: str = "wp-slm"


def create_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt."""
    # For now, we'll use a simple approach
    # In production, you might want more sophisticated handling
    
    # Extract the last user message
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        raise ValueError("No user message found")
        
    last_user_message = user_messages[-1].content
    
    # Include context from previous messages if needed
    context = []
    for msg in messages[:-1]:
        if msg.role == "assistant":
            context.append(f"Previous response: {msg.content}")
            
    # Format the prompt
    prompt = format_prompt(last_user_message, include_system=True)
    
    return prompt


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Convert chat format to prompt
        prompt = create_chat_prompt(request.messages)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or ["USER:", "\n\n\n"],
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        
        # Generate response
        request_id = random_uuid()
        results = await llm_engine.generate(prompt, sampling_params, request_id)
        
        # Extract response
        generated_text = results.outputs[0].text
        response_text = extract_response(prompt + generated_text)
        
        # Format response
        response = ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": len(response_text.split()),  # Approximate
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    try:
        # Format prompt
        prompt = format_prompt(request.prompt, include_system=True)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or ["USER:", "\n\n\n"],
        )
        
        # Generate response
        request_id = random_uuid()
        results = await llm_engine.generate(prompt, sampling_params, request_id)
        
        # Extract response
        generated_text = results.outputs[0].text
        response_text = extract_response(prompt + generated_text)
        
        # Format response
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wp-slm"
        }]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_name}


def initialize_engine(args):
    """Initialize the vLLM engine."""
    global llm_engine, model_name
    
    model_name = args.model.split("/")[-1]  # Extract model name from path
    
    # Engine arguments
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Create async engine
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    print(f"Model loaded: {model_name}")
    print(f"Max sequence length: {args.max_model_len}")


def main():
    parser = argparse.ArgumentParser(description="vLLM server for WordPress SLM")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--tokenizer-mode", type=str, default="auto",
                        choices=["auto", "slow"],
                        help="Tokenizer mode")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for model")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "half", "float16", "bfloat16", "float32"],
                        help="Data type for model weights")
    
    # Engine arguments
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization (0-1)")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm", None],
                        help="Quantization method")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    
    args = parser.parse_args()
    
    # Initialize engine
    initialize_engine(args)
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()