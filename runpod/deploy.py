#!/usr/bin/env python3
"""
Automated deployment script for WordPress SLM on RunPod
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any
from pathlib import Path

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_base = "https://api.runpod.ai/v2"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def create_pod(self, name: str, gpu_type: str = "NVIDIA GeForce RTX 4090", 
                   disk_size: int = 50) -> Dict[str, Any]:
        """Create a GPU pod for development/training."""
        
        config = {
            "name": name,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "gpuTypeId": gpu_type,
            "cloudType": "SECURE",  # or "COMMUNITY" for cheaper
            "containerDiskInGb": disk_size,
            "volumeInGb": 0,
            "minMemoryInGb": 32,
            "minVcpuCount": 8,
            "startupScript": """#!/bin/bash
cd /workspace
git clone https://github.com/zd87pl/wp-slm-training.git wp-slm
cd wp-slm
pip install -r requirements.txt
pip install -e .
echo 'Setup complete! Run training with: make sft'
""",
            "env": [
                {"key": "PYTORCH_CUDA_ALLOC_CONF", "value": "max_split_size_mb:512"},
                {"key": "MODEL_PATH", "value": "/workspace/models/wp-slm"}
            ]
        }
        
        response = requests.post(
            f"{self.api_base}/pods",
            headers=self.headers,
            json=config
        )
        
        if response.status_code == 200:
            pod_data = response.json()
            print(f"✅ Pod created: {pod_data['id']}")
            print(f"   SSH: ssh root@{pod_data['desiredStatus']['ip']} -p {pod_data['desiredStatus']['port']}")
            return pod_data
        else:
            print(f"❌ Failed to create pod: {response.text}")
            return None
            
    def create_serverless_endpoint(self, name: str, docker_image: str,
                                 min_workers: int = 0, max_workers: int = 3) -> Dict[str, Any]:
        """Create a serverless endpoint for inference."""
        
        config = {
            "name": name,
            "dockerImage": docker_image,
            "gpuIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"],
            "minWorkers": min_workers,
            "maxWorkers": max_workers,
            "scalerType": "QUEUE_DEPTH",
            "scalerValue": 1,
            "env": [
                {"key": "MODEL_PATH", "value": "/workspace/models/wp-slm"},
                {"key": "MAX_BATCH_SIZE", "value": "8"}
            ],
            "networkVolumeId": None  # Add your volume ID here
        }
        
        response = requests.post(
            f"{self.api_base}/serverless",
            headers=self.headers,
            json=config
        )
        
        if response.status_code == 200:
            endpoint_data = response.json()
            print(f"✅ Serverless endpoint created: {endpoint_data['id']}")
            print(f"   Endpoint URL: {self.api_base}/{endpoint_data['id']}/runsync")
            return endpoint_data
        else:
            print(f"❌ Failed to create endpoint: {response.text}")
            return None
            
    def run_training_job(self, pod_id: str, config_file: str = "sft_qlora.yaml") -> bool:
        """Submit a training job to a pod."""
        
        command = f"""
cd /workspace/wp-slm
make data  # Prepare data
accelerate launch training/sft_train.py \\
    --config training/config/{config_file} \\
    --train_file data/sft/train.jsonl \\
    --eval_file data/sft/val.jsonl
"""
        
        # Execute command on pod
        response = requests.post(
            f"{self.api_base}/pods/{pod_id}/exec",
            headers=self.headers,
            json={"command": command}
        )
        
        if response.status_code == 200:
            print(f"✅ Training job submitted to pod {pod_id}")
            return True
        else:
            print(f"❌ Failed to submit training job: {response.text}")
            return False
            
    def test_inference_endpoint(self, endpoint_id: str, prompt: str) -> Dict[str, Any]:
        """Test an inference endpoint."""
        
        payload = {
            "input": {
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 512
            }
        }
        
        response = requests.post(
            f"{self.api_base}/{endpoint_id}/runsync",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference test successful!")
            print(f"   Response: {result.get('output', {}).get('response', 'No response')[:200]}...")
            return result
        else:
            print(f"❌ Inference test failed: {response.text}")
            return None
            
    def list_resources(self):
        """List all pods and endpoints."""
        
        # List pods
        pods_response = requests.get(f"{self.api_base}/pods", headers=self.headers)
        if pods_response.status_code == 200:
            pods = pods_response.json()
            print("\n📦 Pods:")
            for pod in pods:
                status = pod.get('desiredStatus', {}).get('status', 'Unknown')
                print(f"   - {pod['name']} ({pod['id']}) - Status: {status}")
                
        # List endpoints
        endpoints_response = requests.get(f"{self.api_base}/serverless", headers=self.headers)
        if endpoints_response.status_code == 200:
            endpoints = endpoints_response.json()
            print("\n🚀 Serverless Endpoints:")
            for endpoint in endpoints:
                print(f"   - {endpoint['name']} ({endpoint['id']})")


def main():
    parser = argparse.ArgumentParser(description="Deploy WordPress SLM on RunPod")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--action", choices=["create-pod", "create-endpoint", "train", "test", "list"],
                        required=True, help="Action to perform")
    parser.add_argument("--name", help="Name for pod/endpoint")
    parser.add_argument("--pod-id", help="Pod ID for training")
    parser.add_argument("--endpoint-id", help="Endpoint ID for testing")
    parser.add_argument("--prompt", default="How do I create a custom post type?",
                        help="Test prompt for inference")
    parser.add_argument("--docker-image", default="runpod/wp-slm:latest",
                        help="Docker image for serverless")
    
    args = parser.parse_args()
    
    deployer = RunPodDeployer(args.api_key)
    
    if args.action == "create-pod":
        if not args.name:
            print("❌ --name required for creating pod")
            sys.exit(1)
        deployer.create_pod(args.name)
        
    elif args.action == "create-endpoint":
        if not args.name:
            print("❌ --name required for creating endpoint")
            sys.exit(1)
        deployer.create_serverless_endpoint(args.name, args.docker_image)
        
    elif args.action == "train":
        if not args.pod_id:
            print("❌ --pod-id required for training")
            sys.exit(1)
        deployer.run_training_job(args.pod_id)
        
    elif args.action == "test":
        if not args.endpoint_id:
            print("❌ --endpoint-id required for testing")
            sys.exit(1)
        deployer.test_inference_endpoint(args.endpoint_id, args.prompt)
        
    elif args.action == "list":
        deployer.list_resources()


if __name__ == "__main__":
    main()