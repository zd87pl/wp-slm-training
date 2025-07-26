# WordPress SLM on Google Vertex AI + WP Engine Atlas Architecture

## Executive Summary

This document outlines the architecture for deploying your custom WordPress SLM (https://huggingface.co/0x7d0/wordpress-slm) on Google Vertex AI and integrating it with a Node.js application on WP Engine Atlas.

## Model Information
- **Model**: TinyLlama-1.1B + WordPress LoRA adapter
- **Location**: https://huggingface.co/0x7d0/wordpress-slm
- **Performance**: 98-99% improvement (Training Loss: 0.0140, Eval Loss: 0.0009)
- **Size**: ~1.1GB base model + 25MB LoRA adapter

## Architecture Overview

```
WP Engine Atlas (Node.js)                Google Vertex AI
├── Express.js API Server                ├── WordPress SLM Model
├── Vertex AI Client SDK                 ├── GPU Acceleration (T4/V100)
├── Redis Caching Layer                  ├── Auto-scaling Infrastructure
├── Authentication & Rate Limiting       ├── Model Monitoring & Logging
└── WordPress REST API Endpoints         └── 99.95% SLA Guarantee
                ↕ HTTPS Predictions
```

## Why Google Vertex AI?

### Enterprise Benefits
✅ **99.95% SLA** - Production-grade reliability  
✅ **Auto-scaling** - Scale to zero, scale up automatically  
✅ **GPU Acceleration** - T4/V100 GPUs for fast inference  
✅ **Built-in Monitoring** - Model performance tracking  
✅ **Cost Optimization** - Pay only for active predictions  
✅ **Security** - IAM, VPC, SOC 2/ISO compliance  

### Technical Advantages
✅ **Native HuggingFace Support** - Direct model deployment  
✅ **Model Versioning** - A/B testing and rollbacks  
✅ **Batch Predictions** - Efficient bulk processing  
✅ **Regional Deployment** - Low latency to Atlas  

## Implementation Plan

### Phase 1: Vertex AI Model Deployment

#### 1.1 Create Custom Prediction Container
```dockerfile
# Dockerfile.vertex
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13:latest

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy prediction code
COPY predictor.py .
COPY model_loader.py .

# Set up prediction service
EXPOSE 8080
CMD ["python", "predictor.py"]
```

#### 1.2 WordPress SLM Predictor
```python
# predictor.py
import torch
import json
import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class WordPressSLMPredictor:
    def __init__(self):
        self.model_id = "0x7d0/wordpress-slm"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the WordPress SLM model"""
        logging.info("Loading WordPress SLM model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base TinyLlama model
        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_id)
        self.model = self.model.to(self.device)
        
        logging.info("WordPress SLM loaded successfully!")
    
    def predict(self, prompt, max_tokens=300, temperature=0.7):
        """Generate WordPress expert response"""
        # Format for TinyLlama chat template
        formatted_prompt = f"<|system|>\nYou are a WordPress expert assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        
        return response

# Global predictor instance
predictor = WordPressSLMPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        instances = data.get('instances', [])
        
        predictions = []
        for instance in instances:
            prompt = instance.get('prompt', '')
            max_tokens = instance.get('max_tokens', 300)
            temperature = instance.get('temperature', 0.7)
            
            response = predictor.predict(prompt, max_tokens, temperature)
            predictions.append({'generated_text': response})
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

#### 1.3 Deploy to Vertex AI
```bash
# Build and push container
docker build -f Dockerfile.vertex -t gcr.io/YOUR_PROJECT/wp-slm-predictor:v1 .
docker push gcr.io/YOUR_PROJECT/wp-slm-predictor:v1

# Upload model to Vertex AI
gcloud ai models upload \
  --region=us-central1 \
  --display-name=wordpress-slm \
  --container-image-uri=gcr.io/YOUR_PROJECT/wp-slm-predictor:v1 \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080

# Create endpoint
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name=wp-slm-endpoint

# Deploy model to endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --traffic-split=0=100 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --min-replica-count=0 \
  --max-replica-count=3
```

### Phase 2: Atlas Node.js Integration

#### 2.1 Install Dependencies
```json
{
  "name": "atlas-wp-slm",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.2",
    "@google-cloud/aiplatform": "^3.20.0",
    "redis": "^4.6.0",
    "helmet": "^7.1.0",
    "express-rate-limit": "^7.1.0",
    "cors": "^2.8.5"
  }
}
```

#### 2.2 Vertex AI Client for Atlas
```javascript
// lib/vertex-ai-client.js
const { PredictionServiceClient } = require('@google-cloud/aiplatform');
const Redis = require('redis');

class VertexAIWordPressSLM {
  constructor(config) {
    this.projectId = config.projectId;
    this.location = config.location;
    this.endpointId = config.endpointId;
    
    // Initialize Vertex AI client
    this.client = new PredictionServiceClient({
      keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS
    });
    
    this.endpoint = `projects/${this.projectId}/locations/${this.location}/endpoints/${this.endpointId}`;
    
    // Initialize Redis cache
    this.redis = Redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });
    this.redis.connect();
    
    // Cache settings
    this.cacheTimeoutSeconds = 3600; // 1 hour
  }
  
  async generateWordPressResponse(prompt, options = {}) {
    const cacheKey = `wp-slm:${Buffer.from(prompt).toString('base64')}:${JSON.stringify(options)}`;
    
    try {
      // Check cache first
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        console.log('Cache hit for WordPress SLM request');
        return JSON.parse(cached);
      }
      
      // Prepare prediction request
      const instances = [{
        prompt: prompt,
        max_tokens: options.maxTokens || 300,
        temperature: options.temperature || 0.7
      }];
      
      const request = {
        endpoint: this.endpoint,
        instances: instances
      };
      
      // Call Vertex AI
      console.log('Calling Vertex AI WordPress SLM...');
      const [response] = await this.client.predict(request);
      
      if (!response.predictions || response.predictions.length === 0) {
        throw new Error('No predictions returned from Vertex AI');
      }
      
      const result = response.predictions[0];
      
      // Cache the result
      await this.redis.setEx(cacheKey, this.cacheTimeoutSeconds, JSON.stringify(result));
      
      return result;
      
    } catch (error) {
      console.error('Vertex AI WordPress SLM error:', error);
      
      // Fallback response
      return {
        generated_text: "I'm sorry, I'm temporarily unable to provide WordPress assistance. Please try again in a moment.",
        source: 'fallback'
      };
    }
  }
  
  async batchGenerate(prompts, options = {}) {
    const instances = prompts.map(prompt => ({
      prompt,
      max_tokens: options.maxTokens || 300,
      temperature: options.temperature || 0.7
    }));
    
    const request = {
      endpoint: this.endpoint,
      instances: instances
    };
    
    const [response] = await this.client.predict(request);
    return response.predictions;
  }
  
  async getHealthStatus() {
    try {
      // Simple health check with a basic prompt
      const result = await this.generateWordPressResponse("What is WordPress?", { maxTokens: 50 });
      return {
        status: 'healthy',
        vertex_ai: 'connected',
        cache: this.redis.isReady ? 'connected' : 'disconnected',
        last_response_length: result.generated_text?.length || 0
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }
}

module.exports = VertexAIWordPressSLM;
```

#### 2.3 Express.js Server for Atlas
```javascript
// server.js
const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const cors = require('cors');
const VertexAIWordPressSLM = require('./lib/vertex-ai-client');

const app = express();
const port = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['*']
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

app.use(express.json());

// Initialize WordPress SLM client
const wpSLM = new VertexAIWordPressSLM({
  projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
  location: process.env.VERTEX_AI_LOCATION || 'us-central1',
  endpointId: process.env.VERTEX_AI_ENDPOINT_ID
});

// WordPress assistance endpoint
app.post('/api/wp-assist', async (req, res) => {
  try {
    const { question, context } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }
    
    // Enhance prompt with context if provided
    let enhancedPrompt = question;
    if (context) {
      enhancedPrompt = `Context: ${context}\n\nQuestion: ${question}`;
    }
    
    const result = await wpSLM.generateWordPressResponse(enhancedPrompt, {
      maxTokens: 500,
      temperature: 0.7
    });
    
    res.json({
      answer: result.generated_text,
      cached: result.source !== 'vertex-ai',
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('WordPress assistance error:', error);
    res.status(500).json({ 
      error: 'WordPress assistance service unavailable',
      message: error.message 
    });
  }
});

// WordPress REST API compatible endpoints
app.get('/wp-json/slm/v1/help/:topic', async (req, res) => {
  try {
    const topic = req.params.topic;
    const prompt = `How do I ${topic.replace(/-/g, ' ')} in WordPress? Please provide a detailed explanation with code examples.`;
    
    const result = await wpSLM.generateWordPressResponse(prompt);
    
    res.json({
      topic: topic,
      guidance: result.generated_text,
      generated_at: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({ error: 'Service unavailable' });
  }
});

// Batch processing endpoint
app.post('/api/wp-batch', async (req, res) => {
  try {
    const { questions } = req.body;
    
    if (!Array.isArray(questions) || questions.length === 0) {
      return res.status(400).json({ error: 'Questions array is required' });
    }
    
    if (questions.length > 10) {
      return res.status(400).json({ error: 'Maximum 10 questions per batch' });
    }
    
    const results = await wpSLM.batchGenerate(questions);
    
    res.json({
      results: results.map((result, index) => ({
        question: questions[index],
        answer: result.generated_text
      })),
      processed_count: results.length
    });
    
  } catch (error) {
    res.status(500).json({ error: 'Batch processing failed' });
  }
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const health = await wpSLM.getHealthStatus();
    res.json(health);
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

// Start server
app.listen(port, () => {
  console.log(`WordPress SLM Atlas server running on port ${port}`);
  console.log(`Vertex AI Project: ${process.env.GOOGLE_CLOUD_PROJECT_ID}`);
  console.log(`Vertex AI Endpoint: ${process.env.VERTEX_AI_ENDPOINT_ID}`);
});
```

#### 2.4 Atlas Environment Configuration
```bash
# Atlas Environment Variables
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_ENDPOINT_ID=your-endpoint-id
GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
REDIS_URL=redis://your-redis-instance:6379
ALLOWED_ORIGINS=https://yourwordpresssite.com,https://yourdomain.com
```

### Phase 3: Cost Optimization & Monitoring

#### 3.1 Cost Analysis
```
Vertex AI Costs (Monthly - 1000 requests):
├── Prediction Requests: ~$20-30
├── Compute Time (T4 GPU): ~$15-25  
├── Storage & Network: ~$5-10
└── Total: ~$40-65/month

Benefits:
├── Enterprise SLA (99.95%)
├── Auto-scaling (pay only when used)
├── Built-in monitoring
└── Professional support
```

#### 3.2 Monitoring Setup
```javascript
// lib/monitoring.js
const { Monitoring } = require('@google-cloud/monitoring');

class WordPressSLMMonitoring {
  constructor(projectId) {
    this.client = new Monitoring.MetricServiceClient();
    this.projectId = projectId;
  }
  
  async recordPredictionMetrics(latency, success, cacheHit) {
    // Record custom metrics for WordPress SLM usage
    const metrics = [
      {
        name: 'wp_slm_prediction_latency',
        value: latency,
        labels: { success: success.toString() }
      },
      {
        name: 'wp_slm_cache_hit_rate',
        value: cacheHit ? 1 : 0
      }
    ];
    
    // Send metrics to Google Cloud Monitoring
    for (const metric of metrics) {
      await this.writeMetric(metric);
    }
  }
}
```

## Deployment Timeline

### Week 1: Vertex AI Setup
- [ ] Create Google Cloud Project
- [ ] Enable Vertex AI APIs  
- [ ] Build and deploy WordPress SLM container
- [ ] Create and test Vertex AI endpoint

### Week 2: Atlas Integration
- [ ] Set up service account authentication
- [ ] Deploy Node.js app with Vertex AI client
- [ ] Configure Redis caching
- [ ] Test end-to-end integration

### Week 3: WordPress Integration  
- [ ] Create WordPress REST API endpoints
- [ ] Implement authentication and rate limiting
- [ ] Add batch processing capabilities
- [ ] Performance testing

### Week 4: Production Readiness
- [ ] Set up monitoring and alerting
- [ ] Implement cost optimization
- [ ] Documentation and training
- [ ] Go-live preparation

## Security Considerations

### Authentication
- Service account with minimal required permissions
- API key rotation policies  
- Request signing and validation

### Network Security
- VPC peering between Atlas and Vertex AI (if needed)
- HTTPS-only communication
- Rate limiting and DDoS protection

### Data Privacy
- No sensitive data logging
- Request/response encryption
- GDPR compliance measures

## Success Metrics

1. **Performance**: <2s average response time
2. **Reliability**: 99.9% uptime 
3. **Cost**: <$50/month for moderate usage
4. **Quality**: Maintain model performance (0.0009 eval loss)
5. **Scalability**: Handle 10x traffic spikes automatically

## Conclusion

This architecture leverages Google Vertex AI's enterprise-grade ML infrastructure to serve your exceptional WordPress SLM (98-99% performance improvement) through a scalable Node.js application on WP Engine Atlas. 

The combination provides professional reliability, automatic scaling, built-in monitoring, and cost optimization while maintaining the high-quality WordPress expertise your model demonstrates.