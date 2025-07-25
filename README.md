# WordPress-Specialized Small Language Model (WP-SLM)

A fine-tuned language model specialized for WordPress development, optimized to run on single GPU workstations.

## Overview

WP-SLM is a domain-specific language model fine-tuned on WordPress documentation, APIs, and best practices. It provides:
- WordPress development assistance
- Code generation for themes and plugins
- REST API integration examples
- Security best practices
- Troubleshooting guidance

## Hardware Requirements

- Single workstation with RTX 4090 (24 GB VRAM)
- Intel i9 processor
- 128 GB system RAM
- Fast NVMe SSD

## Base Model

- Primary: Llama-2-7B (for faster iteration)
- Alternative: Llama-3-8B Instruct

## Quick Start

1. Set up environment:
   ```bash
   conda env create -f environment.yml
   conda activate wp-slm
   ```

2. Prepare data:
   ```bash
   make data
   ```

3. Train model:
   ```bash
   make sft  # Supervised fine-tuning
   make dpo  # Preference optimization
   ```

4. Deploy:
   ```bash
   make serve
   ```

## Project Structure

```
wp-slm/
├── data/                 # Training data pipeline
├── scripts/              # Data processing scripts
├── training/             # Training configurations
├── inference/            # Model serving
├── wp-plugin/            # WordPress integration
├── eval/                 # Evaluation framework
└── Makefile              # Automation commands
```

## Training Pipeline

1. **Data Acquisition**: Scrape WordPress docs, WP-CLI docs, community forums
2. **SFT**: QLoRA fine-tuning with instruction pairs
3. **DPO**: Direct preference optimization for alignment
4. **Evaluation**: Automated testing against WP environments

## WordPress Plugin

The included plugin provides:
- Admin dashboard AI assistant
- Code suggestion blocks
- REST API call generator

## License

GPL-2.0+ (compatible with WordPress ecosystem)