#!/usr/bin/env python3
"""
Setup script for WordPress SLM.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Core dependencies (subset of conda environment)
install_requires = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.0",
    "peft>=0.7.0",
    "trl>=0.7.0",
    "datasets>=2.14.0",
    "sentencepiece>=0.1.99",
    "tyro>=0.6.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "beautifulsoup4>=4.12.0",
    "markdownify>=0.11.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
]

# Optional dependencies
extras_require = {
    "inference": [
        "vllm>=0.2.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "sse-starlette>=1.6.0",
    ],
    "eval": [
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "docker>=6.0.0",
    ],
    "scraping": [
        "readability-lxml>=0.8.1",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.990",
    ],
}

# All extras
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="wp-slm",
    version="1.0.0",
    author="WP-SLM Team",
    description="WordPress-Specialized Small Language Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/wp-slm",
    packages=find_packages(exclude=["tests", "tests.*", "wp-plugin"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            # Data pipeline
            "wp-slm-scrape=scripts.scrape_wp_docs:main",
            "wp-slm-parse=scripts.parse_wp_docs:main",
            "wp-slm-build-sft=scripts.build_sft_pairs:main",
            "wp-slm-split-data=scripts.split_dataset:main",
            "wp-slm-validate=scripts.validate_data:main",
            
            # Training
            "wp-slm-train-sft=training.sft_train:main",
            "wp-slm-train-dpo=training.dpo_train:main",
            
            # Preference data
            "wp-slm-gen-candidates=scripts.gen_candidates:main",
            "wp-slm-gen-prefs=scripts.gen_preferences:main",
            
            # Evaluation
            "wp-slm-eval=eval.run_eval:main",
            
            # Inference
            "wp-slm-serve=inference.serve_vllm:main",
            
            # Tests
            "wp-slm-test=tests.run_tests:run_all_tests",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="wordpress, language-model, llm, fine-tuning, nlp",
)