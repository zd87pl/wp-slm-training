ENV?=wp-slm
PYTHON?=python
ACCELERATE?=accelerate

.PHONY: help env data sft dpo ppo eval serve plugin clean test docker-wp

help:
	@echo "WordPress SLM Development Commands:"
	@echo "  make env        - Create conda environment"
	@echo "  make data       - Run data pipeline (scrape, parse, build pairs)"
	@echo "  make sft        - Run supervised fine-tuning"
	@echo "  make dpo        - Run DPO alignment training"
	@echo "  make ppo        - Run PPO training (optional)"
	@echo "  make eval       - Run evaluation suite"
	@echo "  make serve      - Start inference server"
	@echo "  make plugin     - Package WordPress plugin"
	@echo "  make docker-wp  - Start WordPress test environment"
	@echo "  make clean      - Clean generated files"
	@echo "  make test       - Run all tests"

env:
	conda env create -f environment.yml || true
	@echo "Activate with: conda activate $(ENV)"

data:
	$(PYTHON) scripts/scrape_wp_docs.py
	$(PYTHON) scripts/parse_wp_docs.py
	$(PYTHON) scripts/build_sft_pairs.py
	$(PYTHON) scripts/split_dataset.py

sft:
	$(ACCELERATE) launch training/sft_train.py \
		--config training/config/sft_qlora.yaml \
		--train_file data/sft/train.jsonl \
		--eval_file data/sft/val.jsonl

dpo:
	$(PYTHON) scripts/gen_candidates.py \
		--model outputs/wp-sft-qlora \
		--prompts data/prefs/prompts.jsonl
	$(PYTHON) scripts/gen_preferences.py
	$(ACCELERATE) launch training/dpo_train.py \
		--config training/config/dpo.yaml

ppo:
	$(ACCELERATE) launch training/rlaif_ppo.py \
		--config training/config/ppo.yaml

eval:
	$(PYTHON) eval/run_eval.py \
		--model outputs/wp-dpo \
		--test_file data/eval/test.jsonl

serve:
	$(PYTHON) inference/serve_vllm.py \
		--model outputs/wp-slm-merged \
		--tensor-parallel-size 1 \
		--max-model-len 4096

plugin:
	@echo "Building WordPress plugin..."
	@cd wp-plugin && zip -r ../wp-slm-assistant.zip . -x "*.DS_Store"
	@echo "Plugin packaged as wp-slm-assistant.zip"

docker-wp:
	@echo "Starting WordPress test environment..."
	@docker-compose up -d
	@echo "WordPress available at http://localhost:8080"
	@echo "Admin: http://localhost:8080/wp-admin"

clean:
	rm -rf outputs/
	rm -rf data/processed/
	rm -rf data/sft/*.jsonl
	rm -rf data/prefs/*.jsonl
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

test:
	$(PYTHON) tests/run_tests.py
	$(PYTHON) scripts/validate_data.py --data-dir data