#!/bin/bash
set -eu
PROJECT_DIR="/fsx/proj-jp-stablegpt"
MODEL_ARGS="pretrained=${PROJECT_DIR}/hf_model/stablelm-ja-base-alpha-7b,tokenizer=${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False,trust_remote_code=True"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.2-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
NUM_FEW_SHOTS="0,0,0,0,0,0,0,0"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --verbose \
    --output_path "models/stabilityai/stabilityai-japanese-stablelm-base-alpha-7b/result-0shot-prompt03.json"
