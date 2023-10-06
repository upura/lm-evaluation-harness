#!/bin/bash
set -eu
MODEL_ARGS="pretrained=elyza/ELYZA-japanese-Llama-2-7b-fast,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.2-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEW_SHOTS="0,0,0,0,0,0,0,0"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --verbose \
    --output_path "models/elyza/ELYZA-japanese-Llama-2-7b-fast/result-0shot.json"
