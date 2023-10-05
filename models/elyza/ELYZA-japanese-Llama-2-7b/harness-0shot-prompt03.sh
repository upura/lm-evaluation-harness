#!/bin/bash
set -eu
MODEL_ARGS="pretrained=elyza/ELYZA-japanese-Llama-2-7b,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.2-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
NUM_FEW_SHOTS="0,0,0,0,0,0,0,0"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --verbose \
    --output_path "models/elyza/ELYZA-japanese-Llama-2-7b/result-prompt03.json"
