"""Quick util to test generation on a model."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="float16")
parser.add_argument("--use-fast", action="store_true")
args = parser.parse_args()


prompts = [
    "And we're under way! The first pitch",
    "The Statue of Liberty was a gift from",
    "My name is Arther Lee.",
    "def sum_even_numbers(numbers: list",
    "i purchased a baseball bat from the store for $10.00!",
    "Our solar system is comprised of ",
    "today sort of feels like one of those",
    "On Christmas morning I packed my belongings into a bag and headed out"
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=args.use_fast,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype),
    )
    model = model.to(args.device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
    # if not "token_type_ids" in inspect.signature(model.generate).parameters:
    inputs.pop("token_type_ids", None)
    inputs.pop("attention_mask", None)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        # top_p=0.95,
        temperature=0.75,
        max_new_tokens=128,
    )
    # Write results to file with name of model
    with open(f"generation-test-{args.model.replace('/', '_')}.txt", "w+") as f:
        for prompt, output in zip(prompts, outputs):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Output: {tokenizer.decode(output, skip_special_tokens=True)}\n\n")


if __name__ == "__main__":
    main()
