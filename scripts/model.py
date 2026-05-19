# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

# predefined models
model_fullnames = {
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "opt-2.7b": "facebook/opt-2.7b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "mgpt": "sberbank-ai/mGPT",
    "pubmedgpt": "stanford-crfm/pubmedgpt",
    "mt5-xl": "google/mt5-xl",
    "llama-13b": "huggyllama/llama-13b",
    "llama2-13b": "TheBloke/Llama-2-13B-fp16",
    "bloom-7b1": "bigscience/bloom-7b1",
    "opt-13b": "facebook/opt-13b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
}
float16_models = [
    "gpt-neo-2.7B",
    "gpt-j-6B",
    "gpt-neox-20b",
    "llama-13b",
    "llama2-13b",
    "bloom-7b1",
    "opt-13b",
    "falcon-7b",
    "falcon-7b-instruct",
]


def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name


def load_model(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f"Loading model {model_fullname}...")

    model_kwargs = {}

    if model_name in float16_models:
        model_kwargs["torch_dtype"] = torch.float16

    if "gpt-j" in model_name:
        model_kwargs["revision"] = "float16"

    model = AutoModelForCausalLM.from_pretrained(
        model_fullname,
        cache_dir=cache_dir,
        device_map="auto",  # safe default for HPC
        trust_remote_code=True,
        **model_kwargs,
    )

    model.eval()
    return model


def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)

    optional_tok_kwargs = {
        "padding_side": "right",
    }

    if "facebook/opt-" in model_fullname:
        optional_tok_kwargs["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_fullname, cache_dir=cache_dir, **optional_tok_kwargs
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bloom-7b1")
    parser.add_argument("--cache_dir", type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, args.cache_dir)
    load_model(args.model_name, args.cache_dir)
