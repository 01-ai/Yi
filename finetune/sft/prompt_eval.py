import argparse
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--language", type=str, default="Chinese", choices=["English", "Chinese"]
    )
    parser.add_argument("--eos", type=str, default="<|endoftext|>")

    args = parser.parse_args()

    return args


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=200,
    eos="<|endoftext|>",
):
    stop_token_id = tokenizer.convert_tokens_to_ids(eos)
    # by default, stop_token_id = tokenizer.eos_token_id

    generate_ids = model.generate(
        inputs.input_ids, max_new_tokens=max_new_tokens, eos_token_id=stop_token_id
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("==========Baseline: Greedy=========")
        r_base = generate(
            model_baseline,
            tokenizer,
            inputs,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            eos=args.eos,
        )
        print_utils(r_base)
        print("==========finetune: Greedy=========")
        r_finetune_g = generate(
            model_fintuned,
            tokenizer,
            inputs,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            eos=args.eos,
        )
        print_utils(r_finetune_g)
        print("====================prompt end=============================")
        print()
        print()


def main():
    args = parse_args()

    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(
        args.model_name_or_path_baseline, fast_tokenizer=False
    )

    model_baseline = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path_baseline,
        tokenizer,
        None,
        eval_mode=True,
    )
    model_fintuned = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path_finetune,
        tokenizer,
        None,
        eval_mode=True,
    )

    if args.language == "English":
        prompts = [
            "Human: Hello. Assistant:",
            "Human: Please explain Large Language Model. Assistant:",
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 你好。 Assistant:",
            "Human: 请介绍一下大语言模型? Assistant:",
        ]
    else:
        # TODO:
        prompts = []

    prompt_eval(args, model_baseline, model_fintuned, tokenizer, device, prompts)


if __name__ == "__main__":
    main()
