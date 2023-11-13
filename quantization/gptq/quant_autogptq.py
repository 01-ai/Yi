import argparse
import logging

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def run_quantization(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        legacy=True,
        use_fast=False,
    )

    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_config.max_position_embeddings = 4096  # to avoid OOM
    # Quantize
    quantization_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        dataset="wikitext2",
        tokenizer=tokenizer,
        disable_exllama=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
    )

    # Save quantized model
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="GPT-Q quantize")
    parser.add_argument(
        "--model",
        type=str,
        default="01-ai/Yi-6b",
        help="Pretrained model path locally or name on huggingface",
    )
    parser.add_argument("--output_dir", type=str, help="Output base folder")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
    )
    parser.add_argument("--bits", type=int, default=4, help="Quantize bit(s)")
    parser.add_argument(
        "--group_size", type=int, default=128, help="Quantize group size(s)"
    )

    args = parser.parse_args()
    run_quantization(args)
