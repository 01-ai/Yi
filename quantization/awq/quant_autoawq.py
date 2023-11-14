import argparse
import logging

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def run_quantization(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoAWQForCausalLM.from_pretrained(args.model)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
    }
    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(args.output_dir, safetensors=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AutoAWQ quantize")
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
