from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def run_quantization(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        legacy=True,
        use_fast=False,
    )
    model = (
        AutoAWQForCausalLM.from_quantized(
            args.model,
            trust_remote_code=args.trust_remote_code,
            fuse_layers=True,
            batch_size=args.batch,
        )
        .cuda()
        .eval()
    )

    prompt = "count to 1000: 0 1 2 3"
    prompts = [prompt] * args.batch
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False).to(
        "cuda:0"
    )
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=4096,
    )
    generate_tokens = tokenizer.batch_decode(output_ids)
    print(generate_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AWQ quantized model")
    parser.add_argument("--model", type=str, help="The quantized name")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
    )
    parser.add_argument("--batch", type=int, default=4)

    args = parser.parse_args()
    run_quantization(args)
