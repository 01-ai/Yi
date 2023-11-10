from transformers import AutoModelForCausalLM, AutoTokenizer


def run_quantization(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        legacy=True,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0",
        trust_remote_code=args.trust_remote_code,
        use_safetensors=True,
    ).eval()

    prompt = "count to 1000: 0 1 2 3"
    prompts = [prompt] * 4
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False).to(
        model.device
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

    parser = argparse.ArgumentParser(description="Run GPTQ quantized model")
    parser.add_argument("--model", type=str, help="The quantized name")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
    )

    args = parser.parse_args()
    run_quantization(args)
