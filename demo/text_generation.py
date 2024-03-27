import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_inputs():
    parser = argparse.ArgumentParser(description="Yi-6B text generation demo")
    parser.add_argument(
        "--model",
        type=str,
        default="01-ai/Yi-6B",
        help="pretrained model path locally or name on huggingface",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max number of tokens to generate",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="whether to enable streaming text generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Let me tell you an interesting story about cat Tom and mouse Jerry,",
        help="The prompt to start with",
    )
    parser.add_argument("--cpu", action="store_true", help="Run demo with CPU only")
    args = parser.parse_args()
    return args


def main(args):
    print(args)

    if args.cpu:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device_map, torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextStreamer(tokenizer) if args.streaming else None
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        streamer=streamer,
        # do_sample=True,
        # repetition_penalty=1.3,
        # no_repeat_ngram_size=5,
        # temperature=0.7,
        # top_k=40,
        # top_p=0.8,
    )

    if streamer is None:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    args = parse_inputs()
    main(args)
