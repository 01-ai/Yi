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
        "--tokenizer",
        type=str,
        default="",
        help="tokenizer path locally or name on huggingface",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
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
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|endoftext|>",
        help="End of sentence token",
    )
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )
    streamer = TextStreamer(tokenizer) if args.streaming else None
    outputs = model.generate(
        inputs.input_ids.cuda(),
        max_new_tokens=args.max_tokens,
        streamer=streamer,
        eos_token_id=tokenizer.convert_tokens_to_ids(args.eos_token),
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
        temperature=0.7,
        top_k=40,
        top_p=0.8,
    )
    if streamer is None:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    args = parse_inputs()
    main(args)
