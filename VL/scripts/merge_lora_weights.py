import argparse
from llava.mm_utils import load_pretrained_model


def merge_lora(args):
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path,
                                                                           lora_path=args.lora_path,
                                                                           device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
