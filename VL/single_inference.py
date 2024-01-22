import argparse
import torch
import os
from PIL import Image

from llava.model.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, key_info
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.mm_utils import expand2square


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", multimodal="IMAGE"):
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    image_processor = None
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def single_infer(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    key_info['model_path'] = model_path
    # import pdb;pdb.set_trace()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    image_file = args.image_file
    qs = args.question
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print('----------')
    print('question:',args.question)
    print('outputs:',outputs)
    print('----------')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="01-ai/Yi-VL-6B")
    parser.add_argument("--image-file", type=str, default="images/kobe.jpg")
    parser.add_argument("--question", type=str, default="Who is the man in the image?")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="mm_default")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    single_infer(args)