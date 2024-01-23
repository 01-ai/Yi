import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from PIL import Image

from llava.eval.cmmmu_utils.data_utils import load_yaml, get_cmmmu_data, construct_mmodel_prompt, save_json, CAT_SHORT2LONG
from llava.eval.cmmmu_utils.eval_utils import evaluate, get_multi_choice_prediction, get_short_ans_prediction

from llava.model.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, key_info
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.mm_utils import expand2square, load_pretrained_model

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_dict_name(my_dict):
    for name, value in globals().items():
        if value is my_dict:
            return name


def single_inference(image_processor, tokenizer, model, args, qs, img_path):
    
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    if img_path:
        image = Image.open(img_path)
        image=image.convert('RGB')
        if getattr(model.config, "image_aspect_ratio", None) == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images_t = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
    else:
        crop_size = image_processor.crop_size
        images_t = torch.zeros(1, 3, crop_size['height'], crop_size['width']).to(dtype=torch.bfloat16).cuda()

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_t,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=64,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    outputs = outputs.split(stop_str)[0]
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


def run_model(args, samples, model, tokenizer, image_processor):
    out_samples = []
    for sample in tqdm(samples):
        # try:
        response = single_inference(image_processor, tokenizer, model, args, sample["final_input_prompt"], sample["img_path"])
        if sample['question_type'] == 'multiple-choice':
            pred_ans = get_multi_choice_prediction(response, sample['all_choices'], sample['index2ans'])
        else:  # short answer
            pred_ans = get_short_ans_prediction(response)
        out_samples.append({
            'No': sample['No'],
            'response': response,
            'pred_ans': pred_ans
        })

    return out_samples


def main():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='',
                        help='The name of the experiment')
    parser.add_argument('--config_path', type=str, default="")
    parser.add_argument('--data_path', type=str, default="", help="The path to data root directory.")
    parser.add_argument('--categories', nargs='+',
                        help=f'The name of the mmmu sub-category. Availble: {CAT_SHORT2LONG.keys()}')
    parser.add_argument('--model_type', type=str, default="pretrain_flant5xl")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument("--conv_mode", type=str, default="mm_default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='')
    
    print(CAT_SHORT2LONG.keys())

    args = parser.parse_args()    
    # # load model
    print('Model Loading...')
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    key_info['model_path'] = model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)

    print(f"----- Start to process dataset: mmmu -----\n")

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list and len(value) == 1:
            # assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]
            
    print(args.config)

    for category in os.listdir(args.data_path):
        if os.path.isdir(os.path.join(args.data_path, category)):
            print("#" * 50, f"Running {category}", "#" * 50)
        else:
            continue
        
        sub_dataset = get_cmmmu_data(args.data_path, category)
        # preprare data
        samples = []
        for sample in sub_dataset:
            sample = construct_mmodel_prompt(sample, args.config, zeroshot=True)

            samples.append(sample)

        # run ex
        out_samples = run_model(args, samples, model, tokenizer, image_processor)


        # save resutls
        # output dir
        output_dir = os.path.join(args.results_dir, args.exp_name, category)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_output_path = os.path.join(output_dir, 'output.json')

        save_json(save_output_path, out_samples)
        

    

if __name__ == '__main__':
    main()
