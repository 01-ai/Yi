import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from PIL import Image

from llava.eval.mmmu_utils.data_utils import load_yaml, construct_prompt, save_json, CAT_SHORT2LONG, get_mmmu_hf_data
from llava.eval.mmmu_utils.eval_utils import evaluate, get_multi_choice_prediction, get_short_ans_prediction

from llava.model.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, key_info
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
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


def single_inference(image_processor, tokenizer, model, args, qs, image):
    
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    if image:
        image = image.convert('RGB')
        
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
        response = single_inference(image_processor, tokenizer, model, args, sample["final_input_prompt"], sample["image"])
        
        if sample['question_type'] == 'multiple-choice':
            pred_ans = get_multi_choice_prediction(response, sample['all_choices'], sample['index2ans'])
            if response == sample['gt_content']:
                pred_ans = sample['gt_content']
        else:  # short answer
            pred_ans = get_short_ans_prediction(response)
            
        cur_dict = {}
        
        cur_dict[sample['id']] = response
        
        out_samples.append(cur_dict)

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
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='')
    
    print(CAT_SHORT2LONG.keys())

    args = parser.parse_args()
    if args.categories[0] == 'ALL':
        args.categories = CAT_SHORT2LONG.keys()
    
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
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]
            
    print(args.config)

    from datasets import load_dataset

    # ###### test data
    # tot_data = load_dataset(args.data_path)['test']
    # # tot_data = load_dataset(args.data_path)['validation']
    # tot_data = get_mmmu_hf_data(tot_data)
    # print('begin process data')
    # samples = []
    # for sample in tot_data:
    #     # print(sample)
    #     try:
    #         sample = construct_prompt(sample, args.config)
    #     except:
    #         print('warning!!!!!!!!!!!!', category)
    #         print(sample)
    #         continue            
    #     samples.append(sample)
        
    # print('process test data done')
    # # run ex
    # out_samples = run_model(args, samples, model, tokenizer, image_processor)

    # output_data = {}
    # for item in out_samples:
    #     key = list(item.keys())[0]
    #     response = item[key]
    #     # if response not in ['A','B','C','D','E','F','G']:
    #     #     response = get_short_ans_prediction(response)
    #     output_data[key] = response

    # output_dir = os.path.join(args.results_dir, args.exp_name)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # save_output_path = os.path.join(output_dir, 'test_output.json')
    # with open(save_output_path, 'w') as f:
    #     json.dump(output_data, f, indent=4, ensure_ascii=False)


    ###### validation data
    # tot_data = load_dataset(args.data_path)['test']
    tot_data = load_dataset(args.data_path)['validation']
    tot_data = get_mmmu_hf_data(tot_data)
    print('begin process data')
    samples = []
    for sample in tot_data:
        # print(sample)
        try:
            sample = construct_prompt(sample, args.config)
        except:
            print('warning!!!!!!!!!!!!', category)
            print(sample)
            continue            
        samples.append(sample)
        
    print('process validation data done')
    # run ex
    out_samples = run_model(args, samples, model, tokenizer, image_processor)

    output_data = {}
    for item in out_samples:
        key = list(item.keys())[0]
        response = item[key]
        # if response not in ['A','B','C','D','E','F','G']:
        #     response = get_short_ans_prediction(response)
        output_data[key] = response

    output_dir = os.path.join(args.results_dir, args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_output_path = os.path.join(output_dir, 'validation_output.json')
    with open(save_output_path, 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    

if __name__ == '__main__':
    main()
