"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import re

import pdb

from PIL import Image

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'ee': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

# DATA LOADING
def walk_dir(data_dir, file_types=['.jpg', '.png', '.PNG', '.JPG', '.jpeg', '.JPEG']):
    path_list = []
    for dirpath, _, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def remove_image_tags(input_string):
    return re.sub(r"<image \d+>", "", input_string)

def count_image_tags(input_string):
    matches = re.findall(r"<image \d+>", input_string)
    return len(matches)

def get_mmmu_hf_data(tot_data):
    datas = []
    for data in tot_data:
        # process images
        q_imgs_cnt = count_image_tags(data['question'])
        o_imgs_cnt = 0
        # process question to remove image tags
        # question = re.sub("<img='(.*?)'>", "<image 1>", data['question'])
        question = data['question']
        img_cnt = 0
        img_cnt += count_image_tags(question)
        question = remove_image_tags(question)
        
        # process options to remove image tags
        replaced_options = []
        data['options'] = eval(data['options'])
        for option in data['options']:
            img_cnt += count_image_tags(option)
            o_imgs_cnt += count_image_tags(option)
            replaced_options.append(option)
        data['options'] = replaced_options
        
        # even other forms are wrong already, this img part should still be checked again.
        if q_imgs_cnt == 0:
            for option in data['options']:
                o_imgs_cnt += count_image_tags(option)

        # construct data
        if o_imgs_cnt > 1:  # multiple images in options, used for random selection
            datas.append(
                {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'], 'image': None, 'question_type': data['question_type']})
        else:
            datas.append(
                {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'], 'image': data['image_1'], 'question_type': data['question_type']})
    # print(f"Loading {category} data done. Total {len(datas)} samples out of {len(samples)}, ratio {len(datas)/len(samples)}.")
    return datas


# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)


def save_jsonl(filename, data):
    """
    Save a dictionary of data to a JSON Lines file with the filename as key and caption as value.

    Args:
        filename (str): The path to the file where the data should be saved.
        data (dict): The dictionary containing the data to save where key is the image path and value is the caption.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for img_path, caption in data.items():
            # Extract the base filename without the extension
            base_filename = os.path.basename(img_path)
            # Create a JSON object with the filename as the key and caption as the value
            json_record = json.dumps({base_filename: caption}, ensure_ascii=False)
            # Write the JSON object to the file, one per line
            f.write(json_record + '\n')

def save_args(args, path_dir):
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


# DATA PROCESSING
def construct_prompt(sample, config):
    question = sample['question']
    options = sample['options']
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"{start_chr}. {option}\n"
            # example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:  # short answer, other forms alredy filtered out in prior steps
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict

