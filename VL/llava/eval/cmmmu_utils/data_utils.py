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
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}



# DOMAIN_CAT2SUB_CAT = {
#   'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
#   'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
#   'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
#   'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
#   'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
#   'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
# }


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
    matches = re.findall('<img="(.*?)">', text)
    return matches

def get_cmmmu_data(data_root_path, category):
    fpath = os.path.join(data_root_path, category)
    if not os.path.exists(fpath):
        print(f"Category {category} does not exist. Please check.")
        exit()

    samples = []
    datas = []
    jsonls = walk_dir(fpath, ['.jsonl'])
    for jsonl in jsonls:
        with open(jsonl) as fp:
            for line in fp:
                data = json.loads(line)
                samples.append(data)
    for data in samples:
        # process images
        # q_imgs_paths = parse_img_path(data['question'])
        type = data['type']
        o_imgs_paths = []
        q_imgs_paths = []
        img_list = data['img_list']
        for img in img_list:
            if img.startswith('q_'):
                q_imgs_paths.append(img)
            elif img.startswith('a_'):
                o_imgs_paths.append(img)
        # process question to remove image tags
        # question = re.sub("<img='(.*?)'>", "<image 1>", data['question'])
        question = data['question']
        img_cnt = 0
        for img_path in q_imgs_paths:
            img_cnt += 1
            question = question.replace(f'<img="{img_path}">', f"<图片 {img_cnt}>")
        # process options to remove image tags
        replaced_options = []
        if type == '选择':
            options = {data['option1']:'', data['option2']:'', data['option3']:'', data['option4']:''}

            # 构造options字典
            if len(o_imgs_paths)>1:
                for file_path in o_imgs_paths:
                    # 提取文件名中的编号
                    file_num = file_path.split('_')[-1].split('.')[0]
                    # 将编号映射到相应的选项，并将文件名添加到对应的选项
                    options[data['option'+file_num[-1]]] = file_path
                    

            # 构造最后的option（包含图片的）
            for k,v in options.items():
                if v != '':
                    img_cnt += 1
                    if f'<img="{v}">' in k:
                        replaced_options.append(k.replace(f'<img="{v}">', f"<图片 {img_cnt}>"))
                    else:
                        replaced_options.append(k + f"<图片 {img_cnt}>")
                else:
                    replaced_options.append(k)

        elif type == '判断':
            replaced_options = ['对', '错']
        data['options'] = replaced_options

        if type == '填空':
            check_data_flag = False
        else:
            if (type not in ['判断', '选择', '填空']):
                check_data_flag = True
            else:
                check_data_flag = False

        # even other forms are wrong already, this img part should still be checked again.
        if len(q_imgs_paths) == 0:
            for option in data['options']:
                o_imgs_paths.extend(parse_img_path(option))
            if len(o_imgs_paths) > 1:
                check_data_flag = False
            else:
                check_data_flag = True
                print("Can not found img in question or options")

        if check_data_flag:
            print(category)
            print(data)
            continue

        # construct data
        if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
            datas.append(
                {'No': data['id'],
                 'question': question,
                 'options': data['options'],
                 'answer': data['answer'],
                 'img_path': None,
                 'question_type': data['type'],
                #  'img_type': data['img_type'],
                 'topic_difficulty': data['difficulty_level']
                 })
        else:
            img_path = q_imgs_paths[0]
            img_path = os.path.join(fpath, img_path)
            if not os.path.exists(img_path):
                print(category)
                print("Image Error, does not exist")
                print(data)
                continue
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(e)
                print(category)
                print("Image Error, broken")
                print(data)
                continue
            datas.append(
                {'No': data['id'],
                 'question': question,
                 'options': data['options'],
                 'answer': data['answer'],
                 'img_path': img_path,
                 'question_type': data['type'],
                #  'img_type': data['img_type'],
                 'topic_difficulty': data['difficulty_level']})
    print(f"Loading {category} data done. Total {len(datas)} samples out of {len(samples)}, ratio {len(datas)/len(samples)}.")
    return datas


# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)


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

def remove_image_tags(input_string):
    return re.sub(r"<图片 \d+>", "", input_string)

# DATA PROCESSING
def construct_mmodel_prompt(sample, config, zeroshot):
    # print(sample)
    question = sample['question']
    options = sample['options']
    prediction_range = []
    index2ans = {}

    task_instructions = config['task_instructions']
    
    if sample['question_type'] == '选择':
        formatted_options = ""
        start_chr = 'A'
        for option in options:
            prediction_range.append(start_chr)
            formatted_options += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        
        current_example_template = config['multi_choice_example_format']
        current_example = current_example_template.format(question, formatted_options)
        final_input_prompt = task_instructions[0] + "\n\n" + current_example
        # Ensure gt_content is the actual text of the correct answer
        correct_answer_text = [options[ord(opt.upper()) - ord('A')] for opt in sample['answer']]
    
    elif sample['question_type'] == '判断':
        current_example_template = config['T/F_example_format']
        current_example = current_example_template.format(question)
        final_input_prompt = task_instructions[1] + "\n\n" + current_example
        correct_answer_text = sample['answer']
        
    else:  # For short answer questions.
        current_example_template = config['short_ans_example_format']
        current_example = current_example_template.format(question)
        final_input_prompt = task_instructions[2] + "\n\n" + current_example
        # For short answer, gt_content is the answer provided in the sample
        correct_answer_text = sample['answer']

    final_input_prompt = remove_image_tags(final_input_prompt)
    res_dict = {
        'final_input_prompt': final_input_prompt,
        'empty_prompt': current_example,
        'index2ans': index2ans,
        'all_choices': prediction_range,
        'gt_content': correct_answer_text
    }

    if sample['question_type'] == '选择':
        res_dict['correct_choice'] = sample['answer']
    
    res_dict.update(sample)
    # print(res_dict)
    return res_dict
