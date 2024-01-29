#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    ：2024/1/29 15:32 
# @Author  ：likuan@datagrand.com
import json
import os

dataset_path = '/data/dataset/visual/银行回单/out800_entity20_pos1_neg0'
saved_name = '银行回单_yi.json'
saved_path = os.path.join(dataset_path, saved_name)
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')


data_list = []
for image_name in os.listdir(images_path):
    label_name = image_name.replace('.jpg', '.json')
    data_id = image_name.replace('.jpg', '')
    image_path = os.path.join(images_path, image_name)
    with open(os.path.join(labels_path, label_name), 'r', encoding='utf-8') as f:
        ori_conv = json.load(f)['conversations']
    user_conv = ori_conv[0]
    assistant_conv = ori_conv[1]
    target_conv = {
        "id": data_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image_placeholder>\n" + user_conv['content']
            },
            {
                "from": "assistant",
                "value": assistant_conv['content']
            }
        ]
    }
    data_list.append(target_conv)

with open(saved_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)