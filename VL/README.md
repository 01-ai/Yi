## Quick Start 
1. Dnowload the Yi-VL model.

Model |       Download
|---|---
Yi-VL-34B |• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-VL-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-VL-34B/summary)
Yi-VL-6B | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-VL-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-VL-6B/summary)

2. To set up the environment and install the required packages, execute the following command.
```
git clone https://github.com/01-ai/Yi.git
cd Yi/VL
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -r requirements.txt
```
3. To perform inference of Yi-VL, execute the following command.
```python
python single_inference.py --model-path path-to-yi-vl-model --image-file path-to-image --question question-content
```
A quick example:
```python
CUDA_VISIBLE_DEVICES=0 python single_inference.py --model-path ../model/Yi-VL-34B --image-file images/cats.jpg --question "Describe the cats and what they are doing in detail."
```
Since the temperature is set to 0.2 by default, the ourput is not always the same. An example output is:
```
----------
question: Describe the cats and what they are doing in detail.
outputs: In the image, there are three cats situated on a stone floor. The first cat, with a mix of black, orange, and white fur, is actively eating from a metal bowl. The second cat, which is entirely black, is also engaged in eating from a separate metal bowl. The third cat, a mix of gray and white, is not eating but is instead looking off to the side, seemingly distracted from the food. The bowls are positioned close to each other, and the cats are all within a similar proximity to the bowls. The scene captures a typical moment of feline behavior, with some cats enjoying their meal while others appear indifferent or distracted.
----------
```

## Web demo
You can build a web UI demo for **Yi-VL** models
```python
python web_demo.py --model-path path-to-yi-vl-model
```

## Command Line Interface

Perform conversational inference via the command line interface.

```python
python cli.py --model-path path-to-yi-vl-model --image path-to-image
```

A quick example:

```
CUDA_VISIBLE_DEVICES=0 python cli.py --model-path ../model/Yi-VL-34B --image-file images/cats.jpg

Human: what is in this image
Assistant: cats
Human: what are they doing
Assistant: eating
Human: what are they eating
Assistant: cat food
```

## Finetuning
1. Prepare data

Prepare your own data into the following JSON format.
```json
[
  {
    "image": "images/cat.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image_placeholder>\nDescribe the cats and what they are doing in detail."
      },
      {
        "from": "assistant",
        "value": "In the image, there are three cats situated on a stone floor. The cat on the left is a calico cat, its coat a mix of white, orange, and black. It's eating from a metal bowl. In the middle, there's a gray cat, also eating from a metal bowl. On the right, there's a black cat, eating from a plastic bowl. The cats are all facing away from the camera, engrossed in their meal. The stone floor they're on is gray, and a concrete wall forms the backdrop of the scene. The image captures a peaceful moment of these cats enjoying their food."
      },
    ]
  },
  ...
]
```

2. Finetune Yi-VL

Training scripts are provided in the `scripts` folder. You can use `scripts/finetune.sh`, `scripts/finetune_lora.sh` or `scripts/finetune_qlora.sh` to finetune Yi-VL with your own dataset.

Before running the scrips, you should specify the following parameters. 
- `--model_name_or_path`: the path to Yi-VL model; you can use 6B or 34B model.
- `--data_path`: the path to your own dataset.
- `--image_folder`: the path to the image data folder.
- `--vision_tower`: the path to the ViT model, usually found in the Yi-VL base model folder.

3. Merge lora (Optional)

If you use `lora` or `qlora` for finetuning, you need to merge the lora parameters into the Yi-VL model after finetuning. You can use `scripts/merge_lora.sh` to merge the lora parameters.

## Major difference with LLaVA
1. We change the image token from ```<image>``` to ```<image_placeholder>```. The system prompt is modified to:
```
This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。

### Human: <image_placeholder>
Describe the cats and what they are doing in detail.
### Assistant:
```
2. We add LayNorm in the two-layer MLP of the projection module.
3. We train the parameters of ViT and scale up the input image resolution.
4. We utilize Laion-400M data for pretraining.