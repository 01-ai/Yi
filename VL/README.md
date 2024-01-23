## Quick Start 
1. Dnowload the Yi-VL model.

Model |       Download
|---|---
Yi-VL-34B |• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-VL-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-VL-34B/summary)
Yi-VL-6B | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-VL-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-VL-6B/summary)

2. Set the parameter `mm_vision_tower` in `config.json` to the local ViT path. 
3. To set up the environment and install the required packages, execute the following command.
```
git clone https://github.com/01-ai/Yi.git
cd Yi/VL
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -r requirements.txt
```
4. To perform inference of Yi-VL, execute the following command.
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
5. To evaluate [MMMU](https://mmmu-benchmark.github.io) benchmark, download the [benchmark](https://huggingface.co/datasets/MMMU/MMMU) and execute the following command.
```
python llava/eval/01_eval_mmmu.py \
--exp_name "mmmu" \
--config_path llava/eval/mmmu_configs/yi.yaml \
--data_path MMMU-data-path \
--model_path yi-vl-model-path \
--conv_mode mm_default \
--categories ALL \
--results_dir .
```
6. To evaluate [CMMMU](https://cmmmu-benchmark.github.io) benchmark, download the [benchmark](https://github.com/CMMMU-Benchmark/CMMMU) and execute the following command.
```
python llava/eval/01_eval_cmmmu.py \
--exp_name "cmmmu" \
--config_path llava/eval/cmmmu_configs/yi.yaml \
--data_path cmmmu-data-path \
--model_path yi-vl-model-path \
--conv_mode mm_default \
--categories ALL \
--results_dir .
```

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