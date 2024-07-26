### 🌟使用SWIFT微调
SWIFT是ModelScope开源的一款框架，支持多模态大模型的训练、推理、评测和部署。并且可以直接实现模型训练评测到应用的完整链路。
接下来我们就开始使用SWIFT对Yi模型进行微调！

#### 安装

首先我们拉取SWIFT的代码仓库：

``````bash
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
``````

#### 开始微调

这里我们使用CLI进行微调具体代码如下：

``````bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path 01ai/Yi-1.5-6B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
``````

`--model_id_or_path`可以更换使用的模型

`--dataset` 选择数据集

`--output_dir` 微调后模型的存放位置

如果你想了解更多的关于SWIFT的内容，你取点击[这里](https://github.com/modelscope/swift)！
