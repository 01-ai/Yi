### 🌟使用SWIFT量化

SWIFT是modelscope开源的一款框架，支持多模态大模型的训练、推理、评测和部署。并且可以直接实现模型训练评测到应用的完整链路。

使用SWIFT量化非常方便，只需要几步即可完成量化，在量化中有许多的参数可以进行调节，比如量化的模型、精度、方式等等，具体的可以参考[官方文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md)。

#### 安装

首先我们先安装ms-swift：

``````bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
``````

swift支持使用awq、gptq、bnb、hqq、eetq技术对模型进行量化。

如果你想使用哪一种量化方式就可以直接进行安装：

``````bash
# 使用awq量化:
# autoawq和cuda版本有对应关系，请按照`https://github.com/casper-hansen/AutoAWQ`选择版本
pip install autoawq -U

# 使用gptq量化:
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 使用bnb量化：
pip install bitsandbytes -U

# 使用hqq量化：
# pip install transformers>=4.41
pip install hqq

# 使用eetq量化：
# pip install transformers>=4.41

# 参考https://github.com/NetEase-FuXi/EETQ
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
``````

如果你运行报错可以进行环境对齐(选择)：

``````bash
# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
``````

#### 使用swift开始量化

我们使用awq和hqq量化为示例进行教学。

##### 使用swift进行awq量化

awq量化需要数据集，这里可以使用自定义数据集，这里使用alpaca-zh alpaca-en sharegpt-gpt4:default作为量化数据集：

``````bash
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type yi-1_5-6b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4:default --quant_method awq
``````

量化完成后进行推理同样也可以使用swift具体如下：

`model_type` 替换模型的类型

`model_id_or_path`量化类型

``````bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type yi-1_5-6b-chat \
    --model_id_or_path yi-1_5-6b-chat-awq-int4
``````

##### 使用swift进行hqq量化

对于bnb、hqq、eetq，我们只需要使用swift infer来进行快速量化并推理。

`quant_method`可以修改量化方法

`model_type` 替换模型的类型

`quantization_bit`量化类型

``````bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type yi-1_5-6b-chat \
    --quant_method hqq \
    --quantization_bit 4
``````