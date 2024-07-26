### 🌟Quantize with SWIFT

SWIFT is a framework open-sourced by modelscope that supports training, inference, evaluation, and deployment of multi-modal large models. It can directly implement the complete chain from model training and evaluation to application.

Quantization using SWIFT is very convenient, and can be completed in just a few steps. There are many parameters that can be adjusted during quantization, such as the model to be quantized, precision, method, etc. For details, please refer to the [official documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md).

#### Installation

First, let's install ms-swift:

``````bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
``````

swift supports quantizing models using awq, gptq, bnb, hqq, and eetq techniques.

You can directly install the quantization method you want to use:

``````bash
# Use awq quantization:
# autoawq and cuda versions have a correspondence, please select the version according to `https://github.com/casper-hansen/AutoAWQ`
pip install autoawq -U

# Use gptq quantization:
# auto_gptq and cuda versions have a correspondence, please select the version according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`
pip install auto_gptq -U

# Use bnb quantization:
pip install bitsandbytes -U

# Use hqq quantization:
# pip install transformers>=4.41
pip install hqq

# Use eetq quantization:
# pip install transformers>=4.41

# Refer to https://github.com/NetEase-FuXi/EETQ
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
``````

If you encounter errors during runtime, you can align the environment (optional):

``````bash
# Environment alignment (usually not required to run. If you encounter errors, you can run the following code, the repository uses the latest environment for testing)
pip install -r requirements/framework.txt -U
pip install -r requirements/llm.txt -U
``````

#### Start Quantization with swift

We will use awq and hqq quantization as examples for teaching.

##### AWQ Quantization with swift

awq quantization requires a dataset, you can use a custom dataset here, here we use alpaca-zh alpaca-en sharegpt-gpt4:default as the quantization dataset:

``````bash
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type yi-1_5-6b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4:default --quant_method awq
``````

After quantization, you can also use swift for inference, as follows:

Replace `model_type` with the type of your model.

Replace `model_id_or_path` with the path to your quantized model.

``````bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type yi-1_5-6b-chat \
    --model_id_or_path yi-1_5-6b-chat-awq-int4
``````

##### HQQ Quantization with swift

For bnb, hqq, and eetq, we only need to use swift infer for quick quantization and inference.

You can modify the quantization method with `quant_method`.

Replace `model_type` with the type of your model.

Replace `quantization_bit` with the desired quantization bit.

``````bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type yi-1_5-6b-chat \
    --quant_method hqq \
    --quantization_bit 4
``````