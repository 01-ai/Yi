### 🌟使用AutoAWQ量化

AutoAWQ 是一款易于使用的 4 位量化模型软件包。与 FP16 相比，AutoAWQ 可将模型速度提高 3 倍并将内存需求降低 3 倍。

AutoAWQ 实现了激活感知权重量化 (AWQ) 算法来量化 LLM。

此次所有演示，均采用Yi-1.5-6B-Chat作为示例。

下面是此次演示显存和硬盘占用情况：

| 模型 | 显存使用 | 硬盘占用  |
|--|------|-------|
| Yi-1.5-6B-Chat | 6G   | 24.5G |

#### 安装
AWQ的版本兼容问题比较容易出错 首先我们检查torch和cuda的版本：

```python
import torch
print(torch.__version__)
```
这里需要注意如果想要使用pip进行安装，必须满足cuda>=12.1：
```shell
pip install autoawq
```
对于 CUDA 11.8、ROCm 5.6 和 ROCm 5.7，推荐从源码进行安装：

```shell
git clone https://github.com/casper-hansen/AutoAWQ.git
cd AutoAWQ
pip install -e .
```
#### 加载模型
AWQ完全兼容transformers，你可以直接粘贴huggingface的模型路径。

或者也可以直接将模型路径替换为本地加载下载好的模型或者是你已经微调好的模型：
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
# model_path是模型的路径，这里从huggingface加载Yi模型，如果你有已经微调好的Yi模型，你同样直接替换model_path即可
model_path = '01-ai/Yi-1.5-6B-Chat'
# quant_path为量化后模型的路径
quant_path = 'Yi-1.5-6B-Chat-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(
    model_path
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)
```
#### 保存模型
量化后的模型可以直接保存：
```python
# 保存模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```
同样也可以直接将模型挂载到云盘上，更快速方便的下载：
```python
from google.colab import drive
import shutil

# 挂载Google云端硬盘
drive.mount('/content/drive')

# 将文件或文件夹同步到云端硬盘的指定路径
# 假设你想要同步的文件或文件夹位于Colab的当前工作目录下
# 并且你想要将其同步到云端硬盘的MyDrive/Yi-1.5-6B-Chat-awq文件夹中

# 定义本地文件或文件夹的路径
local_path = 'Yi-1.5-6B-Chat-awq'

# 定义云端硬盘的目标路径
drive_path = '/content/drive/MyDrive/Yi-1.5-6B-Chat-awq'

# 同步操作
# 使用copytree
shutil.copytree(local_path, drive_path)
print(f"文件夹'{local_path}'已同步到'{drive_path}'。")

```
#### 使用量化后的模型
我们通过transformers直接可以使用量化模型：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# model_path = quant_path
model_path = 'Yi-1.5-6B-Chat-awq'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
# 提示词
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```