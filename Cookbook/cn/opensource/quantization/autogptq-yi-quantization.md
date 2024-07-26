### 🌟使用AutoGPTQ量化

AutoGPTQ是一个基于 GPTQ 算法，简单易用且拥有用户友好型接口的大语言模型量化工具包。

我们可以通过AutoGPTQ量化我们的Yi系列模型。

| 模型 | 显存使用 | 硬盘占用 |
|--|------|------|
| Yi-1.5-6B-Chat | 7G   | 27G  |

#### 安装
推荐从源码进行安装：

```shell
git clone https://github.com/AutoGPTQ/AutoGPTQ
cd AutoGPTQ
pip install .
```
#### 加载模型

同样的我们还是可以通过transformers加载模型，或许加载微调好的模型。

只需要替换模型路径即可，具体代码如下。

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# 配置量化超参数
# 加载分词器和模型
model_path = "01-ai/Yi-1.5-6B-Chat"
quant_path = "Yi-1.5-6B-Chat-GPTQ"
quantize_config = BaseQuantizeConfig(
    bits=8, # 量化为8-bit 模型
    group_size=128, # 推荐128
    damp_percent=0.01,
    desc_act=False,  # 设为 False 可以显著提升推理速度
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
```
#### 量化与保存模型
model.quantize(examples)中样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask。

这里需要注意自己的数据格式!
```python
import torch
examples = []
messages = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello! It's great to see you today. How can I assist you"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
model_inputs = tokenizer([text])
input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
examples.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))
model.quantize(examples)

model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
```
#### 使用模型
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from transformers import AutoTokenizer, GenerationConfig

quantized_model_dir = 'Yi-1.5-6B-Chat-GPTQ'

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir,
                                          use_fast=True,
                                          trust_remote_code=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,  
                                           device_map="auto", 
                                           use_safetensors=True, 
                                           trust_remote_code=True)


output = tokenizer.decode(model.generate(
    **tokenizer("<|im_start|>user Hi!<|im_end|> <|im_start|>assistant", return_tensors="pt").to(model.device),
    max_new_tokens=512)[0]
                          )
print(output)
```