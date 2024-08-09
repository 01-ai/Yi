### 🌟Quantize with AutoGPTQ

AutoGPTQ is a large language model quantization toolkit based on the GPTQ algorithm, which is simple, easy to use and has a user-friendly interface.

We can quantize our Yi series models through AutoGPTQ.

| Model | Memory Usage | Disk Usage |
|--|------|------|
| Yi-1.5-6B-Chat | 7G   | 27G  |

#### Installation
Installation from source is recommended:

```shell
git clone https://github.com/AutoGPTQ/AutoGPTQ
cd AutoGPTQ
pip install .
```

#### Load the model

Similarly, we can still load the model through transformers, or perhaps load a fine-tuned model.

Just replace the model path, the specific code is as follows.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Configure quantization hyperparameters
# Load tokenizer and model
model_path = "01-ai/Yi-1.5-6B-Chat"
quant_path = "Yi-1.5-6B-Chat-GPTQ"
quantize_config = BaseQuantizeConfig(
    bits=8, # Quantize to 8-bit model
    group_size=128, # Recommended 128
    damp_percent=0.01,
    desc_act=False,  # Setting to False can significantly improve inference speed
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
```

#### Quantize and save the model
The data type of the samples in model.quantize(examples) should be List[Dict], where the keys of the dictionary are and only input_ids and attention_mask.

Please pay attention to your data format here!
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

#### Using the model
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