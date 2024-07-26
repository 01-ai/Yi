### 🌟Quantize with AutoAWQ

AutoAWQ is an easy-to-use 4-bit quantization model package. Compared to FP16, AutoAWQ can increase model speed by 3 times and reduce memory requirements by 3 times.

AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm to quantize LLMs.

All the following demos use Yi-1.5-6B-Chat as an example.

Here are the memory and disk usage for this demo:

| Model | Memory Usage | Disk Usage |
|--|------|-------|
| Yi-1.5-6B-Chat | 6G   | 24.5G |

#### Installation
AWQ's version compatibility issues are prone to errors. First we check the version of torch and cuda:

```python
import torch
print(torch.__version__)
```
It should be noted that to install using pip, you must meet the requirement cuda>=12.1:
```shell
pip install autoawq
```
For CUDA 11.8, ROCm 5.6 and ROCm 5.7, installation from source is recommended:

```shell
git clone https://github.com/casper-hansen/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

#### Load the model
AWQ is fully compatible with transformers, you can directly paste the huggingface model path.

Or you can directly replace the model path with a locally downloaded model or a model you have fine-tuned:
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
# model_path is the path to the model, here we load the Yi model from huggingface, if you have a fine-tuned Yi model, you can also directly replace model_path
model_path = '01-ai/Yi-1.5-6B-Chat'
# quant_path is the path to the quantized model
quant_path = 'Yi-1.5-6B-Chat-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load the model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(
    model_path
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)
```
#### Save the model
The quantized model can be saved directly:
```python
# Save the model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```
You can also directly mount the model to the cloud drive for faster and more convenient download:
```python
from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Synchronize files or folders to a specific path on Google Drive
# Suppose the file or folder you want to synchronize is in the current working directory of Colab
# And you want to sync it to the MyDrive/Yi-1.5-6B-Chat-awq folder on Google Drive

# Define the path to the local file or folder
local_path = 'Yi-1.5-6B-Chat-awq'

# Define the target path on Google Drive
drive_path = '/content/drive/MyDrive/Yi-1.5-6B-Chat-awq'

# Sync operation
# Use copytree
shutil.copytree(local_path, drive_path)
print(f"Folder '{local_path}' synced to '{drive_path}'.")

```
#### Using the quantized model
We can use the quantized model directly through transformers:
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
# Prompt
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```