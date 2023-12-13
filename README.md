<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_dark.svg" width="200px">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="200px"> 
  <img alt="specify theme context for images" src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg">
</picture>

</br>
</br>

<a href="https://github.com/01-ai/Yi/issues">
  <img src="https://img.shields.io/github/issues/01-ai/Yi?logo=github">
</a>
<a href="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml">
  <img src="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml/badge.svg">
</a>
<a href="https://huggingface.co/01-ai">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-01--ai-blue">
</a>
<a href="https://www.modelscope.cn/organization/01ai/">
  <img src="https://img.shields.io/badge/ModelScope-01--ai-blue">
</a>
<a href="https://wisemodel.cn/organization/01.AI">
  <img src="https://img.shields.io/badge/WiseModel-01--ai-blue">
</a>
<a href="https://replicate.com/01-ai">
  <img src="https://img.shields.io/badge/Replicate-01--ai-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Qm94PSIwIDAgMTAwMCAxMDAwIiBjbGFzcz0ibG9nbyIgZmlsbD0iY3VycmVudENvbG9yIiB4bWw6c3BhY2U9InByZXNlcnZlIj4KICA8Zz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCw0MjcuNiAxMDAwLDU0MC42IDYwMy40LDU0MC42IDYwMy40LDEwMDAgNDc3LDEwMDAgNDc3LDQyNy42IAkiPjwvcG9seWdvbj4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCwyMTMuOCAxMDAwLDMyNyAzNjQuOCwzMjcgMzY0LjgsMTAwMCAyMzguNCwxMDAwIDIzOC40LDIxMy44IAkiPjwvcG9seWdvbj4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCwwIDEwMDAsMTEzLjIgMTI2LjQsMTEzLjIgMTI2LjQsMTAwMCAwLDEwMDAgMCwwIAkiPjwvcG9seWdvbj4KICA8L2c+Cjwvc3ZnPg==">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
  <img src="https://img.shields.io/badge/Model_License-Model_Agreement-lightblue">
</a>
<a href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B">
</a>
</div>

Table of Contents

- What is Yi?
  - [Introduction](#introduction)
  - Capabilities (WIP) 
  - Models (WIP)
  - [News](#)

- Why Yi?
  - Use cases (WIP)
  - [Ecosystem](#ecosystem)
  - [Benchmarks](#benchmarks)

- Who can use Yi?

- How to use Yi? 
  - [Download]
  - [Quick start]

# What is Yi? 

## Introduction 📌

Yi series models are the next generation of open source large language models trained from strach by [01.AI](https://01.ai/).

- As general-purpose language models, Yi series models excel in handling various tasks, including natural language generation, assistant-like chat, and more.

- Targeted as a bilingual language model and trained on 3T multilingual corpus, Yi series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more. For example,

  - For English language capability, Yi series models ranked 2nd (just behind GPT-4), outperforming other LLMs (such as LLaMA2-chat-70B, Claude 2, and ChatGPT) on the [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) in Dec 2023.
  
  - For Chinese language capability, Yi series models landed in 2nd place (following GPT4), surpassing other LLMs (such as Baidu ERNIE, Qwen, and Baichuan) on the [SuperCLUE](https://www.superclueai.com/) in Oct 2023.

- (Credits to LLaMA) Thanks to the Transformer and LLaMA open-source communities, as they reducing the efforts required to build from scratch and enabling the utilization of the same tools within the AI ecosystem. If you're interested in Yi's adoption of LLaMA architecture and license usage policy, see [Yi's relation with LLaMA](./docs/yi_relation_llama.md).

## News 📌

<details open>
<summary>🎯 <b>2023/11/23</b>: The chat models are open to public.</summary>

This release contains two chat models based on previous released base models, two 8-bits models quantized by GPTQ, two 4-bits models quantized by AWQ.

- `Yi-34B-Chat`
- `Yi-34B-Chat-4bits`
- `Yi-34B-Chat-8bits`
- `Yi-6B-Chat`
- `Yi-6B-Chat-4bits`
- `Yi-6B-Chat-8bits`

You can try some of them interactively at:

- [Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
- [Replicate](https://replicate.com/01-ai)
</details>

<details open>
<summary>🔔 <b>2023/11/23</b>: The Yi Series Models Community License Agreement is updated to v2.1.</summary>
</details>

<details>
<summary>🔥 <b>2023/11/08</b>: Invited test of Yi-34B chat model.</summary>

Application form:

- [English](https://cn.mikecrm.com/l91ODJf)
- [Chinese](https://cn.mikecrm.com/gnEZjiQ)

</details>

<details>
<summary>🎯 <b>2023/11/05</b>: The base model of <code>Yi-6B-200K</code> and <code>Yi-34B-200K</code>.</summary>

This release contains two base models with the same parameter sizes of previous
release, except that the context window is extended to 200K.

</details>

<details>
<summary>🎯 <b>2023/11/02</b>: The base model of <code>Yi-6B</code> and <code>Yi-34B</code>.</summary>

The first public release contains two bilingual (English/Chinese) base models
with the parameter sizes of 6B and 34B.  Both of them are trained with 4K
sequence length and can be extended to 32K during inference time.

</details>

# Why Yi? 

## Ecosystem 📌

Yi has a comprehensive ecosystem, offering a range of tools, services, and models to enrich your experiences and maximize productivity.

### Yi's upstream

The Yi series models follow the same model architecture as LLaMA. By choosing Yi, you can leverage existing tools, libraries, and resources within the LLaMA ecosystem, eliminating the need to create new tools and enhancing development efficiency.

For example, the Yi series models are saved in the format of the LLaMA model. You can directly use LLaMAForCausalLM and LLaMATokenizer to load the model.

```
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34b", device_map="auto")
```

### Yi's downstream

> 💡 Tip
> 
> - Feel free to create a PR and share the fantastic work you've built using the Yi series models.
>
> - To help others quickly understand your work, it is recommended to use the format of `<model-name>: <model-intro> + <model-highlights>`.

#### Serving 

If you want to get up with Yi in a few minutes, you can use the following services built upon Yi.

- [Yi-34B-Chat](https://platform.lingyiwanwu.com/) (Yi official beta): you can chat with it. **Note** that currently it's available through a whitelist. Welcome to apply and experience it firsthand!
  
- [Yi-6B-Chat (Replicate)](https://replicate.com/01-ai): you can use this model with more options by setting additional parameters and calling APIs.
  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models): you can use this service to run Yi models locally with added flexibility and customization.
  
#### Quantation

If you have limited computational capabilities, you can use Yi's quantized models as follows. 

These quantized models have reduced precision and but offer increased efficiency, such as faster inference speed and smaller RAM usage.

- TheBloke/Yi-34B-GPTQ 
- TheBloke/Yi-34B-GGUF
- TheBloke/Yi-34B-AWQ
  
#### Fine-tuning

If you're seeking to explore the diverse capabilities within Yi's thriving family, you can delve into Yi's fine-tuned models as below.

- [TheBloke Models](https://huggingface.co/TheBloke): this site hosts numerous fine-tuned models derived from various LLMs including Yi. 
  
  This is not an exhaustive list for Yi, but to name a few sorted on downloads:
  - [TheBloke/dolphin-2_2-yi-34b-AWQ](https://huggingface.co/TheBloke/dolphin-2_2-yi-34b-AWQ)
  - [TheBloke/Yi-34B-Chat-AWQ](https://huggingface.co/TheBloke/Yi-34B-Chat-AWQ)
  - [TheBloke/Yi-34B-Chat-GPTQ](https://huggingface.co/TheBloke/Yi-34B-Chat-GPTQ)
  
- [SUSTech/SUS-Chat-34B](https://huggingface.co/SUSTech/SUS-Chat-34B): this model ranked first among all models below 70B and outperformed the twice larger deepseek-llm-67b-chat. You can check the result on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
  
- [OrionStarAI/OrionStar-Yi-34B-Chat-Llama](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat-Llama): this model excelled beyond other models (such as GPT-4, Qwen-14B-Chat, Baichuan2-13B-Chat) in C-Eval and CMMLU evaluations on the [OpenCompass LLM Leaderboard](https://opencompass.org.cn/leaderboard-llm). 
  
- [NousResearch/Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B): this model is trained with 200K context length and 3 epochs on the Capybara dataset. 

## Benchmarks 📌




# Who can use Yi?

Yi series models are free for personal and academic purposes. For free commercial use, you only need to send an email to [get official commercial permission](https://www.lingyiwanwu.com/yi-license).

# How to use Yi?

For how to [DOWNLOAD](https://huggingface.co/01-ai) and GET STARTED with Yi quickly!

## Usage

Feel free to [create an issue](https://github.com/01-ai/Yi/issues/new) if you
encounter any problem when using the **Yi** series models.

### 1. Prepare development environment

#### 1.1 Docker
The best approach to try the **Yi** series models is through Docker with GPUs. We
provide the following docker images to help you get started.

- `registry.lingyiwanwu.com/ci/01-ai/yi:latest`
- `ghcr.io/01-ai/yi:latest`

Note that the `latest` tag always points to the latest code in the `main`
branch. To test a stable version, please replace it with a specific
[tag](https://github.com/01-ai/Yi/tags).

#### 1.2 Local development environment
We use [`conda-lock`](https://github.com/conda/conda-lock) to generate fully reproducible lock files for conda environments. You can refer to [conda-lock.yml](./conda-lock.yml) for the exact versions of the dependencies. Additionally, we utilize [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for installing these dependencies.

To install the dependencies, please follow these steps:
1. Install `micromamba` by following the instructions available [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
2. Execute `micromamba install -y -n yi -f conda-lock.yml` to create a conda environment named `yi` and install the necessary dependencies.

### 2. Download the model (optional)

By default, the model weights and tokenizer will be downloaded from
[Hugging Face](https://huggingface.co/01-ai) automatically in the next step. You
can also download them manually from the following places:

- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

### 3. Examples

#### 3.1 Use the chat model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '01-ai/Yi-34b-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
```


To construct the prompt template manually, you can refer the `chat_template` field in the `tokenizer_config.json` [file](https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json#L60).

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

#### 3.2 Use the base model

```bash
python demo/text_generation.py
```

To reuse the downloaded models in the previous step, you can provide the extra
`--model` argument:

```bash
python demo/text_generation.py  --model /path/to/model
```

Or if you'd like to get your hands dirty:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B")
inputs = tokenizer("There's a place where time stands still. A place of breath taking wonder, but also", return_tensors="pt")
max_length = 256

outputs = model.generate(
    inputs.input_ids.cuda(),
    max_length=max_length,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    repetition_penalty=1.3,
    no_repeat_ngram_size=5,
    temperature=0.7,
    top_k=40,
    top_p=0.8,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

<details>

<summary>Output</summary>

**Prompt**: There's a place where time stands still. A place of breath taking wonder, but also

**Generation**: There's a place where time stands still. A place of breath taking wonder, but also of great danger. A place where the very air you breathe could kill you. A place where the only way to survive is to be prepared.
The place is called the Arctic.
The Arctic is a vast, frozen wilderness. It is a place of extremes. The temperatures can drop to -40 degrees Celsius. The winds can reach speeds of 100 kilometers per hour. The sun can shine for 24 hours a day, or not at all for weeks on end.
The Arctic is also a place of great beauty. The ice and snow are a pristine white. The sky is a deep blue. The sunsets are spectacular.
But the Arctic is also a place of great danger. The ice can be treacherous. The winds can be deadly. The sun can be blinding.
The Arctic is a place where the only way to survive is to be prepared.
The Arctic is a place of extremes. The temperatures can drop to -40 degrees Celsius. The winds can reach speeds of 100 kilometers per hour. The sun can shine for 24 hours a day, or not at all for weeks on end.
The Arctic is a place of great beauty. The ice and snow are a

</details>

For more advanced usage, please refer to the
[doc](https://github.com/01-ai/Yi/tree/main/demo).

#### 3.3 Finetuning from the base model:

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

Once finished, you can compare the finetuned model and the base model with the following command:

```bash
bash finetune/scripts/run_eval.sh
```

For more advanced usage like fine-tuning based on your custom data, please refer
the [doc](https://github.com/01-ai/Yi/tree/main/finetune).

#### 3.4 Quantization

##### GPT-Q
```bash
python quantization/gptq/quant_autogptq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

Once finished, you can then evaluate the resulting model as follows:

```bash
python quantization/gptq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

For a more detailed explanation, please read the [doc](https://github.com/01-ai/Yi/tree/main/quantization/gptq)

##### AWQ
```bash
python quantization/awq/quant_autoawq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

Once finished, you can then evaluate the resulting model as follows:

```bash
python quantization/awq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

For more detailed explanation, please read the [doc](https://github.com/01-ai/Yi/tree/main/quantization/awq)

## FAQ

1. **What dataset was this trained with?**

    The dataset we use contains Chinese & English only. We used approximately 3T
    tokens. The detailed number and its construction will be described in the
    upcoming technical report.

## Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

## License

The source code in this repo is licensed under the [Apache 2.0
license](https://github.com/01-ai/Yi/blob/main/LICENSE). The Yi series models
are fully open for academic research and free commercial usage with permission
via applications. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).
For free commercial use, you only need to send an email to [get official commercial permission](https://www.lingyiwanwu.com/yi-license).
