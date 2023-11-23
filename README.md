<div align="center">
<p align="center">
<img src="https://github.com/01-ai/Yi/raw/main/assets/img/Yi.svg?sanitize=true" width="200px">
</p>

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

## Introduction

The **Yi** series models are large language models trained from scratch by
developers at [01.AI](https://01.ai/).

## News

<details open>
<summary>🎯 <b>2023/11/23</b>: The chat models are open to public.</summary>

This release contains two chat models based on previous released base models, two 8-bits models quntinized by GPTQ, two 4-bits models quantinized by AWQ.

- `Yi-34B-Chat`
- `Yi-34B-Chat-4bits`
- `Yi-34B-Chat-8bits`
- `Yi-6B-Chat`
- `Yi-6B-Chat-4bits`
- `Yi-6B-Chat-8bits`

You can try some of them interactively at:

- [HuggingFace](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
- [Replicate](https://replicate.com/01-ai)
</details>

<details>
<summary>🔔 <b>2023/11/23</b>: The commercial licensing agreement for the Yi series models is updated to v2.1.</summary>
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

## Model Performance

### Base Model Performance

| Model         |   MMLU   |  CMMLU   |  C-Eval  |  GAOKAO  |   BBH    | Common-sense Reasoning | Reading Comprehension | Math & Code |
| :------------ | :------: | :------: | :------: | :------: | :------: | :--------------------: | :-------------------: | :---------: |
|               |  5-shot  |  5-shot  |  5-shot  |  0-shot  | 3-shot@1 |           -            |           -           |      -      |
| LLaMA2-34B    |   62.6   |    -     |    -     |    -     |   44.1   |          69.9          |         68.0          |    26.0     |
| LLaMA2-70B    |   68.9   |   53.3   |    -     |   49.8   |   51.2   |          71.9          |         69.4          |    36.8     |
| Baichuan2-13B |   59.2   |   62.0   |   58.1   |   54.3   |   48.8   |          64.3          |         62.4          |    23.0     |
| Qwen-14B      |   66.3   |   71.0   |   72.1   |   62.5   |   53.4   |          73.3          |         72.5          |  **39.8**   |
| Skywork-13B   |   62.1   |   61.8   |   60.6   |   68.1   |   41.7   |          72.4          |         61.4          |    24.9     |
| InternLM-20B  |   62.1   |   59.0   |   58.8   |   45.5   |   52.5   |          78.3          |           -           |    30.4     |
| Aquila-34B    |   67.8   |   71.4   |   63.1   |    -     |    -     |           -            |           -           |      -      |
| Falcon-180B   |   70.4   |   58.0   |   57.8   |   59.0   |   54.0   |          77.3          |         68.8          |    34.0     |
| Yi-6B         |   63.2   |   75.5   |   72.0   |   72.2   |   42.8   |          72.3          |         68.7          |    19.8     |
| Yi-6B-200K    |   64.0   |   75.3   |   73.5   |   73.9   |   42.0   |          72.0          |         69.1          |    19.0     |
| **Yi-34B**    | **76.3** | **83.7** |   81.4   |   82.8   | **54.3** |        **80.1**        |         76.4          |    37.1     |
| Yi-34B-200K   |   76.1   |   83.6   | **81.9** | **83.4** |   52.7   |          79.7          |       **76.6**        |    36.3     |

While benchmarking open-source models, we have observed a disparity between the
results generated by our pipeline and those reported in public sources (e.g.
OpenCompass). Upon conducting a more in-depth investigation of this difference,
we have discovered that various models may employ different prompts,
post-processing strategies, and sampling techniques, potentially resulting in
significant variations in the outcomes. Our prompt and post-processing strategy
remains consistent with the original benchmark, and greedy decoding is employed
during evaluation without any post-processing for the generated content. For
scores that were not reported by the original authors (including scores reported
with different settings), we try to get results with our pipeline.

To evaluate the model's capability extensively, we adopted the methodology
outlined in Llama2. Specifically, we included PIQA, SIQA, HellaSwag, WinoGrande,
ARC, OBQA, and CSQA to assess common sense reasoning. SquAD, QuAC, and BoolQ
were incorporated to evaluate reading comprehension. CSQA was exclusively tested
using a 7-shot setup, while all other tests were conducted with a 0-shot
configuration. Additionally, we introduced GSM8K (8-shot@1), MATH (4-shot@1),
HumanEval (0-shot@1), and MBPP (3-shot@1) under the category "Math & Code". Due
to technical constraints, we did not test Falcon-180 on QuAC and OBQA; the score
is derived by averaging the scores on the remaining tasks. Since the scores for
these two tasks are generally lower than the average, we believe that
Falcon-180B's performance was not underestimated.

### Chat Model Performance

| Model                   | MMLU      | MMLU      | CMMLU     | CMMLU     | C-Eval(val)<sup>*</sup> | C-Eval(val)<sup>*</sup> | Truthful QA | BBH       | BBH       | GSM8k     | GSM8k     |
| ----------------------- | --------- | --------- | --------- | --------- | ----------------------- | ----------------------- | ----------- | --------- | --------- | --------- | --------- |
|                         | 0-shot    | 5-shot    | 0-shot    | 5-shot    | 0-shot                  | 5-shot                  | 0-shot      | 0-shot    | 3-shot    | 0-shot    | 4-shot    |
| LLaMA2-13B-Chat         | 50.88     | 47.33     | 27.47     | 35.08     | 27.93                   | 35.88                   | 36.84       | 32.90     | 58.22     | 36.85     | 2.73      |
| LLaMA2-70B-Chat         | 59.42     | 59.86     | 36.10     | 40.99     | 34.99                   | 41.31                   | 53.95       | 42.36     | 58.53     | 47.08     | 58.68     |
| Baichuan2-13B-Chat      | 55.09     | 50.14     | 58.64     | 59.47     | 56.02                   | 54.75                   | 48.98       | 38.81     | 47.15     | 45.72     | 23.28     |
| Qwen-14B-Chat           | 63.99     | 64.98     | 67.73     | 70.57     | 66.12                   | 70.06                   | 52.49       | 49.65     | 54.98     | 59.51     | 61.18     |
| InternLM-Chat-20B       | 55.55     | 57.42     | 53.55     | 53.75     | 51.19                   | 53.57                   | 51.75       | 42.41     | 36.68     | 15.69     | 43.44     |
| AquilaChat2-34B v1.2    | 65.15     | 66.70     | 67.51     | 70.02     | **82.99**               | **89.38**               | **64.33**   | 20.12     | 34.28     | 11.52     | 48.45     |
| Yi-6B-Chat              | 58.24     | 60.99     | 69.44     | 74.71     | 68.80                   | 74.22                   | 50.58       | 39.70     | 47.15     | 38.44     | 44.88     |
| Yi-6B-Chat-8bits(GPTQ)  | 58.29     | 60.96     | 69.21     | 74.69     | 69.17                   | 73.85                   | 49.85       | 40.35     | 47.26     | 39.42     | 44.88     |
| Yi-6B-Chat-4bits(AWQ)   | 56.78     | 59.89     | 67.70     | 73.29     | 67.53                   | 72.29                   | 50.29       | 37.74     | 43.62     | 35.71     | 38.36     |
| Yi-34B-Chat             | **67.62** | 73.46     | **79.11** | **81.34** | 77.04                   | 78.53                   | 62.43       | 51.41     | **71.74** | **71.65** | **75.97** |
| Yi-34B-Chat-8bits(GPTQ) | 66.24     | **73.69** | 79.05     | 81.23     | 76.82                   | 78.97                   | 61.84       | **52.08** | 70.97     | 70.74     | 75.74     |
| Yi-34B-Chat-4bits(AWQ)  | 65.77     | 72.42     | 78.21     | 80.50     | 75.71                   | 77.27                   | 61.84       | 48.30     | 69.39     | 70.51     | 74.00     |

We evaluated various benchmarks using both zero-shot and few-shot methods, except for TruthfulQA. Generally, the zero-shot approach is more common in chat models. Our evaluation strategy involves generating responses while following instructions explicitly or implicitly (such as using few-shot examples). We then isolate relevant answers from the generated text. Some models are not well-suited to produce output in the specific format required by instructions in few datasets, which leads to suboptimal results. 

<strong>*</strong>: C-Eval results are evaluated on the validation datasets

### Quantized Chat Model Performance

We also provide both 4-bit (AWQ) and 8-bit (GPTQ) quantized Yi chat models. Evaluation results on various benchmarks have shown that the quantized models have negligible losses. Additionally, they reduce the memory footprint size. After testing different configurations of prompts and generation lengths, we highly recommend following the guidelines in the memory footprint table below when selecting a device to run our models.

|                         | batch=1 | batch=4 | batch=16 | batch=32 |
| ----------------------- | ------- | ------- | -------- | -------- |
| Yi-34B-Chat             | 65GiB   | 68GiB   | 76GiB    | >80GiB   |
| Yi-34B-Chat-8bits(GPTQ) | 35GiB   | 37GiB   | 46GiB    | 58GiB    |
| Yi-34B-Chat-4bits(AWQ)  | 19GiB   | 20GiB   | 30GiB    | 40GiB    |
| Yi-6B-Chat              | 12GiB   | 13GiB   | 15GiB    | 18GiB    |
| Yi-6B-Chat-8bits(GPTQ)  | 7GiB    | 8GiB    | 10GiB    | 14GiB    |
| Yi-6B-Chat-4bits(AWQ)   | 4GiB    | 5GiB    | 7GiB     | 10GiB    |

Note: All the numbers in the table represent the minimum recommended memory for running models of the corresponding size.

### Limitations of Chat Model

The released chat model has undergone exclusive training using Supervised Fine-Tuning (SFT). Compared to other standard chat models, our model produces more diverse responses, making it suitable for various downstream tasks, such as creative scenarios. Furthermore, this diversity is expected to enhance the likelihood of generating higher quality responses, which will be advantageous for subsequent Reinforcement Learning (RL) training.

However, this higher diversity might amplify certain existing issues, including:

- **Hallucination**: This refers to the model generating factually incorrect or nonsensical information. With the model's responses being more varied, there's a higher chance of hallucination that are not based on accurate data or logical reasoning.
- **Non-determinism in re-generation**: When attempting to regenerate or sample responses, inconsistencies in the outcomes may occur. The increased diversity can lead to varying results even under similar input conditions.
- **Cumulative Error**: This occurs when errors in the model's responses compound over time. As the model generates more diverse responses, the likelihood of small inaccuracies building up into larger errors increases, especially in complex tasks like extended reasoning, mathematical problem-solving, etc.

To achieve more coherent and consistent responses, it is advisable to adjust generation configuration parameters such as`temperature`,`top_p`, or`top_k`. These adjustments can help in the balance between creativity and coherence in the model's outputs.



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
[HuggingFace](https://huggingface.co/01-ai) automatically in the next step. You
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

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B", trust_remote_code=True)
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

Once finished, you can then evaluate the resulted model as follows:

```bash
python quantization/awq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

For more detailed explanation, please read the [doc](https://github.com/01-ai/Yi/tree/main/quantization/awq)

## Ecosystem

🤗 You are encouraged to create a PR and share your awesome work built on top of
the Yi series models.

- Serving
  - [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models): Efficiently run Yi models locally.
- Quantization
  - [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
  - [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ)
- Finetuning
  - [NousResearch/Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B)

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
via applications. All usage must adhere to the [Model License
Agreement 2.0](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).
To apply for the official commercial license, please contact us
([yi@01.ai](mailto:yi@01.ai)).
