<p align="left">
    &nbspEnglish&nbsp | &nbsp; <a href="README_CN.md">中文</a>
</p>
<br><br>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_dark.svg" width="200px">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="200px"> 
  <img alt="specify theme context for images" src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="200px">
</picture>

</br>
</br>

<a href="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml">
  <img src="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml/badge.svg">
</a>
<a href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B">
</a>

</div>

<div id="top"></div>  

<div align="center">
  <h3 align="center">Building the Next Generation of Open-Source and Bilingual LLMs</h3>
</div>
<p align="center">
🤗 <a href="https://huggingface.co/01-ai" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/01ai/" target="_blank">ModelScope</a> • 🟣 <a href="https://wisemodel.cn/organization/01.AI" target="_blank">wisemodel</a>
</p> 

<p align="center">
    👩‍🚀 Ask questions or discuss ideas on <a href="https://github.com/01-ai/Yi/discussions" target="_blank"> GitHub </a>
</p> 

<p align="center">
    👋 Join us on <a href="https://discord.gg/hYUwWddeAu" target="_blank"> 👾 Discord </a> or <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245" target="_blank"> 💬 WeChat </a>
</p> 

<p align="center">
    📝 Check out  <a href="https://arxiv.org/abs/2403.04652"> Yi Tech Report </a>
</p> 

<p align="center">
    📚 Grow at <a href="#learning-hub"> Yi Learning Hub </a>
</p> 

<!-- DO NOT REMOVE ME -->

<hr>

<details open>
<summary></b>📕 Table of Contents</b></summary>

- [What is Yi?](#what-is-yi)
  - [Introduction](#introduction)
  - [Models](#models)
    - [Chat models](#chat-models)
    - [Base models](#base-models)
    - [Model info](#model-info)
  - [News](#news)
- [How to use Yi?](#how-to-use-yi)
  - [Quick start](#quick-start)
    - [Choose your path](#choose-your-path)
    - [pip](#quick-start---pip)
    - [docker](#quick-start---docker)
    - [llama.cpp](#quick-start---llamacpp)
    - [conda-lock](#quick-start---conda-lock)
    - [Web demo](#web-demo)
  - [Fine-tuning](#fine-tuning)
  - [Quantization](#quantization)
  - [Deployment](#deployment)
  - [FAQ](#faq)
  - [Learning hub](#learning-hub)
- [Why Yi?](#why-yi)
  - [Ecosystem](#ecosystem)
    - [Upstream](#upstream)
    - [Downstream](#downstream)
      - [Serving](#serving)
      - [Quantization](#quantization-1)
      - [Fine-tuning](#fine-tuning-1)
      - [API](#api)
  - [Benchmarks](#benchmarks)
    - [Base model performance](#base-model-performance)
    - [Chat model performance](#chat-model-performance)
  - [Tech report](#tech-report)
    - [Citation](#citation)
- [Who can use Yi?](#who-can-use-yi)
- [Misc.](#misc)
  - [Acknowledgements](#acknowledgments)
  - [Disclaimer](#disclaimer)
  - [License](#license)

</details>

<hr>

# What is Yi?

## Introduction 

- 🤖 The Yi series models are the next generation of open-source large language models trained from scratch by [01.AI](https://01.ai/).

- 🙌 Targeted as a bilingual language model and trained on 3T multilingual corpus, the Yi series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more. For example,
  
  - Yi-34B-Chat model **landed in second place (following GPT-4 Turbo)**, outperforming other LLMs (such as GPT-4, Mixtral, Claude) on the AlpacaEval Leaderboard (based on data available up to January 2024).

  - Yi-34B model **ranked first among all existing open-source models** (such as Falcon-180B, Llama-70B, Claude) in **both English and Chinese** on various benchmarks, including Hugging Face Open LLM Leaderboard (pre-trained) and C-Eval (based on data available up to November 2023).
  
  - 🙏 (Credits to Llama) Thanks to the Transformer and Llama open-source communities, as they reduce the efforts required to build from scratch and enable the utilization of the same tools within the AI ecosystem.  

  <details style="display: inline;"><summary> If you're interested in Yi's adoption of Llama architecture and license usage policy, see  <span style="color:  green;">Yi's relation with Llama.</span> ⬇️</summary> <ul> <br>
  
  
> 💡 TL;DR
> 
> The Yi series models adopt the same model architecture as Llama but are **NOT** derivatives of Llama.

- Both Yi and Llama are based on the Transformer structure, which has been the standard architecture for large language models since 2018.

- Grounded in the Transformer architecture, Llama has become a new cornerstone for the majority of state-of-the-art open-source models due to its excellent stability, reliable convergence, and robust compatibility. This positions Llama as the recognized foundational framework for models including Yi.

- Thanks to the Transformer and Llama architectures, other models can leverage their power, reducing the effort required to build from scratch and enabling the utilization of the same tools within their ecosystems.

- However, the Yi series models are NOT derivatives of Llama, as they do not use Llama's weights.

  - As Llama's structure is employed by the majority of open-source models, the key factors of determining model performance are training datasets, training pipelines, and training infrastructure.

  - Developing in a unique and proprietary way, Yi has independently created its own high-quality training datasets, efficient training pipelines, and robust training infrastructure entirely from the ground up. This effort has led to excellent performance with Yi series models ranking just behind GPT4 and surpassing Llama on the [Alpaca Leaderboard in Dec 2023](https://tatsu-lab.github.io/alpaca_eval/). 
</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## News 

<details>
  <summary>🎯 <b>2024-05-13</b>: The <a href="https://github.com/01-ai/Yi-1.5">Yi-1.5 series models </a> are open-sourced, further improving coding, math, reasoning, and instruction-following abilities.</summary>
</details>

<details>
  <summary>🎯 <b>2024-03-16</b>: The <code>Yi-9B-200K</code> is open-sourced and available to the public.</summary>
</details>

<details>
  <summary>🎯 <b>2024-03-08</b>: <a href="https://arxiv.org/abs/2403.04652">Yi Tech Report</a> is published! </summary>
</details>


<details open>
  <summary>🔔 <b>2024-03-07</b>: The long text capability of the Yi-34B-200K has been enhanced. </summary>
  <br>
In the "Needle-in-a-Haystack" test, the Yi-34B-200K's performance is improved by 10.5%, rising from 89.3% to an impressive 99.8%. We continue to pre-train the model on 5B tokens long-context data mixture and demonstrate a near-all-green performance.
</details>

<details open>
  <summary>🎯 <b>2024-03-06</b>: The <code>Yi-9B</code> is open-sourced and available to the public.</summary>
  <br>
<code>Yi-9B</code> stands out as the top performer among a range of similar-sized open-source models (including Mistral-7B, SOLAR-10.7B, Gemma-7B, DeepSeek-Coder-7B-Base-v1.5 and more), particularly excelling in code, math, common-sense reasoning, and reading comprehension.
</details>

<details open>
  <summary>🎯 <b>2024-01-23</b>: The Yi-VL models, <code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code> and <code><a href="https://huggingface.co/01-ai/Yi-VL-6B">Yi-VL-6B</a></code>, are open-sourced and available to the public.</summary>
  <br>
  <code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code> has ranked <strong>first</strong> among all existing open-source models in the latest benchmarks, including <a href="https://arxiv.org/abs/2311.16502">MMMU</a> and <a href="https://arxiv.org/abs/2401.11944">CMMMU</a> (based on data available up to January 2024).</li>
</details>


<details>
<summary>🎯 <b>2023-11-23</b>: <a href="#chat-models">Chat models</a> are open-sourced and available to the public.</summary>
<br>This release contains two chat models based on previously released base models, two 8-bit models quantized by GPTQ, and two 4-bit models quantized by AWQ.

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

<details>
  <summary>🔔 <b>2023-11-23</b>: The Yi Series Models Community License Agreement is updated to <a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">v2.1</a>.</summary>
</details>

<details> 
<summary>🔥 <b>2023-11-08</b>: Invited test of Yi-34B chat model.</summary>
<br>Application form:

- [English](https://cn.mikecrm.com/l91ODJf)
- [Chinese](https://cn.mikecrm.com/gnEZjiQ)
</details>

<details>
<summary>🎯 <b>2023-11-05</b>: <a href="#base-models">The base models, </a><code>Yi-6B-200K</code> and <code>Yi-34B-200K</code>, are open-sourced and available to the public.</summary>
<br>This release contains two base models with the same parameter sizes as the previous
release, except that the context window is extended to 200K.
</details>

<details>
<summary>🎯 <b>2023-11-02</b>: <a href="#base-models">The base models, </a><code>Yi-6B</code> and <code>Yi-34B</code>, are open-sourced and available to the public.</summary>
<br>The first public release contains two bilingual (English/Chinese) base models
with the parameter sizes of 6B and 34B.  Both of them are trained with 4K
sequence length and can be extended to 32K during inference time.

</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## Models

Yi models come in multiple sizes and cater to different use cases. You can also fine-tune Yi models to meet your specific requirements. 

If you want to deploy Yi models, make sure you meet the [software and hardware requirements](#deployment).

### Chat models

| Model | Download  |
|---|---|
|Yi-34B-Chat	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-34B-Chat) |
|Yi-34B-Chat-4bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-4bits/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-34B-Chat-4bits) |
|Yi-34B-Chat-8bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-8bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-8bits/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-34B-Chat-8bits) |
|Yi-6B-Chat| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat) |
|Yi-6B-Chat-4bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-4bits/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-4bits) |
|Yi-6B-Chat-8bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-8bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-8bits/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits) |

<sub><sup> - 4-bit series models are quantized by AWQ. <br> - 8-bit series models are quantized by GPTQ <br> - All quantized models have a low barrier to use since they can be deployed on consumer-grade GPUs (e.g., 3090, 4090). </sup></sub>

### Base models

| Model | Download |
|---|---|
|Yi-34B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits) |
|Yi-34B-200K|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-200K/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits)|
|Yi-9B|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B)  • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-9B)|
|Yi-9B-200K | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B-200K)  • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-9B-200K)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits) |
|Yi-6B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits) |
|Yi-6B-200K	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-200K/summary)  • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-6B-Chat-8bits) |

<sub><sup> - 200k is roughly equivalent to 400,000 Chinese characters.  <br> - If you want to use the previous version of the Yi-34B-200K (released on Nov 5, 2023), run `git checkout 069cd341d60f4ce4b07ec394e82b79e94f656cf` to download the weight. </sup></sub>

### Model info

- For chat and base models

<table>
<thead>
<tr>
<th>Model</th>
<th>Intro</th>
<th>Default context window</th>
<th>Pretrained tokens</th>
<th>Training Data Date</th>
</tr>
</thead>
<tbody><tr>
<td>6B series models</td>
<td>They are suitable for personal and academic use.</td>
<td rowspan="3">4K</td>
<td>3T</td>
<td rowspan="3">Up to June 2023</td>
</tr>
<tr>
<td>9B series models</td>
<td>It is the best at coding and math in the Yi series models.</td>
<td>Yi-9B is continuously trained based on Yi-6B, using 0.8T tokens.</td>
</tr>
<tr>
<td>34B series models</td>
<td>They are suitable for personal, academic, and commercial (particularly for small and medium-sized enterprises) purposes. It&#39;s a cost-effective solution that&#39;s affordable and equipped with emergent ability.</td>
<td>3T</td>
</tr>
</tbody></table>


- For chat models
  
  <details style="display: inline;"><summary>For chat model limitations, see the explanations below. ⬇️</summary>
   <ul>
    <br>The released chat model has undergone exclusive training using Supervised Fine-Tuning (SFT). Compared to other standard chat models, our model produces more diverse responses, making it suitable for various downstream tasks, such as creative scenarios. Furthermore, this diversity is expected to enhance the likelihood of generating higher quality responses, which will be advantageous for subsequent Reinforcement Learning (RL) training.

    <br>However, this higher diversity might amplify certain existing issues, including:
      <li>Hallucination: This refers to the model generating factually incorrect or nonsensical information. With the model's responses being more varied, there's a higher chance of hallucination that are not based on accurate data or logical reasoning.</li>
      <li>Non-determinism in re-generation: When attempting to regenerate or sample responses, inconsistencies in the outcomes may occur. The increased diversity can lead to varying results even under similar input conditions.</li>
      <li>Cumulative Error: This occurs when errors in the model's responses compound over time. As the model generates more diverse responses, the likelihood of small inaccuracies building up into larger errors increases, especially in complex tasks like extended reasoning, mathematical problem-solving, etc.</li>
      <li>To achieve more coherent and consistent responses, it is advisable to adjust generation configuration parameters such as temperature, top_p, or top_k. These adjustments can help in the balance between creativity and coherence in the model's outputs.</li>
  </ul>
  </details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>


# How to use Yi?

- [Quick start](#quick-start)
  - [Choose your path](#choose-your-path)
  - [pip](#quick-start---pip)
  - [docker](#quick-start---docker)
  - [conda-lock](#quick-start---conda-lock)
  - [llama.cpp](#quick-start---llamacpp)
  - [Web demo](#web-demo)
- [Fine-tuning](#fine-tuning)
- [Quantization](#quantization)
- [Deployment](#deployment)
- [FAQ](#faq)
- [Learning hub](#learning-hub)

## Quick start

Getting up and running with Yi models is simple with multiple choices available. 

### Choose your path

Select one of the following paths to begin your journey with Yi!

 ![Quick start - Choose your path](https://github.com/01-ai/Yi/blob/main/assets/img/quick_start_path.png?raw=true)

#### 🎯 Deploy Yi locally

If you prefer to deploy Yi models locally, 

  - 🙋‍♀️ and you have **sufficient** resources (for example, NVIDIA A800 80GB), you can choose one of the following methods:
    - [pip](#quick-start---pip)
    - [Docker](#quick-start---docker)
    - [conda-lock](#quick-start---conda-lock)

  - 🙋‍♀️ and you have **limited** resources (for example, a MacBook Pro), you can use [llama.cpp](#quick-start---llamacpp).

#### 🎯 Not to deploy Yi locally

If you prefer not to deploy Yi models locally, you can explore Yi's capabilities using any of the following options.

##### 🙋‍♀️ Run Yi with APIs

If you want to explore more features of Yi, you can adopt one of these methods:

- Yi APIs (Yi official)
  - [Early access has been granted](https://x.com/01AI_Yi/status/1735728934560600536?s=20) to some applicants. Stay tuned for the next round of access!

- [Yi APIs](https://replicate.com/01-ai/yi-34b-chat/api?tab=nodejs) (Replicate)

##### 🙋‍♀️ Run Yi in playground

If you want to chat with Yi with more customizable options (e.g., system prompt, temperature, repetition penalty, etc.), you can try one of the following options:

  - [Yi-34B-Chat-Playground](https://platform.lingyiwanwu.com/prompt/playground) (Yi official)
    - Access is available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)).
  
  - [Yi-34B-Chat-Playground](https://replicate.com/01-ai/yi-34b-chat) (Replicate) 

##### 🙋‍♀️ Chat with Yi

 If you want to chat with Yi, you can use one of these online services, which offer a similar user experience:

- [Yi-34B-Chat](https://huggingface.co/spaces/01-ai/Yi-34B-Chat) (Yi official on Hugging Face)
  - No registration is required.

- [Yi-34B-Chat](https://platform.lingyiwanwu.com/) (Yi official beta)
  - Access is available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)).

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - pip 

This tutorial guides you through every step of running **Yi-34B-Chat locally on an A800 (80G)** and then performing inference.

#### Step 0: Prerequisites

- Make sure Python 3.10 or a later version is installed.

- If you want to run other Yi models, see [software and hardware requirements](#deployment).

#### Step 1: Prepare your environment 

To set up the environment and install the required packages, execute the following command.

```bash
git clone https://github.com/01-ai/Yi.git
cd yi
pip install -r requirements.txt
```

#### Step 2: Download the Yi model

You can download the weights and tokenizer of Yi models from the following sources:

- [Hugging Face](https://huggingface.co/01-ai)
- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

#### Step 3: Perform inference

You can perform inference with Yi chat or base models as below.

##### Perform inference with Yi chat model

1. Create a file named  `quick_start.py` and copy the following content to it.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '<your-model-path>'

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

2. Run `quick_start.py`.

    ```bash
    python quick_start.py
    ```

    Then you can see an output similar to the one below. 🥳

    ```bash
    Hello! How can I assist you today?
    ```

##### Perform inference with Yi base model

- Yi-34B

  The steps are similar to [pip - Perform inference with Yi chat model](#perform-inference-with-yi-chat-model).

  You can use the existing file [`text_generation.py`](https://github.com/01-ai/Yi/tree/main/demo).

  ```bash
  python demo/text_generation.py  --model <your-model-path>
  ```

  Then you can see an output similar to the one below. 🥳

  <details>

  <summary>Output. ⬇️ </summary>

  <br>

  **Prompt**: Let me tell you an interesting story about cat Tom and mouse Jerry,

  **Generation**: Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

  </details>

- Yi-9B
  
  Input

  ```bash
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  MODEL_DIR = "01-ai/Yi-9B"
  model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
  
  input_text = "# write the quick sort algorithm"
  inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
  outputs = model.generate(**inputs, max_length=256)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  ```

  Output

  ```bash
  # write the quick sort algorithm
  def quick_sort(arr):
      if len(arr) <= 1:
          return arr
      pivot = arr[len(arr) // 2]
      left = [x for x in arr if x < pivot]
      middle = [x for x in arr if x == pivot]
      right = [x for x in arr if x > pivot]
      return quick_sort(left) + middle + quick_sort(right)
  
  # test the quick sort algorithm
  print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
  ```


<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - Docker 
<details>
<summary> Run Yi-34B-chat locally with Docker: a step-by-step guide. ⬇️</summary> 
<br>This tutorial guides you through every step of running <strong>Yi-34B-Chat on an A800 GPU</strong> or <strong>4*4090</strong> locally and then performing inference.
 <h4>Step 0: Prerequisites</h4>
<p>Make sure you've installed <a href="https://docs.docker.com/engine/install/?open_in_browser=true">Docker</a> and <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a>.</p>

<h4> Step 1: Start Docker </h4>
<pre><code>docker run -it --gpus all \
-v &lt;your-model-path&gt;: /models
ghcr.io/01-ai/yi:latest
</code></pre>
<p>Alternatively, you can pull the Yi Docker image from <code>registry.lingyiwanwu.com/ci/01-ai/yi:latest</code>.</p>

<h4>Step 2: Perform inference</h4>
    <p>You can perform inference with Yi chat or base models as below.</p>

<h5>Perform inference with Yi chat model</h5>
    <p>The steps are similar to <a href="#perform-inference-with-yi-chat-model">pip - Perform inference with Yi chat model</a>.</p>
    <p><strong>Note</strong> that the only difference is to set <code>model_path = '&lt;your-model-mount-path&gt;'</code> instead of <code>model_path = '&lt;your-model-path&gt;'</code>.</p>
<h5>Perform inference with Yi base model</h5>
    <p>The steps are similar to <a href="#perform-inference-with-yi-base-model">pip - Perform inference with Yi base model</a>.</p>
    <p><strong>Note</strong> that the only difference is to set <code>--model &lt;your-model-mount-path&gt;'</code> instead of <code>model &lt;your-model-path&gt;</code>.</p>
</details>

### Quick start - conda-lock

<details>
<summary>You can use <code><a href="https://github.com/conda/conda-lock">conda-lock</a></code> to generate fully reproducible lock files for conda environments. ⬇️</summary>
<br>
You can refer to <a href="https://github.com/01-ai/Yi/blob/ebba23451d780f35e74a780987ad377553134f68/conda-lock.yml">conda-lock.yml</a>  for the exact versions of the dependencies. Additionally, you can utilize <code><a href="https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html">micromamba</a></code> for installing these dependencies.
<br>
To install the dependencies, follow these steps:

1. Install micromamba by following the instructions available <a href="https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html">here</a>.

2. Execute <code>micromamba install -y -n yi -f conda-lock.yml</code> to create a conda environment named <code>yi</code> and install the necessary dependencies.
</details>


### Quick start - llama.cpp
<a href="https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md">The following tutorial </a> will guide you through every step of running a quantized model (<a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a>) locally and then performing inference.
<details>
<summary> Run Yi-chat-6B-2bits locally with llama.cpp: a step-by-step guide. ⬇️</summary> 
<br><a href="https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md">This tutorial</a> guides you through every step of running a quantized model (<a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a>) locally and then performing inference.</p>

- [Step 0: Prerequisites](#step-0-prerequisites)
- [Step 1: Download llama.cpp](#step-1-download-llamacpp)
- [Step 2: Download Yi model](#step-2-download-yi-model)
- [Step 3: Perform inference](#step-3-perform-inference)

#### Step 0: Prerequisites 

- This tutorial assumes you use a MacBook Pro with 16GB of memory and an Apple M2 Pro chip.
  
- Make sure [`git-lfs`](https://git-lfs.com/) is installed on your machine.
  
#### Step 1: Download `llama.cpp`

To clone the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) repository, run the following command.

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

#### Step 2: Download Yi model

2.1 To clone [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) with just pointers, run the following command.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

2.2 To download a quantized Yi model ([yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf)), run the following command.

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

#### Step 3: Perform inference

To perform inference with the Yi model, you can use one of the following methods.

- [Method 1: Perform inference in terminal](#method-1-perform-inference-in-terminal)
  
- [Method 2: Perform inference in web](#method-2-perform-inference-in-web)

##### Method 1: Perform inference in terminal

To compile `llama.cpp` using 4 threads and then conduct inference, navigate to the `llama.cpp` directory, and run the following command.

> ##### Tips
> 
> - Replace `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` with the actual path of your model.
>
> - By default, the model operates in completion mode.
> 
> - For additional output customization options (for example, system prompt, temperature, repetition penalty, etc.), run `./main -h` to check detailed descriptions and usage.

```bash
make -j4 && ./main -m /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf -p "How do you feed your pet fox? Please answer this question in 6 simple steps:\nStep 1:" -n 384 -e

...

How do you feed your pet fox? Please answer this question in 6 simple steps:

Step 1: Select the appropriate food for your pet fox. You should choose high-quality, balanced prey items that are suitable for their unique dietary needs. These could include live or frozen mice, rats, pigeons, or other small mammals, as well as fresh fruits and vegetables.

Step 2: Feed your pet fox once or twice a day, depending on the species and its individual preferences. Always ensure that they have access to fresh water throughout the day.

Step 3: Provide an appropriate environment for your pet fox. Ensure it has a comfortable place to rest, plenty of space to move around, and opportunities to play and exercise.

Step 4: Socialize your pet with other animals if possible. Interactions with other creatures can help them develop social skills and prevent boredom or stress.

Step 5: Regularly check for signs of illness or discomfort in your fox. Be prepared to provide veterinary care as needed, especially for common issues such as parasites, dental health problems, or infections.

Step 6: Educate yourself about the needs of your pet fox and be aware of any potential risks or concerns that could affect their well-being. Regularly consult with a veterinarian to ensure you are providing the best care.

...

```

Now you have successfully asked a question to the Yi model and got an answer! 🥳

##### Method 2: Perform inference in web

1. To initialize a lightweight and swift chatbot, run the following command.

    ```bash
    cd llama.cpp
    ./server --ctx-size 2048 --host 0.0.0.0 --n-gpu-layers 64 --model /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf
    ```

    Then you can get an output like this:


    ```bash
    ...
    
    llama_new_context_with_model: n_ctx      = 2048
    llama_new_context_with_model: freq_base  = 5000000.0
    llama_new_context_with_model: freq_scale = 1
    ggml_metal_init: allocating
    ggml_metal_init: found device: Apple M2 Pro
    ggml_metal_init: picking default device: Apple M2 Pro
    ggml_metal_init: ggml.metallib not found, loading from source
    ggml_metal_init: GGML_METAL_PATH_RESOURCES = nil
    ggml_metal_init: loading '/Users/yu/llama.cpp/ggml-metal.metal'
    ggml_metal_init: GPU name:   Apple M2 Pro
    ggml_metal_init: GPU family: MTLGPUFamilyApple8 (1008)
    ggml_metal_init: hasUnifiedMemory              = true
    ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
    ggml_metal_init: maxTransferRate               = built-in GPU
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =   128.00 MiB, ( 2629.44 / 10922.67)
    llama_new_context_with_model: KV self size  =  128.00 MiB, K (f16):   64.00 MiB, V (f16):   64.00 MiB
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =     0.02 MiB, ( 2629.45 / 10922.67)
    llama_build_graph: non-view tensors processed: 676/676
    llama_new_context_with_model: compute buffer total size = 159.19 MiB
    ggml_backend_metal_buffer_type_alloc_buffer: allocated buffer, size =   156.02 MiB, ( 2785.45 / 10922.67)
    Available slots:
    -> Slot 0 - max context: 2048
    
    llama server listening at http://0.0.0.0:8080
    ```

2. To access the chatbot interface, open your web browser and enter `http://0.0.0.0:8080` into the address bar. 
   
    ![Yi model chatbot interface - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp1.png?raw=true)


3. Enter a question, such as "How do you feed your pet fox? Please answer this question in 6 simple steps" into the prompt window, and you will receive a corresponding answer.

    ![Ask a question to Yi model - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp2.png?raw=true)

</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Web demo

You can build a web UI demo for Yi **chat** models (note that Yi base models are not supported in this senario).

[Step 1: Prepare your environment](#step-1-prepare-your-environment). 

[Step 2: Download the Yi model](#step-2-download-the-yi-model).

Step 3. To start a web service locally, run the following command.

```bash
python demo/web_demo.py -c <your-model-path>
```

You can access the web UI by entering the address provided in the console into your browser. 

 ![Quick start - web demo](https://github.com/01-ai/Yi/blob/main/assets/img/yi_34b_chat_web_demo.gif?raw=true)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Fine-tuning

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

Once finished, you can compare the finetuned model and the base model with the following command:

```bash
bash finetune/scripts/run_eval.sh
```
<details style="display: inline;"><summary>For advanced usage (like fine-tuning based on your custom data), see the explanations below. ⬇️ </summary> <ul>

### Finetune code for Yi 6B and 34B

#### Preparation

##### From Image

By default, we use a small dataset from [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) to finetune the base model.
You can also prepare your customized dataset in the following `jsonl` format:

```json
{ "prompt": "Human: Who are you? Assistant:", "chosen": "I'm Yi." }
```

And then mount them in the container to replace the default ones:

```bash
docker run -it \
    -v /path/to/save/finetuned/model/:/finetuned-model \
    -v /path/to/train.jsonl:/yi/finetune/data/train.json \
    -v /path/to/eval.jsonl:/yi/finetune/data/eval.json \
    ghcr.io/01-ai/yi:latest \
    bash finetune/scripts/run_sft_Yi_6b.sh
```

##### From Local Server

Make sure you have conda. If not, use

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

Then, create a conda env:

```bash
conda create -n dev_env python=3.10 -y
conda activate dev_env
pip install torch==2.0.1 deepspeed==0.10 tensorboard transformers datasets sentencepiece accelerate ray==2.7
```

#### Hardware Setup

For the Yi-6B model, a node with 4 GPUs, each with GPU memory larger than 60GB, is recommended.

For the Yi-34B model, because the usage of the zero-offload technique consumes a lot of CPU memory, please be careful to limit the number of GPUs in the 34B finetune training. Please use CUDA_VISIBLE_DEVICES to limit the number of GPUs (as shown in scripts/run_sft_Yi_34b.sh).

A typical hardware setup for finetuning the 34B model is a node with 8 GPUs (limited to 4 in running by CUDA_VISIBLE_DEVICES=0,1,2,3), each with GPU memory larger than 80GB, and total CPU memory larger than 900GB.

#### Quick Start

Download a LLM-base model to MODEL_PATH (6B and 34B). A typical folder of models is like:

```bash
|-- $MODEL_PATH
|   |-- config.json
|   |-- pytorch_model-00001-of-00002.bin
|   |-- pytorch_model-00002-of-00002.bin
|   |-- pytorch_model.bin.index.json
|   |-- tokenizer_config.json
|   |-- tokenizer.model
|   |-- ...
```

Download a dataset from huggingface to local storage DATA_PATH, e.g. Dahoas/rm-static.

```bash
|-- $DATA_PATH
|   |-- data
|   |   |-- train-00000-of-00001-2a1df75c6bce91ab.parquet
|   |   |-- test-00000-of-00001-8c7c51afc6d45980.parquet
|   |-- dataset_infos.json
|   |-- README.md
```

`finetune/yi_example_dataset` has example datasets, which are modified from [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)

```bash
|-- $DATA_PATH
    |--data
        |-- train.jsonl
        |-- eval.jsonl
```

`cd` into the scripts folder, copy and paste the script, and run. For example:

```bash
cd finetune/scripts

bash run_sft_Yi_6b.sh
```

For the Yi-6B base model, setting training_debug_steps=20 and num_train_epochs=4 can output a chat model, which takes about 20 minutes.

For the Yi-34B base model, it takes a relatively long time for initialization. Please be patient.

#### Evaluation

```bash
cd finetune/scripts

bash run_eval.sh
```

Then you'll see the answer from both the base model and the finetuned model.
</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quantization

#### GPT-Q
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

<details style="display: inline;"><summary>For details, see the explanations below. ⬇️</summary> <ul>

#### GPT-Q quantization

[GPT-Q](https://github.com/IST-DASLab/gptq) is a PTQ (Post-Training Quantization)
method. It saves memory and provides potential speedups while retaining the accuracy
of the model. 

Yi models can be GPT-Q quantized without a lot of efforts. 
We provide a step-by-step tutorial below.

To run GPT-Q, we will use [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) and
[exllama](https://github.com/turboderp/exllama).
And the huggingface transformers has integrated optimum and auto-gptq to perform
GPTQ quantization on language models.

##### Do Quantization

The `quant_autogptq.py` script is provided for you to perform GPT-Q quantization:

```bash
python quant_autogptq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### Run Quantized Model

You can run a quantized model using the `eval_quantized_model.py`:

```bash
python eval_quantized_model.py --model /quantized_model --trust_remote_code
```
</ul>
</details>

#### AWQ

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
<details style="display: inline;"><summary>For details, see the explanations below. ⬇️</summary> <ul>

#### AWQ quantization

[AWQ](https://github.com/mit-han-lab/llm-awq) is a PTQ (Post-Training Quantization)
method. It's an efficient and accurate low-bit weight quantization (INT3/4) for LLMs.

Yi models can be AWQ quantized without a lot of efforts. 
We provide a step-by-step tutorial below.

To run AWQ, we will use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

##### Do Quantization

The `quant_autoawq.py` script is provided for you to perform AWQ quantization:

```bash
python quant_autoawq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### Run Quantized Model

You can run a quantized model using the `eval_quantized_model.py`:

```bash
python eval_quantized_model.py --model /quantized_model --trust_remote_code
```


</ul>
</details>
<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Deployment

If you want to deploy Yi models, make sure you meet the software and hardware requirements. 

#### Software requirements

Before using Yi quantized models, make sure you've installed the correct software listed below.

| Model | Software
|---|---
Yi 4-bit quantized models | [AWQ and CUDA](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install-from-pypi)
Yi 8-bit quantized models |  [GPTQ and CUDA](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#quick-installation)

#### Hardware requirements

Before deploying Yi in your environment, make sure your hardware meets the following requirements.

##### Chat models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br>  1 x A10 (24 GB)  <br> 1 x A30 (24 GB)              |
| Yi-6B-Chat-4bits     | 4 GB          | 1 x RTX 3060 (12 GB)<br> 1 x RTX 4060 (8 GB)                   |
| Yi-6B-Chat-8bits     | 8 GB          | 1 x RTX 3070 (8 GB) <br> 1 x RTX 4060 (8 GB)                   |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 (24 GB)<br> 1 x A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)  <br> 1 x A100 (40 GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090 (24 GB) <br> 2 x RTX 4090 (24 GB)<br> 1 x A800  (40 GB) |

Below are detailed minimum VRAM requirements under different batch use cases.

|  Model                  | batch=1 | batch=4 | batch=16 | batch=32 |
| ----------------------- | ------- | ------- | -------- | -------- |
| Yi-6B-Chat              | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-4bits  | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-6B-Chat-8bits  | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-34B-Chat       | 65 GB   | 68 GB   | 76 GB    | > 80 GB   |
| Yi-34B-Chat-4bits | 19 GB   | 20 GB   | 30 GB    | 40 GB    |
| Yi-34B-Chat-8bits | 35 GB   | 37 GB   | 46 GB    | 58 GB    |

##### Base models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|----------------------|--------------|:-------------------------------------:|
| Yi-6B                | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)                |
| Yi-6B-200K           | 50 GB         | 1 x A800 (80 GB)                            |
| Yi-9B                | 20 GB         | 1 x RTX 4090 (24 GB)                           |
| Yi-34B               | 72 GB         | 4 x RTX 4090 (24 GB) <br> 1 x A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### FAQ
<details>
<summary> If you have any questions while using the Yi series models, the answers provided below could serve as a helpful reference for you. ⬇️</summary> 
<br> 

#### 💡Fine-tuning
- <strong>Base model or Chat model - which to fine-tune?</strong>
  <br>The choice of pre-trained language model for fine-tuning hinges on the computational resources you have at your disposal and the particular demands of your task.
    - If you are working with a substantial volume of fine-tuning data (say, over 10,000 samples), the Base model could be your go-to choice.
    - On the other hand, if your fine-tuning data is not quite as extensive, opting for the Chat model might be a more fitting choice.
    - It is generally advisable to fine-tune both the Base and Chat models, compare their performance, and then pick the model that best aligns with your specific requirements.
- <strong>Yi-34B versus Yi-34B-Chat for full-scale fine-tuning - what is the difference?</strong>
  <br>
  The key distinction between full-scale fine-tuning on `Yi-34B`and `Yi-34B-Chat` comes down to the fine-tuning approach and outcomes.
    - Yi-34B-Chat employs a Special Fine-Tuning (SFT) method, resulting in responses that mirror human conversation style more closely.
    - The Base model's fine-tuning is more versatile, with a relatively high performance potential.
    - If you are confident in the quality of your data, fine-tuning with `Yi-34B` could be your go-to.
    - If you are aiming for model-generated responses that better mimic human conversational style, or if you have doubts about your data quality, `Yi-34B-Chat` might be your best bet.

#### 💡Quantization
- <strong>Quantized model versus original model - what is the performance gap?</strong>
    - The performance variance is largely contingent on the quantization method employed and the specific use cases of these models. For instance, when it comes to models provided by the AWQ official, from a Benchmark standpoint, quantization might result in a minor performance drop of a few percentage points.
    - Subjectively speaking, in situations like logical reasoning, even a 1% performance shift could impact the accuracy of the output results.
    
#### 💡General
- <strong>Where can I source fine-tuning question answering datasets?</strong>
    - You can find fine-tuning question answering datasets on platforms like Hugging Face, with datasets like [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) readily available. 
    - Additionally, Github offers fine-tuning frameworks, such as [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which integrates pre-made datasets.

- <strong>What is the GPU memory requirement for fine-tuning Yi-34B FP16?</strong>
  <br>
  The GPU memory needed for fine-tuning 34B FP16 hinges on the specific fine-tuning method employed. For full parameter fine-tuning, you'll need 8 GPUs each with 80 GB; however, more economical solutions like Lora require less. For more details, check out [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Also, consider using BF16 instead of FP16 for fine-tuning to optimize performance.

- <strong>Are there any third-party platforms that support chat functionality for the Yi-34b-200k model?</strong>
  <br>
  If you're looking for third-party Chats, options include [fireworks.ai](https://fireworks.ai/login?callbackURL=https://fireworks.ai/models/fireworks/yi-34b-chat).
  </details>

### Learning hub

<details>
<summary> If you want to learn Yi, you can find a wealth of helpful educational resources here. ⬇️</summary> 
<br> 

Welcome to the Yi learning hub! 

Whether you're a seasoned developer or a newcomer, you can find a wealth of helpful educational resources to enhance your understanding and skills with Yi models, including insightful blog posts, comprehensive video tutorials, hands-on guides, and more.  

The content you find here has been generously contributed by knowledgeable Yi experts and passionate enthusiasts. We extend our heartfelt gratitude for your invaluable contributions! 

At the same time, we also warmly invite you to join our collaborative effort by contributing to Yi. If you have already made contributions to Yi, please don't hesitate to showcase your remarkable work in the table below.

With all these resources at your fingertips, you're ready to start your exciting journey with Yi. Happy learning! 🥳

#### Tutorials

##### Blog tutorials

| Deliverable                                                  | Date       | Author                                                       |
| ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| [使用 Dify、Meilisearch、零一万物模型实现最简单的 RAG   应用（三）：AI 电影推荐](https://mp.weixin.qq.com/s/Ri2ap9_5EMzdfiBhSSL_MQ) | 2024-05-20 | [苏洋](https://github.com/soulteary)                         |
| [使用autodl服务器，在A40显卡上运行，   Yi-34B-Chat-int4模型，并使用vllm优化加速，显存占用42G，速度18 words-s](https://blog.csdn.net/freewebsys/article/details/134698597?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-17-134698597-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-05-20 | [fly-iot](https://gitee.com/fly-iot)                         |
| [Yi-VL   最佳实践](https://modelscope.cn/docs/yi-vl最佳实践) | 2024-05-20 | [ModelScope](https://github.com/modelscope)                  |
| [一键运行零一万物新鲜出炉Yi-1.5-9B-Chat大模型](https://mp.weixin.qq.com/s/ntMs2G_XdWeM3I6RUOBJrA) | 2024-05-13 | [Second State](https://github.com/second-state)              |
| [零一万物开源Yi-1.5系列大模型](https://mp.weixin.qq.com/s/d-ogq4hcFbsuL348ExJxpA) | 2024-05-13 | [刘聪](https://github.com/liucongg)                          |
| [零一万物Yi-1.5系列模型发布并开源！ 34B-9B-6B   多尺寸，魔搭社区推理微调最佳实践教程来啦！](https://mp.weixin.qq.com/s/3wD-0dCgXB646r720o8JAg) | 2024-05-13 | [ModelScope](https://github.com/modelscope)                  |
| [Yi-34B   本地部署简单测试](https://blog.csdn.net/arkohut/article/details/135331469?ops_request_misc=%7B%22request%5Fid%22%3A%22171636390616800185813639%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636390616800185813639&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-10-135331469-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-05-13 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [驾辰龙跨Llama持Wasm，玩转Yi模型迎新春过大年（上）](https://blog.csdn.net/weixin_53443275/article/details/136091398?ops_request_misc=%7B%22request%5Fid%22%3A%22171636390616800185813639%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636390616800185813639&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-5-136091398-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-05-13 | [Words  worth](https://blog.csdn.net/weixin_53443275?type=blog) |
| [驾辰龙跨Llama持Wasm，玩转Yi模型迎新春过大年（下篇）](https://blog.csdn.net/weixin_53443275/article/details/136096309) | 2024-05-13 | [Words  worth](https://blog.csdn.net/weixin_53443275?type=blog) |
| [Ollama新增两个命令，开始支持零一万物Yi-1.5系列模型](https://mp.weixin.qq.com/s/bBgzGJvUqIohodcy9U-pFw) | 2024-05-13 | AI工程师笔记                                                 |
| [使用零一万物 200K 模型和 Dify 快速搭建模型应用](https://zhuanlan.zhihu.com/p/686774859) | 2024-05-13 | [苏洋](https://github.com/soulteary)                         |
| [(持更) 零一万物模型折腾笔记：社区 Yi-34B 微调模型使用](https://zhuanlan.zhihu.com/p/671549900) | 2024-05-13 | [苏洋](https://github.com/soulteary)                         |
| [Python+ERNIE-4.0-8K-Yi-34B-Chat大模型初探](https://mp.weixin.qq.com/s/WaygSfn5T8ZPB1mPdGADEQ) | 2024-05-11 | 江湖评谈                                                     |
| [技术布道   Vue及Python调用零一万物模型和Prompt模板（通过百度千帆大模型平台）](https://blog.csdn.net/ucloud2012/article/details/137187469) | 2024-05-11 | [MumuLab](https://blog.csdn.net/ucloud2012?type=blog)        |
| [多模态大模型Yi-VL-plus体验 效果很棒](https://zhuanlan.zhihu.com/p/694736111) | 2024-04-27 | [大家好我是爱因](https://www.zhihu.com/people/iamein)        |
| [使用autodl服务器，两个3090显卡上运行，   Yi-34B-Chat-int4模型，并使用vllm优化加速，显存占用42G，速度23 words-s](https://blog.csdn.net/freewebsys/article/details/134725765?ops_request_misc=%7B%22request%5Fid%22%3A%22171636356716800211598950%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636356716800211598950&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-9-134725765-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-04-27 | [fly-iot](https://gitee.com/fly-iot)                         |
| [Getting Started with Yi-1.5-9B-Chat](https://www.secondstate.io/articles/yi-1.5-9b-chat/) | 2024-04-27 | [Second State](https://github.com/second-state)              |
| [基于零一万物yi-vl-plus大模型简单几步就能批量生成Anki图片笔记](https://mp.weixin.qq.com/s/_ea6g0pzzeO4WyYtuWycWQ) | 2024-04-24 | [正经人王同学](https://github.com/zjrwtx)                    |
| [【AI开发：语言】一、Yi-34B超大模型本地部署CPU和GPU版](https://blog.csdn.net/alarey/article/details/137769471?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-16-137769471-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-04-21 | [My的梦想已实现](https://blog.csdn.net/alarey?type=blog)     |
| [【Yi-34B-Chat-Int4】使用4个2080Ti显卡11G版本，运行Yi-34B模型，5年前老显卡是支持的，可以正常运行，速度   21 words-s，vllm要求算力在7以上的显卡就可以](https://blog.csdn.net/freewebsys/article/details/134754086) | 2024-03-22 | [fly-iot](https://gitee.com/fly-iot)                         |
| [零一万物大模型部署+微调总结](https://blog.csdn.net/v_wus/article/details/135704126?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-18-135704126-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-03-22 | [v_wus](https://blog.csdn.net/v_wus?type=blog)               |
| [零一万物Yi大模型vllm推理时Yi-34B或Yi-6bchat重复输出的解决方案](https://blog.csdn.net/qq_39667443/article/details/136028776?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-6-136028776-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-03-02 | [郝铠锋](https://blog.csdn.net/qq_39667443?type=blog)        |
| [Yi-34B微调训练](https://blog.csdn.net/lsjlnd/article/details/135336984?ops_request_misc=%7B%22request%5Fid%22%3A%22171636343416800188513953%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636343416800188513953&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-12-135336984-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-03-02 | [lsjlnd](https://blog.csdn.net/lsjlnd?type=blog)             |
| [实测零一万物Yi-VL多模态语言模型：能准确“识图吃瓜”](https://mp.weixin.qq.com/s/fu4O9XvJ03JhimsEyI-SsQ) | 2024-02-02 | [苏洋](https://github.com/soulteary)                         |
| [零一万物开源Yi-VL多模态大模型，魔搭社区推理&微调最佳实践来啦！](https://zhuanlan.zhihu.com/p/680098411) | 2024-01-26 | [ModelScope](https://github.com/modelscope)                  |
| [单卡 3 小时训练 Yi-6B 大模型 Agent：基于 Llama   Factory 实战](https://zhuanlan.zhihu.com/p/678989191) | 2024-01-22 | [郑耀威](https://github.com/hiyouga)                         |
| [零一科技Yi-34B   Chat大模型环境搭建&推理](https://blog.csdn.net/zzq1989_/article/details/135597181?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-8-135597181-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-01-15 | [要养家的程序员](https://blog.csdn.net/zzq1989_?type=blog)   |
| [基于LLaMA   Factory，单卡3小时训练专属大模型 Agent](https://blog.csdn.net/m0_59596990/article/details/135760285?ops_request_misc=%7B%22request%5Fid%22%3A%22171636343416800188513953%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636343416800188513953&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-10-135760285-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-01-15 | [机器学习社区](https://blog.csdn.net/m0_59596990?type=blog)  |
| [双卡   3080ti 部署 Yi-34B 大模型 - Gradio + vLLM 踩坑全记录](https://blog.csdn.net/arkohut/article/details/135321242?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-10-135321242-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-01-02 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [【大模型部署实践-3】3个能在3090上跑起来的4bits量化Chat模型（baichuan2-13b、InternLM-20b、Yi-34b）](https://blog.csdn.net/qq_40302568/article/details/135040985?ops_request_misc=%7B%22request%5Fid%22%3A%22171636168816800227489911%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636168816800227489911&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-30-135040985-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2024-01-02 | [aq_Seabiscuit](https://blog.csdn.net/qq_40302568?type=blog) |
| [只需 24G   显存，用 vllm 跑起来 Yi-34B 中英双语大模型](https://blog.csdn.net/arkohut/article/details/135274973) | 2023-12-28 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [零一万物模型官方   Yi-34B 模型本地离线运行部署使用笔记（物理机和docker两种部署方式），200K 超长文本内容，34B 干翻一众 70B   模型，打榜分数那么高，这模型到底行不行？](https://blog.csdn.net/u014374009/article/details/136327696) | 2023-12-28 | [代码讲故事](https://blog.csdn.net/u014374009?type=blog)     |
| [LLM -   大模型速递之 Yi-34B 入门与 LoRA 微调](https://blog.csdn.net/BIT_666/article/details/134990402) | 2023-12-18 | [BIT_666](https://bitddd.blog.csdn.net/?type=blog)           |
| [通过vllm框架进行大模型推理](https://blog.csdn.net/weixin_45920955/article/details/135300561?ops_request_misc=%7B%22request%5Fid%22%3A%22171636343416800188513953%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=171636343416800188513953&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~times_rank-13-135300561-null-null.nonecase&utm_term=Yi大模型&spm=1018.2226.3001.4450) | 2023-12-18 | [土山炮](https://blog.csdn.net/weixin_45920955?type=blog)    |
| [CPU 混合推理，非常见大模型量化方案：“二三五六” 位量化方案](https://zhuanlan.zhihu.com/p/671698216) | 2023-12-12 | [苏洋](https://github.com/soulteary)                         |
| [零一万物模型折腾笔记：官方 Yi-34B 模型基础使用](https://zhuanlan.zhihu.com/p/671387298) | 2023-12-10 | [苏洋](https://github.com/soulteary)                         |
| [Running Yi-34B-Chat locally using LlamaEdge](https://www.secondstate.io/articles/yi-34b/) | 2023-11-30 | [Second State](https://github.com/second-state)              |
| [本地运行零一万物 34B 大模型，使用 Llama.cpp &   21G 显存](https://zhuanlan.zhihu.com/p/668921042) | 2023-11-26 | [苏洋](https://github.com/soulteary)                         |

##### GitHub Project

| Deliverable                                                  | Date       | Author                                      |
| ------------------------------------------------------------ | ---------- | ------------------------------------------- |
| [yi-openai-proxy](https://github.com/soulteary/yi-openai-proxy) | 2024-05-11 | [苏洋](https://github.com/soulteary)        |
| [基于零一万物 Yi 模型和 B 站构建大语言模型高质量训练数据集](https://github.com/zjrwtx/bilibiliQA_databuilder) | 2024-04-29 | [正经人王同学](https://github.com/zjrwtx)   |
| [基于视频网站和零一万物大模型构建大语言模型高质量训练数据集](https://github.com/zjrwtx/VideoQA_databuilder) | 2024-04-25 | [正经人王同学](https://github.com/zjrwtx)   |
| [基于零一万物yi-34b-chat-200k输入任意文章地址，点击按钮即可生成无广告或推广内容的简要笔记，并生成分享图给好友](https://github.com/zjrwtx/open_summary) | 2024-04-24 | [正经人王同学](https://github.com/zjrwtx)   |
| [Food-GPT-Yi-model](https://github.com/ThisisHubert/FoodGPT-Yi-model) | 2024-04-21 | [Hubert S](https://github.com/ThisisHubert) |

##### Video tutorials

| Deliverable                                                  | Date       | Author                                                       |
| ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| [Run dolphin-2.2-yi-34b on IoT Devices](https://www.youtube.com/watch?v=NJ89T5mO25Y) | 2023-11-30 | [Second State](https://github.com/second-state)              |
| [只需 24G 显存，用 vllm 跑起来 Yi-34B 中英双语大模型](https://www.bilibili.com/video/BV17t4y1f7Ee/) | 2023-12-28 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [Install Yi 34B Locally - Chinese English Bilingual LLM](https://www.youtube.com/watch?v=CVQvj4Wrh4w&t=476s) | 2023-11-05 | [Fahd Mirza](https://www.youtube.com/@fahdmirza)             |
| [Dolphin Yi 34b - Brand New Foundational Model TESTED](https://www.youtube.com/watch?v=On3Zuv27V3k&t=85s) | 2023-11-27 | [Matthew Berman](https://www.youtube.com/@matthew_berman)    |
| [Yi-VL-34B 多模态大模型 - 用两张 A40 显卡跑起来](https://www.bilibili.com/video/BV1Q5411y7AG/) | 2024-01-28 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [4060Ti 16G显卡安装零一万物最新开源的Yi-1.5版大语言模型](https://www.bilibili.com/video/BV16i421X7Jx/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-05-14 | [titan909](https://space.bilibili.com/526393761)             |
| [Yi-1.5: True Apache 2.0 Competitor to LLAMA-3](https://www.youtube.com/watch?v=KCDYrfWeTRc) | 2024-05-13 | [Prompt Engineering](https://www.youtube.com/@engineerprompt) |
| [Install Yi-1.5 Model Locally - Beats Llama 3 in Various Benchmarks](https://www.youtube.com/watch?v=Ba-G7Il0UkA) | 2024-05-13 | [Fahd Mirza](https://www.youtube.com/@fahdmirza)             |
| [how to install Ollama and run Yi 6B](https://www.youtube.com/watch?v=4Jnar7OUHqQ) | 2024-05-13 | [Ridaa Davids](https://www.youtube.com/@quantanovabusiness)  |
| [地表最强混合智能AI助手：llama3_70B+Yi_34B+Qwen1.5_110B](https://www.bilibili.com/video/BV1Xm411C7V1/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-05-04 | [朱扎特](https://space.bilibili.com/494512200?spm_id_from=333.788.0.0) |
| [ChatDoc学术论文辅助--基于Yi-34B和langchain进行PDF知识库问答](https://www.bilibili.com/video/BV11i421C7B5/?spm_id_from=333.999.0.0&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-05-03 | [朱扎特](https://space.bilibili.com/494512200?spm_id_from=333.788.0.0) |
| [基于Yi-34B的领域知识问答项目演示](https://www.bilibili.com/video/BV1zZ42177ZA/?spm_id_from=333.999.0.0&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-05-02 | [朱扎特](https://space.bilibili.com/494512200?spm_id_from=333.788.0.0) |
| [使用RTX4090+GaLore算法 全参微调Yi-6B大模型](https://www.bilibili.com/video/BV1ax4y1U7Ep/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-03-24 | [小工蚂创始人](https://space.bilibili.com/478674499?spm_id_from=333.788.0.0) |
| [无内容审查NSFW大语言模型Yi-34B-Chat蒸馏版测试,RolePlay,《天龙八部》马夫人康敏,本地GPU,CPU运行](https://www.youtube.com/watch?v=VL-W0TnLCns) | 2024-03-20 | [刘悦的技术博客](https://v3u.cn/)                            |
| [无内容审查NSFW大语言模型整合包,Yi-34B-Chat,本地CPU运行,角色扮演潘金莲](https://www.youtube.com/watch?v=rBvbgwz3oHM) | 2024-03-16 | [刘悦的技术博客](https://v3u.cn/)                            |
| [量化 Yi-34B-Chat 并在单卡 RTX 4090 使用 vLLM 部署](https://www.bilibili.com/video/BV1jx421y7xj/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-03-05 | [白鸽巢](https://space.bilibili.com/138938660?spm_id_from=333.788.0.0) |
| [Yi-VL-34B（5）：使用3个3090显卡24G版本，运行Yi-VL-34B模型，支持命令行和web界面方式，理解图片的内容转换成文字](https://www.bilibili.com/video/BV1BB421z7oA/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-27 | [fly-iot](https://gitee.com/fly-iot)                         |
| [Win环境KoboldCpp本地部署大语言模型进行各种角色扮演游戏](https://www.bilibili.com/video/BV14J4m1e77f/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-25 | [魚蟲蟲](https://space.bilibili.com/431981179?spm_id_from=333.788.0.0) |
| [无需显卡本地部署Yi-34B-Chat进行角色扮演游戏 P2](https://www.bilibili.com/video/BV19v421677y/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-23 | [魚蟲蟲](https://space.bilibili.com/431981179?spm_id_from=333.788.0.0) |
| [【wails】（2）：使用go-llama.cpp 运行 yi-01-6b大模型，使用本地CPU运行，速度还可以，等待下一版本更新](https://www.bilibili.com/video/BV194421F7Fy/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-20 | [fly-iot](https://gitee.com/fly-iot)                         |
| [【xinference】（6）：在autodl上，使用xinference部署yi-vl-chat和qwen-vl-chat模型，可以使用openai调用成功](https://www.bilibili.com/video/BV19Z421z7cv/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-06 | [fly-iot](https://gitee.com/fly-iot)                         |
| [无需显卡本地部署Yi-34B-Chat进行角色扮演游戏 P1](https://www.bilibili.com/video/BV1tU421o7Co/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-02-05 | [魚蟲蟲](https://space.bilibili.com/431981179?spm_id_from=333.788.0.0) |
| [2080Ti部署YI-34B大模型 xinference-oneapi-fastGPT本地知识库使用指南](https://www.bilibili.com/video/BV1hC411z7xu/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-30 | [小饭护法要转码](https://space.bilibili.com/39486865?spm_id_from=333.788.0.0) |
| [Best Story Writing AI Model - Install Yi 6B 200K Locally on Windows](https://www.youtube.com/watch?v=cZs2jRtl0bs) | 2024-01-22 | [Fahd Mirza](https://www.youtube.com/@fahdmirza)             |
| [Mac 本地运行大语言模型方法与常见问题指南（Yi 34B 模型+32 GB 内存测试）](https://www.bilibili.com/video/BV1VT4y1b7Th/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-21 | [小吴苹果机器人](https://space.bilibili.com/1732749682?spm_id_from=333.788.0.0) |
| [【Dify知识库】（11）：Dify0.4.9改造支持MySQL，成功接入yi-6b 做对话，本地使用fastchat启动，占8G显存，完成知识库配置](https://www.bilibili.com/video/BV1ia4y1y7JH/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-21 | [fly-iot](https://gitee.com/fly-iot)                         |
| [这位LLM先生有点暴躁,用的是YI-6B的某个量化版,#LLM #大语言模型 #暴躁老哥](https://www.youtube.com/watch?v=eahXJrdtQuc) | 2024-01-20 | [晓漫吧](https://www.youtube.com/@xiaomanba)                 |
| [大模型推理 NvLink 桥接器有用吗｜双卡 A6000 测试一下](https://www.bilibili.com/video/BV1AW4y1w7DC/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-17 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [大模型推理 A40 vs A6000 谁更强 - 对比 Yi-34B 的单、双卡推理性能](https://www.bilibili.com/video/BV1aK4y1z7GF/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-15 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [C-Eval 大语言模型评测基准- 用 LM Evaluation Harness + vLLM 跑起来](https://www.bilibili.com/video/BV1Yw411g7ZL/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-11 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [双显卡部署 Yi-34B 大模型 - vLLM + Gradio 踩坑记录](https://www.bilibili.com/video/BV1p94y1c7ak/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2024-01-01 | [漆妮妮](https://space.bilibili.com/1262370256)              |
| [手把手教学！使用 vLLM 快速部署 Yi-34B-Chat](https://www.bilibili.com/video/BV1ew41157Mk/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-26 | [白鸽巢](https://space.bilibili.com/138938660?spm_id_from=333.788.0.0) |
| [如何训练企业自己的大语言模型？Yi-6B LORA微调演示 #小工蚁](https://www.bilibili.com/video/BV1uc41117zz/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-21 | [小工蚂创始人](https://space.bilibili.com/478674499?spm_id_from=333.788.0.0) |
| [Yi-34B（4）：使用4个2080Ti显卡11G版本，运行Yi-34B模型，5年前老显卡是支持的，可以正常运行，速度 21 words/s](https://www.bilibili.com/video/BV1nj41157L3/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-02 | [fly-iot](https://gitee.com/fly-iot)                         |
| [使用autodl服务器，RTX 3090 * 3 显卡上运行， Yi-34B-Chat模型，显存占用60G](https://www.bilibili.com/video/BV1BM411R7ae/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-01 | [fly-iot](https://gitee.com/fly-iot)                         |
| [使用autodl服务器，两个3090显卡上运行， Yi-34B-Chat-int4模型，用vllm优化，增加 --num-gpu 2，速度23 words/s](https://www.bilibili.com/video/BV1Hu4y1L7BH/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-01 | [fly-iot](https://gitee.com/fly-iot)                         |
| [Yi大模型一键本地部署 技术小白玩转AI](https://www.bilibili.com/video/BV16H4y117md/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-12-01 | [技术小白玩转AI](https://space.bilibili.com/3546586137234288?spm_id_from=333.788.0.0) |
| [01.AI's Yi-6B: Overview and Fine-Tuning](https://www.youtube.com/watch?v=mye-UOkAliQ) | 2023-11-28 | [AI Makerspace](https://www.youtube.com/@AI-Makerspace)      |
| [Yi 34B Chat LLM outperforms Llama 70B](https://www.youtube.com/watch?v=RYtrF-R5jDc) | 2023-11-27 | [DLExplorer](https://www.youtube.com/@DLExplorers-lg7dt)     |
| [How to run open source models on mac Yi 34b on m3 Max](https://www.youtube.com/watch?v=GAo-dopkgjI) | 2023-11-26 | [TECHNO PREMIUM](https://www.youtube.com/@technopremium91)   |
| [Yi-34B - 200K - The BEST & NEW CONTEXT WINDOW KING ](https://www.youtube.com/watch?v=7WBojwwv5Qo) | 2023-11-24 | [Prompt Engineering](https://www.youtube.com/@engineerprompt) |
| [Yi 34B : The Rise of Powerful Mid-Sized Models - Base,200k & Chat](https://www.youtube.com/watch?v=bWCjwtu_tHs) | 2023-11-24 | [Sam Witteveen](https://www.youtube.com/@samwitteveenai)     |
| [在IoT设备运行破解版李开复大模型dolphin-2.2-yi-34b（还可作为私有OpenAI API服务器）](https://www.bilibili.com/video/BV1SQ4y18744/?spm_id_from=333.337.search-card.all.click&vd_source=ab85f93e294a2f6be11db57c29c6d706) | 2023-11-15 | [Second State](https://github.com/second-state)              |
| [Run dolphin-2.2-yi-34b on IoT Devices (Also works as a Private OpenAI API Server)](https://www.youtube.com/watch?v=NJ89T5mO25Y) | 2023-11-14 | [Second State](https://github.com/second-state)              |
| [How to Install Yi 34B 200K Llamafied on Windows Laptop](https://www.youtube.com/watch?v=enoha4K4HkQ) | 2023-11-11 | [Fahd Mirza](https://www.youtube.com/@fahdmirza)             |

</details>


# Why Yi? 

  - [Ecosystem](#ecosystem)
    - [Upstream](#upstream)
    - [Downstream](#downstream)
      - [Serving](#serving)
      - [Quantization](#quantization-1)
      - [Fine-tuning](#fine-tuning-1)
      - [API](#api)
  - [Benchmarks](#benchmarks)
    - [Chat model performance](#chat-model-performance)
    - [Base model performance](#base-model-performance)
      - [Yi-34B and Yi-34B-200K](#yi-34b-and-yi-34b-200k)
      - [Yi-9B](#yi-9b)

## Ecosystem

Yi has a comprehensive ecosystem, offering a range of tools, services, and models to enrich your experiences and maximize productivity.

- [Upstream](#upstream)
- [Downstream](#downstream)
  - [Serving](#serving)
  - [Quantization](#quantization-1)
  - [Fine-tuning](#fine-tuning-1)
  - [API](#api)

### Upstream

The Yi series models follow the same model architecture as Llama. By choosing Yi, you can leverage existing tools, libraries, and resources within the Llama ecosystem, eliminating the need to create new tools and enhancing development efficiency.

For example, the Yi series models are saved in the format of the Llama model. You can directly use `LlamaForCausalLM` and `LlamaTokenizer` to load the model. For more information, see [Use the chat model](#31-use-the-chat-model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34b", device_map="auto")
```

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Downstream

> 💡 Tip
> 
> - Feel free to create a PR and share the fantastic work you've built using the Yi series models.
>
> - To help others quickly understand your work, it is recommended to use the format of `<model-name>: <model-intro> + <model-highlights>`.

#### Serving 

If you want to get up with Yi in a few minutes, you can use the following services built upon Yi.

- Yi-34B-Chat: you can chat with Yi using one of the following platforms:
  - [Yi-34B-Chat | Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
  - [Yi-34B-Chat | Yi Platform](https://platform.lingyiwanwu.com/): **Note** that currently it's available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)) and experience it firsthand!
  
- [Yi-6B-Chat (Replicate)](https://replicate.com/01-ai): you can use this model with more options by setting additional parameters and calling APIs.
  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models): you can use this service to run Yi models locally with added flexibility and customization.
  
#### Quantization

If you have limited computational capabilities, you can use Yi's quantized models as follows. 

These quantized models have reduced precision but offer increased efficiency, such as faster inference speed and smaller RAM usage.

- [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ) 
- [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
- [TheBloke/Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)
  
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

#### API

- [amazing-openai-api](https://github.com/soulteary/amazing-openai-api): this tool converts Yi model APIs into the OpenAI API format out of the box.
- [LlamaEdge](https://www.secondstate.io/articles/yi-34b/#create-an-openai-compatible-api-service-for-the-yi-34b-chat-model): this tool builds an OpenAI-compatible API server for Yi-34B-Chat using a portable Wasm (WebAssembly) file, powered by Rust.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## Tech report

For detailed capabilities of the Yi series model, see [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652).

### Citation

```
@misc{ai2024yi,
    title={Yi: Open Foundation Models by 01.AI},
    author={01. AI and : and Alex Young and Bei Chen and Chao Li and Chengen Huang and Ge Zhang and Guanwei Zhang and Heng Li and Jiangcheng Zhu and Jianqun Chen and Jing Chang and Kaidong Yu and Peng Liu and Qiang Liu and Shawn Yue and Senbin Yang and Shiming Yang and Tao Yu and Wen Xie and Wenhao Huang and Xiaohui Hu and Xiaoyi Ren and Xinyao Niu and Pengcheng Nie and Yuchi Xu and Yudong Liu and Yue Wang and Yuxuan Cai and Zhenyu Gu and Zhiyuan Liu and Zonghong Dai},
    year={2024},
    eprint={2403.04652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Benchmarks 

- [Chat model performance](#chat-model-performance)
- [Base model performance](#base-model-performance)

### Chat model performance

Yi-34B-Chat model demonstrates exceptional performance, ranking first among all existing open-source models in the benchmarks including MMLU, CMMLU, BBH, GSM8k, and more.

![Chat model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_chat.png?raw=true) 

<details>
<summary> Evaluation methods and challenges. ⬇️ </summary>

- **Evaluation methods**: we evaluated various benchmarks using both zero-shot and few-shot methods, except for TruthfulQA.
- **Zero-shot vs. few-shot**: in chat models, the zero-shot approach is more commonly employed.
- **Evaluation strategy**: our evaluation strategy involves generating responses while following instructions explicitly or implicitly (such as using few-shot examples). We then isolate relevant answers from the generated text.
- **Challenges faced**: some models are not well-suited to produce output in the specific format required by instructions in few datasets, which leads to suboptimal results.

<strong>*</strong>: C-Eval results are evaluated on the validation datasets
</details>

### Base model performance

#### Yi-34B and Yi-34B-200K 

The Yi-34B and Yi-34B-200K models stand out as the top performers among open-source models, especially excelling in MMLU, CMMLU, common-sense reasoning, reading comprehension, and more.

![Base model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_base.png?raw=true)

<details>
<summary> Evaluation methods. ⬇️</summary>

- **Disparity in results**: while benchmarking open-source models, a disparity has been noted between results from our pipeline and those reported by public sources like OpenCompass.
- **Investigation findings**: a deeper investigation reveals that variations in prompts, post-processing strategies, and sampling techniques across models may lead to significant outcome differences.
- **Uniform benchmarking process**: our methodology aligns with the original benchmarks—consistent prompts and post-processing strategies are used, and greedy decoding is applied during evaluations without any post-processing for the generated content.
- **Efforts to retrieve unreported scores**: for scores that were not reported by the original authors (including scores reported with different settings), we try to get results with our pipeline.
- **Extensive model evaluation**: to evaluate the model’s capability extensively, we adopted the methodology outlined in Llama2. Specifically, we included PIQA, SIQA, HellaSwag, WinoGrande, ARC, OBQA, and CSQA to assess common sense reasoning. SquAD, QuAC, and BoolQ were incorporated to evaluate reading comprehension.
- **Special configurations**: CSQA was exclusively tested using a 7-shot setup, while all other tests were conducted with a 0-shot configuration. Additionally, we introduced GSM8K (8-shot@1), MATH (4-shot@1), HumanEval (0-shot@1), and MBPP (3-shot@1) under the category "Math & Code".
- **Falcon-180B caveat**: Falcon-180B was not tested on QuAC and OBQA due to technical constraints. Its performance score is an average from other tasks, and considering the generally lower scores of these two tasks, Falcon-180B's capabilities are likely not underestimated.
</details>

#### Yi-9B

Yi-9B is almost the best among a range of similar-sized open-source models (including Mistral-7B, SOLAR-10.7B, Gemma-7B, DeepSeek-Coder-7B-Base-v1.5 and more), particularly excelling in code, math, common-sense reasoning, and reading comprehension.

![Yi-9B benchmark - details](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_details.png?raw=true)

- In terms of **overall** ability (Mean-All), Yi-9B performs the best among similarly sized open-source models, surpassing DeepSeek-Coder, DeepSeek-Math, Mistral-7B, SOLAR-10.7B, and Gemma-7B.

  ![Yi-9B benchmark - overall](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_overall.png?raw=true)

- In terms of **coding** ability (Mean-Code), Yi-9B's performance is second only to DeepSeek-Coder-7B, surpassing Yi-34B, SOLAR-10.7B, Mistral-7B, and Gemma-7B.

  ![Yi-9B benchmark - code](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_code.png?raw=true)

- In terms of **math** ability (Mean-Math), Yi-9B's performance is second only to DeepSeek-Math-7B, surpassing SOLAR-10.7B, Mistral-7B, and Gemma-7B.

  ![Yi-9B benchmark - math](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_math.png?raw=true)

- In terms of **common sense and reasoning** ability (Mean-Text), Yi-9B's performance is on par with Mistral-7B, SOLAR-10.7B, and Gemma-7B.

  ![Yi-9B benchmark - text](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_text.png?raw=true)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

# Who can use Yi?

Everyone! 🙌 ✅

The code and weights of the Yi series models are distributed under the [Apache 2.0 license](https://github.com/01-ai/Yi/blob/main/LICENSE), which means the Yi series models are free for personal usage, academic purposes, and commercial use. 

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

# Misc.

### Acknowledgments

A heartfelt thank you to each of you who have made contributions to the Yi community! You have helped Yi not just a project, but a vibrant, growing home for innovation.

[![yi contributors](https://contrib.rocks/image?repo=01-ai/yi&max=2000&columns=15)](https://github.com/01-ai/yi/graphs/contributors)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### License

The code and weights of the Yi-1.5 series models are distributed under the [Apache 2.0 license](https://github.com/01-ai/Yi/blob/main/LICENSE).

If you create derivative works based on this model, please include the following attribution in your derivative works:

    This work is a derivative of [The Yi Series Model You Base On] by 01.AI, used under the Apache 2.0 License.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

