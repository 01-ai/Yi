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
<a href="https://github.com/01-ai/Yi/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
  <img src="https://img.shields.io/badge/Model_License-Yi_License-lightblue">
</a>
<a href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B">
</a>

</div>

<div align="center">
  <h3 align="center">Building the Next Generation of Open-Source and Bilingual LLMs</h3>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/01-ai" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/01ai/" target="_blank">ModelScope</a> • ✡️ <a href="https://wisemodel.cn/organization/01.AI" target="_blank">WiseModel</a>
</p> 

<p align="center">
    👩‍🚀 Ask questions or discuss ideas on <a href="https://github.com/01-ai/Yi/discussions" target="_blank"> GitHub </a>!
</p> 

<p align="center">
    👋 Join us on 💬 <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245" target="_blank"> WeChat (Chinese) </a>!
</p> 

<p align="center">
    📚 Grow at <a href="https://github.com/01-ai/Yi/blob/main/docs/learning_hub.md"> Yi Learning Hub </a>!
</p> 
<hr>

<ul>
  <li>🙌 本文由 Yi 和志愿者共同翻译完成，感谢每一位传递知识的火炬手。</li>

  <li>🤗 欢迎大家 <a href="https://github.com/01-ai/Yi/discussions/314">加入我们</a>，开启知识之火旅程，共绘技术内容图谱。</li>
  
  <li>📝 本文翻译使用了 <a href="https://huggingface.co/spaces/01-ai/Yi-34B-Chat">Yi-34B-Chat</a>，关于翻译时使用的 prompt 及最佳实践，参阅 <a href="https://github.com/01-ai/Yi/wiki/%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AE%A1%E6%A0%A1%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF">「翻译与审校的正确姿势」</a>。</li>
</ul>



<!-- DO NOT REMOVE ME -->

<hr>

<details open>
<summary></b>📕 Table of Contents</b></summary>


- [🟢 What is Yi?](#-what-is-yi)
  - [📌 Introduction](#-introduction)
  - [🎯 Models](#-models)
    - [Chat models](#chat-models)
    - [Base models](#base-models)
    - [Other info](#other-info)
  - [🎉 News](#-news)
- [🟢 How to use Yi?](#-how-to-use-yi)
  - [快速上手](#quick-start)
    - [选择你的学习路径](#choose-your-path)
    - [快速上手 - 使用 PyPi（pip install）](#pip)
    - [快速上手 - 使用 llama.cpp 量化运行](#llamacpp-快速入门)
    - [快速上手 - 使用 Web Demo](#网页版演示应用web-demo)
    - [快速上手 - 使用 Docker](#docker-快速入门)
  - [Fine tune](#fine-tune)
  - [Quantization](#quantization)
  - [Deployment](#deployment)
  - [Learning hub](#learning-hub)
- [🟢 Why Yi?](#-why-yi)
  - [🌎 Ecosystem](#-ecosystem)
    - [💦 Upstream](#-upstream)
    - [🌊 Downstream](#-downstream)
      - [🔗 Serving](#-serving)
      - [⚙️ Quantitation](#️-quantitation)
      - [🛠️ Fine-tuning](#️-fine-tuning)
      - [API](#api)
  - [📌 Benchmarks](#-benchmarks)
    - [📊 Base model performance](#-base-model-performance)
    - [📊 Chat model performance](#-chat-model-performance)
- [🟢 Who can use Yi?](#-who-can-use-yi)
- [🟢 Misc.](#-misc)
  - [Ackknowledgements](#acknowledgments)
  - [📡 Disclaimer](#-disclaimer)
  - [🪪 License](#-license)

</details>

<hr>

# 🟢 What is Yi?

## 📌 Introduction 

- 🤖 The Yi series models are the next generation of open-source large language models trained from scratch by [01.AI](https://01.ai/).

- 🙌 Targeted as a bilingual language model and trained on 3T multilingual corpus, the Yi series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more. For example,

  - For English language capability, the Yi series models ranked 2nd (just behind GPT-4), outperforming other LLMs (such as LLaMA2-chat-70B, Claude 2, and ChatGPT) on the [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) in Dec 2023.
  
  - For Chinese language capability, the Yi series models landed in 2nd place (following GPT-4), surpassing other LLMs (such as Baidu ERNIE, Qwen, and Baichuan) on the [SuperCLUE](https://www.superclueai.com/) in Oct 2023.
  
  - 🙏 (Credits to LLaMA) Thanks to the Transformer and LLaMA open-source communities, as they reducing the efforts required to build from scratch and enabling the utilization of the same tools within the AI ecosystem.  
  <details style="display: inline;"><summary> If you're interested in Yi's adoption of LLaMA architecture and license usage policy, see  <span style="color:  green;">Yi's relation with LLaMA</span> ⬇️</summary> <ul> <br>
> 💡 TL;DR
> 
> The Yi series models adopt the same model architecture as LLaMA but are **NOT** derivatives of LLaMA.

- Both Yi and LLaMA are all based on the Transformer structure, which has been the standard architecture for large language models since 2018.

- Grounded in the Transformer architecture, LLaMA has become a new cornerstone for the majority of state-of-the-art open-source models due to its excellent stability, reliable convergence, and robust compatibility. This positions LLaMA as the recognized foundational framework for models including Yi.

- Thanks to the Transformer and LLaMA architectures, other models can leverage their power, reducing the effort required to build from scratch and enabling the utilization of the same tools within their ecosystems.

- However, the Yi series models are NOT derivatives of LLaMA, as they do not use LLaMA's weights.

  - As LLaMA's structure is employed by the majority of open-source models, the key factors of determining model performance are training datasets, training pipelines, and training infrastructure.

  - Developing in a unique and proprietary way, Yi has independently created its own high-quality training datasets, efficient training pipelines, and robust training infrastructure entirely from the ground up. This effort has led to excellent performance with Yi series models ranking just behind GPT4 and surpassing LLaMA on the [Alpaca Leaderboard in Dec 2023](https://tatsu-lab.github.io/alpaca_eval/). 
</ul>
</details>




<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

## 🎯 Models

Yi models come in multiple sizes and cater to different use cases. You can also fine-tune Yi models to meet your specific requirements. 

If you want to deploy Yi models, make sure you meet the [software and hardware requirements](#deployment).

### Chat models

| Model | Download  
|---|---
Yi-34B-Chat	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)
Yi-34B-Chat-4bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-4bits/summary)
Yi-34B-Chat-8bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-8bits/summary)
Yi-6B-Chat| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat/summary)
Yi-6B-Chat-4bits |	• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-4bits/summary)
Yi-6B-Chat-8bits	|  • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-8bits/summary)


<sub><sup> - 4-bit series models are quantized by AWQ. <br> - 8-bit series models are quantized by GPTQ <br> - All quantized models have a low barrier to use since they can be deployed on consumer-grade GPUs (e.g., 3090, 4090). </sup></sub>

### Base models

| Model | Download | 
|---|---|
Yi-34B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B/summary)
Yi-34B-200K|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-200K/summary)
Yi-6B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B/summary)
Yi-6B-200K	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-200K) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-200K/summary)

<sub><sup> - 200k is roughly equivalent to 400,000 Chinese characters.  </sup></sub>

### Other info

- For chat and base models:

  - 6B series models are suitable for personal and academic use.

  - 34B series models suitable for personal, academic, and commercial (particularly for small and medium-sized enterprises) purposes. It's a cost-effective solution that's affordable and equipped with emergent ability.

  - The **default context window** is **4k tokens**.
    
  - The pretrained tokens are 3T.
    
  - The training data are up to June 2023.	

- For chat models:
  
  <details style="display: inline;"><summary>For chat model limitations, see ⬇️</summary>
   <ul>
    <br>The released chat model has undergone exclusive training using Supervised Fine-Tuning (SFT). Compared to other standard chat models, our model produces more diverse responses, making it suitable for various downstream tasks, such as creative scenarios. Furthermore, this diversity is expected to enhance the likelihood of generating higher quality responses, which will be advantageous for subsequent Reinforcement Learning (RL) training.

    <br>However, this higher diversity might amplify certain existing issues, including:
      <li>Hallucination: This refers to the model generating factually incorrect or nonsensical information. With the model's responses being more varied, there's a higher chance of hallucination that are not based on accurate data or logical reasoning.</li>
      <li>Non-determinism in re-generation: When attempting to regenerate or sample responses, inconsistencies in the outcomes may occur. The increased diversity can lead to varying results even under similar input conditions.</li>
      <li>Cumulative Error: This occurs when errors in the model's responses compound over time. As the model generates more diverse responses, the likelihood of small inaccuracies building up into larger errors increases, especially in complex tasks like extended reasoning, mathematical problem-solving, etc.</li>
      <li>To achieve more coherent and consistent responses, it is advisable to adjust generation configuration parameters such as temperature, top_p, or top_k. These adjustments can help in the balance between creativity and coherence in the model's outputs.</li>
</ul>
</details>

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

## 🎉 News 

<details>
<summary>🎯 <b>2023/11/23</b>: The chat models are open to public.</summary>

This release contains two chat models based on previously released base models, two 8-bit models quantized by GPTQ, and two 4-bit models quantized by AWQ.

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

This release contains two base models with the same parameter sizes as the previous
release, except that the context window is extended to 200K.

</details>

<details>
<summary>🎯 <b>2023/11/02</b>: The base model of <code>Yi-6B</code> and <code>Yi-34B</code>.</summary>

The first public release contains two bilingual (English/Chinese) base models
with the parameter sizes of 6B and 34B.  Both of them are trained with 4K
sequence length and can be extended to 32K during inference time.

</details>

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

# 🟢 How to use Yi?

- [Quick start](#quick-start)
  - [Choose your path](#choose-your-parth)
  - [pip](#pip)
  - [llama.cpp](https://github.com/01-ai/Yi/blob/main/docs/yi_llama.cpp.md)
  - [Web demo](#web-demo)
- [Fine tune](#fine-tune)
- [Quantization](#quantization)
- [Deployment](https://github.com/01-ai/Yi/blob/main/docs/deployment.md)
- [Learning hub](https://github.com/01-ai/Yi/blob/main/docs/learning_hub.md)

## Quick start

Getting up and running with Yi models is simple with multiple choices available. 

### Choose your path

Select one of the following paths to begin your journey with Yi!

 ![Quick start - Choose your path](https://github.com/01-ai/Yi/blob/main/assets/img/quick_start_path.png)

#### 🎯 Deploy Yi locally

If you prefer to deploy Yi models locally, 

  - 🙋‍♀️ and you have **sufficient** resources (for example, NVIDIA A800 80GB), you can choose one of the following methods:
    - [pip](#pip)
    - [Docker](./docs/README_legacy_cn.md#11-docker)
    - [conda-lock](https://github.com/01-ai/Yi/blob/main/docs/README_legacy.md#12-local-development-environment)

  - 🙋‍♀️ and you have **limited** resources (for example, a MacBook Pro), you can use [llama.cpp](#quick-start---llamacpp)

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

### Quick start - pip

This tutorial guides you through every step of running **Yi-34B-Chat locally on an A800 (80G)** and then performing inference.

#### Step 0: Prerequistes
 
- Make sure Python 3.10 or a later version is installed.

- If you want to run other Yi models, see [software and hardware requirements](#deployment)

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

The steps are similar to [pip - Perform inference with Yi chat model](#perform-inference-with-yi-chat-model).

You can use the existing file [`text_generation.py`](https://github.com/01-ai/Yi/tree/main/demo).

```bash
python demo/text_generation.py  --model <your-model-path>
```

Then you can see an output similar to the one below. 🥳

<details>

<summary>Output ⬇️ </summary>

<br>

**Prompt**: Let me tell you an interesting story about cat Tom and mouse Jerry,

**Generation**: Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

</details>

### Docker 快速入门

<details>
<summary> 🚀 以下是如何在本地运行 Yi-34B-Chat 模型的详细教程 👇</summary>
<br>这个教程将指导你如何在本地 A800 GPU 上运行 <strong>Yi-34B-Chat</strong> 模型，并执行推理。
<h4>步骤0: 准备工作</h4>
<p>确保你已经安装了 <a href="https://docs.docker.com/engine/install/?open_in_browser=true">Docker</a> 和 <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a>。</p>
<h4>步骤1: 启动 Docker 容器</h4>
<pre><code>docker run -it --gpus all \
-v &lt;your-model-path&gt;: /models
ghcr.io/01-ai/yi:latest
</code></pre>
<p>或者，你也可以从<code>registry.lingyiwanwu.com/ci/01-ai/yi:latest</code> 拉取已经构建好的 Yi Docker 镜像。</p>

<h4>步骤2: 执行推理</h4>
    <p>你可以使用Yi 的聊天模型或基础模型来执行推理。</p>
    
<h5>使用 Yi 聊天模型执行推理</h5>
    <p>执行推理的步骤与使用<a href="#perform-inference-with-yi-chat-model">pip安装指南</a>类似。</p>
    <p><strong>请注意</strong> 唯一不同的是你需要设置 <code>model_path = '&lt;your-model-mount-path&gt;'</code> 而不是 <code>model_path = '&lt;your-model-path&gt;'</code>。</p>
<h5>使用 Yi 基础模型执行推理</h5>
    <p>执行推理的步骤与使用<a href="#perform-inference-with-yi-chat-model">pip安装指南</a>类似。</p>
    <p><strong>请注意</strong> 唯一不同的是你需要设置 <code>--model &lt;your-model-mount-path&gt;'</code> 而不是 <code>model &lt;your-model-path&gt;</code>。</p>
</details>



### llama.cpp 快速入门
<details>
<summary> 🚀 以下是使用 llama.cpp 在本地运行 Yi-chat-6B-2bits 模型的详细教程👇 </summary> 
<br>该教程分享如何在本地运行 <a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a> 量化模型，并且进行推理。</p>

- [步骤 0: 必备条件](#step-0-prerequisites)
- [Step 1: Download llama.cpp](#step-1-download-llamacpp)
- [Step 2: Download Yi model](#step-2-download-yi-model)
- [Step 3: Perform inference](#step-3-perform-inference)

#### 步骤 0: 必备条件

- 首先，你需要一台装备了 16GB 内存和 Apple M2 Pro 芯片的 MacBook Pro。

- 确保你的电脑上安装了 [`git-lfs`](https://git-lfs.com/) 。
  
#### 步骤 1: 下载 `llama.cpp`

要克隆 [`llama.cpp`](https://github.com/ggerganov/llama.cpp) 仓库，请运行以下命令：

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

#### 步骤 2: 克隆 Yi 模型

步骤 2.1: 运行以下命令，从 [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) 克隆仅包含代码的仓库：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

步骤 2.2: 要下载一个量化后的 Yi 模型 [yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf)，执行以下命令：

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

#### 步骤 3: 执行推理

想要体验 Yi 模型（进行模型推理），我们可以选择以下任意一种方法。

- [方法 1：在终端中执行推理](#method-1-perform-inference-in-terminal)
  
- [方法 2：在网页上执行推理](#method-2-perform-inference-in-web)

### 方法一：在终端中执行推理

要使用4个线程编译 `llama.cpp` 并随后进行推理，请导航到 `llama.cpp` 所在的目录，并运行以下命令。

> ### 提示
>
> - 将 `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` 替换为你的模型的实际路径。
>
> - 默认情况下，模型是续写模式。
> - 要查看更多输出自定义选项的详细描述和使用方法（例如系统提示、温度、重复惩罚等），运行 `./main -h` 进行检查。

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

恭喜你！你已经成功地向Yi模型提出了问题并得到了回答！🥳

### 方法二：在网页上进行推理

1. 要初始化一个轻量级、快速的聊天机器人，请导航到 `llama.cpp` 目录，并运行以下命令。

    ```bash
    ./server --ctx-size 2048 --host 0.0.0.0 --n-gpu-layers 64 --model /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf
    ```

    然后，你将看到以下类似的输出：

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

2. 要访问聊天机器人界面，打开你的网络浏览器，并在地址栏中输入 `http://0.0.0.0:8080`。

    ![Yi模型聊天机器人界面 - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp1.png)

3. 在提示窗口中输入一个问题，比如“你如何喂养你的宠物狐狸？请用6个简单的步骤回答这个问题”，你将会收到一个相应的答案。

    ![向Yi模型提问 - llama.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp2.png)

</ul>
</details>

### 网页版演示应用（Web Demo）

你可以使用 Yi **聊天模型**（Yi-34B-Chat）打造一个 Web Demo。注意：Yi 基础模型（Yi-34B）不支持该功能。

[第一步：准备环境](#step-1-prepare-your-environment)

[第二步：下载模型](#step-2-download-the-yi-model)

第三步：启动网页服务，运行以下命令。

```bash
python demo/web_demo.py -c <你的模型路径>
```

命令运行完毕后，你可以在浏览器中输入控制台提供的网址，来使用 Web Demo。

 ![快速上手 - Web Demo](./assets/img/yi_34b_chat_web_demo.gif)

### Finetuning

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

Once finished, you can compare the finetuned model and the base model with the following command:

```bash
bash finetune/scripts/run_eval.sh
```
<details style="display: inline;"><summary>For advanced usage (like fine-tuning based on your custom data), see ⬇️</summary> <ul>

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

For the Yi-6B model, a node with 4 GPUs, each has GPU mem larger than 60GB is recommended.

For the Yi-34B model, because the usage of zero-offload technique takes a lot CPU memory, please be careful to limit the GPU numbers in 34B finetune training. Please use CUDA_VISIBLE_DEVICES to limit the GPU number (as shown in scripts/run_sft_Yi_34b.sh).

A typical hardware setup for finetuning 34B model is a node with 8GPUS (limit to 4 in running by CUDA_VISIBLE_DEVICES=0,1,2,3), each has GPU mem larger than 80GB, with total CPU mem larger than 900GB.

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

Then you'll see the answer from both the base model and the finetuned model
</ul>
</details>

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

<details style="display: inline;"><summary>For a more detailed explanation, see ⬇️</summary> <ul>

#### GPT-Q quantization

[GPT-Q](https://github.com/IST-DASLab/gptq) is a PTQ(Post-Training Quantization)
method. It's memory saving and provides potential speedups while retaining the accuracy
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
<details style="display: inline;"><summary>For detailed explanations, see ⬇️</summary> <ul>

#### AWQ quantization

[AWQ](https://github.com/mit-han-lab/llm-awq) is a PTQ(Post-Training Quantization)
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
<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

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
|----------------------|--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | RTX 3090 <br> RTX 4090 <br>  A10 <br> A30             |
| Yi-6B-Chat-4bits     | 4 GB          | RTX 3060 <br>  RTX 4060                     |
| Yi-6B-Chat-8bits     | 8 GB          | RTX 3070 <br> RTX 4060                     |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 <br> A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | RTX 3090  <br> RTX 4090 <br> A10 <br> A30 <br> A100 (40GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090  <br> 2 x RTX 4090 <br> A800  (40GB) |

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
| Yi-6B                | 15 GB         | RTX3090 <br> RTX4090 <br> A10 <br> A30               |
| Yi-6B-200K           | 50 GB         | A800 (80 GB)                            |
| Yi-34B               | 72 GB         | 4 x RTX 4090 <br> A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |

### Learning hub

<details>
<summary> If you want to learn Yi, you can find a wealth of helpful educational resources here ⬇️</summary> 
<br> 
  
Welcome to the Yi learning hub! 

Whether you're a seasoned developer or a newcomer, you can find a wealth of helpful educational resources to enhance your understanding and skills with Yi models, including insightful blog posts, comprehensive video tutorials, hands-on guides, and more.  

The content you find here has been generously contributed by knowledgeable Yi experts and passionate enthusiasts. We extend our heartfelt gratitude for your invaluable contributions! 

At the same time, we also warmly invite you to join our collaborative effort by contributing to Yi. If you have already made contributions to Yi, please don't hesitate to showcase your remarkable work in the table below.

With all these resources at your fingertips, you're ready to start your exciting journey with Yi. Happy learning! 🥳

#### Tutorials

| Type        | Deliverable                                            |      Date      |     Author     |
|-------------|--------------------------------------------------------|----------------|----------------|
| Blog        | [本地运行零一万物 34B 大模型，使用 Llama.cpp & 21G 显存](https://zhuanlan.zhihu.com/p/668921042)                  |  2023-11-26  |  [苏洋](https://github.com/soulteary)  |
| Blog        | [Running Yi-34B-Chat locally using LlamaEdge](https://www.secondstate.io/articles/yi-34b/)                   |  2023-11-30  |  [Second State](https://github.com/second-state)  |
| Blog        | [零一万物模型折腾笔记：官方 Yi-34B 模型基础使用](https://zhuanlan.zhihu.com/p/671387298)                           | 2023-12-10 |  [苏洋](https://github.com/soulteary)  |
| Blog        | [CPU 混合推理，非常见大模型量化方案：“二三五六” 位量化方案](https://zhuanlan.zhihu.com/p/671698216)                  | 2023-12-12 |  [苏洋](https://github.com/soulteary)  |
| Video       | [只需 24G 显存，用 vllm 跑起来 Yi-34B 中英双语大模型](https://www.bilibili.com/video/BV17t4y1f7Ee/)               | 2023-12-28 |  漆妮妮  |
| Video       | [Install Yi 34B Locally - Chinese English Bilingual LLM](https://www.youtube.com/watch?v=CVQvj4Wrh4w&t=476s) | 2023-11-05  |  Fahd Mirza  |
</details>


# 🟢 Why Yi? 

  - [🌎 Ecosystem](#-ecosystem)
    - [💦 Upstream](#-upstream)
    - [🌊 Downstream](#-downstream)
      - [🔗 Serving](#-serving)
      - [⚙️ Quantitation](#️-quantitation)
      - [🛠️ Fine-tuning](#️-fine-tuning)
      - [API](#api)
  - [📌 Benchmarks](#-benchmarks)
    - [📊 Chat model performance](#-chat-model-performance)
    - [📊 Base model performance](#-base-model-performance)
 
## 🌎 Ecosystem

Yi has a comprehensive ecosystem, offering a range of tools, services, and models to enrich your experiences and maximize productivity.

- [💦 Upstream](#-upstream)
- [🌊 Downstream](#-downstream)
  - [🔗 Serving](#-serving)
  - [⚙️ Quantitation](#️-quantitation)
  - [🛠️ Fine-tuning](#️-fine-tuning)
  - [API](#api)

### 💦 Upstream

The Yi series models follow the same model architecture as LLaMA. By choosing Yi, you can leverage existing tools, libraries, and resources within the LLaMA ecosystem, eliminating the need to create new tools and enhancing development efficiency.

For example, the Yi series models are saved in the format of the LLaMA model. You can directly use `LLaMAForCausalLM` and `LLaMATokenizer` to load the model. For more information, see [Use the chat model](#31-use-the-chat-model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34b", device_map="auto")
```

### 🌊 Downstream

> 💡 Tip
> 
> - Feel free to create a PR and share the fantastic work you've built using the Yi series models.
>
> - To help others quickly understand your work, it is recommended to use the format of `<model-name>: <model-intro> + <model-highlights>`.

#### 🔗 Serving 

If you want to get up with Yi in a few minutes, you can use the following services built upon Yi.

- Yi-34B-Chat: you can chat with Yi using one of the following platforms:
  - [Yi-34B-Chat | Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
  - [Yi-34B-Chat | Yi Platform](https://platform.lingyiwanwu.com/): **Note** that currently it's available through a whitelist. Welcome to apply (fill out a form in [English](https://cn.mikecrm.com/l91ODJf) or [Chinese](https://cn.mikecrm.com/gnEZjiQ)) and experience it firsthand!
  
- [Yi-6B-Chat (Replicate)](https://replicate.com/01-ai): you can use this model with more options by setting additional parameters and calling APIs.
  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models): you can use this service to run Yi models locally with added flexibility and customization.
  
#### ⚙️ Quantitation

If you have limited computational capabilities, you can use Yi's quantized models as follows. 

These quantized models have reduced precision but offer increased efficiency, such as faster inference speed and smaller RAM usage.

- [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ) 
- [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
- [TheBloke/Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)
  
#### 🛠️ Fine-tuning

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

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

## 📌 Benchmarks 

- [📊 Chat model performance](#-chat-model-performance)
- [📊 Base model performance](#-base-model-performance)

### 📊 Chat model performance

Yi-34B-Chat model demonstrates exceptional performance, ranking first among all existing open-source models in the benchmarks including MMLU, CMMLU, BBH, GSM8k, and more.

![Chat model performance](./assets/img/benchmark_chat.png) 

<details>
<summary> Evaluation methods and challenges ⬇️ </summary>

- **Evaluation methods**: we evaluated various benchmarks using both zero-shot and few-shot methods, except for TruthfulQA.
- **Zero-shot vs. few-shot**: in chat models, the zero-shot approach is more commonly employed.
- **Evaluation strategy**: our evaluation strategy involves generating responses while following instructions explicitly or implicitly (such as using few-shot examples). We then isolate relevant answers from the generated text.
- **Challenges faced**: some models are not well-suited to produce output in the specific format required by instructions in few datasets, which leads to suboptimal results.

<strong>*</strong>: C-Eval results are evaluated on the validation datasets
</details>

### 📊 Base model performance

The Yi-34B and Yi-34B-200K models stand out as the top performers among open-source models, especially excelling in MMLU, CMML, common-sense reasoning, reading comprehension, and more.

![Base model performance](./assets/img/benchmark_base.png)

<details>
<summary> Evaluation methods ⬇️</summary>

- **Disparity in Results**: while benchmarking open-source models, a disparity has been noted between results from our pipeline and those reported by public sources like OpenCompass.
- **Investigation Findings**: a deeper investigation reveals that variations in prompts, post-processing strategies, and sampling techniques across models may lead to significant outcome differences.
- **Uniform Benchmarking Process**: our methodology aligns with the original benchmarks—consistent prompts and post-processing strategies are used, and greedy decoding is applied during evaluations without any post-processing for the generated content.
- **Efforts to Retrieve Unreported Scores**: for scores that were not reported by the original authors (including scores reported with different settings), we try to get results with our pipeline.
- **Extensive Model Evaluation**: to evaluate the model’s capability extensively, we adopted the methodology outlined in Llama2. Specifically, we included PIQA, SIQA, HellaSwag, WinoGrande, ARC, OBQA, and CSQA to assess common sense reasoning. SquAD, QuAC, and BoolQ were incorporated to evaluate reading comprehension.
- **Special Configurations**: CSQA was exclusively tested using a 7-shot setup, while all other tests were conducted with a 0-shot configuration. Additionally, we introduced GSM8K (8-shot@1), MATH (4-shot@1), HumanEval (0-shot@1), and MBPP (3-shot@1) under the category "Math & Code".
- **Falcon-180B Caveat**: Falcon-180B was not tested on QuAC and OBQA due to technical constraints. Its performance score is an average from other tasks, and considering the generally lower scores of these two tasks, Falcon-180B's capabilities are likely not underestimated.
</details>

# 🟢 Who can use Yi?

Everyone! 🙌 ✅

- The Yi series models are free for personal usage, academic purposes, and commercial use. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)
  
- For free commercial use, you only need to [complete this form](https://www.lingyiwanwu.com/yi-license) to get a Yi Model Commercial License.

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

# 🟢 Misc.

### Acknowledgments

A heartfelt thank you to each of you who have made contributions to the Yi community! You have helped Yi not just a project, but a vibrant, growing home for innovation.

<!---
ref https://github.com/ngryman/contributor-faces
npx contributor-faces --exclude "*bot*" --limit 70 --repo "https://github.com/01-ai/Yi"

change the height and width for each of the contributors from 80 to 50 at ref index.js.
--->

[//]: contributor-faces
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/ZhaoFancy"><img style="margin:0" src="https://avatars.githubusercontent.com/u/139539780?v=4" title="ZhaoFancy" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/Anonymitaet"><img style="margin:0" src="https://avatars.githubusercontent.com/u/50226895?v=4" title="Anonymitaet" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/findmyway"><img style="margin:0" src="https://avatars.githubusercontent.com/u/5612003?v=4" title="findmyway" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/shiyue-loop"><img style="margin:0" src="https://avatars.githubusercontent.com/u/150643331?v=4" title="shiyue-loop" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/richardllin"><img style="margin:0" src="https://avatars.githubusercontent.com/u/1932744?v=4" title="richardllin" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/jiangchengSilent"><img style="margin:0" src="https://avatars.githubusercontent.com/u/143983063?v=4" title="jiangchengSilent" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/loofahcus"><img style="margin:0" src="https://avatars.githubusercontent.com/u/15729967?v=4" title="loofahcus" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/Yimi81"><img style="margin:0" src="https://avatars.githubusercontent.com/u/66633207?v=4" title="Yimi81" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/ly-nld"><img style="margin:0" src="https://avatars.githubusercontent.com/u/38471793?v=4" title="ly-nld" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/WayTooWill"><img style="margin:0" src="https://avatars.githubusercontent.com/u/119883899?v=4" title="WayTooWill" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/kai01ai"><img style="margin:0" src="https://avatars.githubusercontent.com/u/140378742?v=4" title="kai01ai" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/forpanyang"><img style="margin:0" src="https://avatars.githubusercontent.com/u/138085590?v=4" title="forpanyang" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/0x1111"><img style="margin:0" src="https://avatars.githubusercontent.com/u/750392?v=4" title="0x1111" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/angeligareta"><img style="margin:0" src="https://avatars.githubusercontent.com/u/32129522?v=4" title="angeligareta" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/xffxff"><img style="margin:0" src="https://avatars.githubusercontent.com/u/30254428?v=4" title="xffxff" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/tpoisonooo"><img style="margin:0" src="https://avatars.githubusercontent.com/u/7872421?v=4" title="tpoisonooo" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/tdolan21"><img style="margin:0" src="https://avatars.githubusercontent.com/u/40906019?v=4" title="tdolan21" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/statelesshz"><img style="margin:0" src="https://avatars.githubusercontent.com/u/28150734?v=4" title="statelesshz" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/renxiaoyi"><img style="margin:0" src="https://avatars.githubusercontent.com/u/10918916?v=4" title="renxiaoyi" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/markli404"><img style="margin:0" src="https://avatars.githubusercontent.com/u/116385770?v=4" title="markli404" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/fecet"><img style="margin:0" src="https://avatars.githubusercontent.com/u/41792945?v=4" title="fecet" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/cArlIcon"><img style="margin:0" src="https://avatars.githubusercontent.com/u/7384654?v=4" title="cArlIcon" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/alabulei1"><img style="margin:0" src="https://avatars.githubusercontent.com/u/45785633?v=4" title="alabulei1" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/eltociear"><img style="margin:0" src="https://avatars.githubusercontent.com/u/22633385?v=4" title="eltociear" width="50" height="50"></a>
<a style="display:inline-block;width=50px;height=50px" href="https://github.com/Gmgge"><img style="margin:0" src="https://avatars.githubusercontent.com/u/48548141?v=4" title="Gmgge" width="50" height="50"></a>

[//]: contributor-faces

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

### 📡 Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>

### 🪪 License

The source code in this repo is licensed under the [Apache 2.0
license](https://github.com/01-ai/Yi/blob/main/LICENSE). The Yi series models are fully open for academic research and free for commercial use, with automatic permission granted upon application. All usage must adhere to the [Yi Series Models Community License Agreement 2.1](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).
For free commercial use, you only need to send an email to [get official commercial permission](https://www.lingyiwanwu.com/yi-license).

<div align="right"> [ <a href="#building-the-next-generation-of-open-source-and-bilingual-llms">Back to top ⬆️ </a> ] </div>
