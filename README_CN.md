<p align="left">
    <a href="README.md">English</a> &nbsp; | &nbsp中文&nbsp</a>
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
<a href="https://github.com/01-ai/Yi/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
  <img src="https://img.shields.io/badge/Model_License-Yi_License-lightblue">
</a>
<a href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B">
</a>

<div id="top"></div>

</div>
<div align="center">
  <h3 align="center">打造新一代开源双语大语言模型</h3>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/01-ai" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/01ai/" target="_blank">魔搭社区 ModelScope</a> • ✡️ <a href="https://wisemodel.cn/organization/01.AI" target="_blank">始智AI WiseModel</a>
</p> 

<p align="center">
    👩‍🚀 欢迎你来 <a href="https://github.com/01-ai/Yi/discussions" target="_blank"> GitHub </a> 提问讨论
</p> 

<p align="center">
    👋 欢迎你加入我们的 💬 <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245" target="_blank"> 微信群 </a>一起交流
</p> 

<p align="center">
    📚 欢迎你来 <a href="#学习俱乐部"> Yi 学习俱乐部 </a>探索新知
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
<summary></b>📕 目录</b></summary>

- [🟢 Yi 是什么?](#-yi-是什么)
  - [📌 介绍](#-介绍)
  - [🎯 模型](#-模型)
    - [聊天模型](#聊天模型)
    - [基座模型](#基座模型)
    - [其他信息](#其他信息)
  - [🎉 最新动态](#-最新动态)
- [🟢 如何使用 Yi?](#-如何使用-yi)
  - [快速上手](#快速上手)
    - [选择路径](#选择路径)
    - [快速上手 - 使用 pip](#快速上手---pip)
    - [快速上手 - 使用 Docker](#快速上手---docker)
    - [快速上手 - 使用 conda-lock](#快速上手---conda-lock)
    - [快速上手 - 使用 llama.cpp](#快速上手---llamacpp)
    - [快速上手 - 使用网页演示](#快速上手---使用网页演示)
  - [微调](#微调)
  - [量化](#量化)
  - [部署](#部署)
  - [学习俱乐部](#学习俱乐部)
- [🟢 为什么选择Yi？](#-为什么选择yi)
  - [🌎 生态系统](#-生态系统)
    - [💦 上游](#-上游)
    - [🌊 下游](#-下游)
      - [🔗 服务](#-服务)
      - [⚙️ 量化](#️-量化)
      - [🛠️ 微调](#️-微调)
      - [API](#api)
  - [📌 基准测试](#-基准测试)
    - [📊 聊天模型性能](#-聊天模型性能)
    - [📊 基座模型性能](#-基座模型性能)
- [🟢 谁可以使用 Yi？](#-谁可以使用-yi)
- [🟢 其他内容](#-其他内容)
  - [致谢](#致谢)
  - [📡 免责声明](#-免责声明)
  - [🪪 许可证](#-许可证)

</details>

<hr>

# 🟢 Yi 是什么?

## 📌 介绍

- 🤖 Yi 系列模型是 [01.AI](https://01.ai/) 从零训练的新一代开源大语言模型。

- 🙌 Yi 系列模型是一个双语语言模型，在 3T 多语言语料库上训练而成，是全球最强大的大型语言模型之一。本系列模型在语言认知、常识推理、阅读理解等方面展现出巨大的潜力。例如，

  - 英语语言能力方面，Yi系列模型在2023年12月的 [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) 排行榜上排名第二（仅次于GPT-4），超过了其他大语言模型（LLM），如 Llama2-chat-70B、Claude 2 和 ChatGPT。

  - 中文语言能力方面，Yi系列模型在2023年10月的 [SuperCLUE](https://www.superclueai.com/) 排行榜上排名第二（仅次于GPT-4），超过了其他大语言模型，如百度ERNIE、Qwen 和 Baichuan。
  - 🙏 （感谢 Llama ）感谢 Transformer 和 Llama 开源社区，简化了 [01.AI](https://01.ai/) 从零开始构建大模型的工作，[01.AI](https://01.ai/) 也能够在人工智能生态系统中使用相同的工具。

  <details style="display: inline;"><summary> 如果你对 Yi 采用 Llama 架构及其许可使用政策感兴趣，参阅 <span style="color:  green;"> Yi 与 Llama 的关系。</span> ⬇️</summary> <ul> <br>

> 💡 简短总结
> 
> Yi 系列模型采用模型架构与Llama相同，但它们**不是**Llama的衍生品。


- Yi 和 Llama 都是基于 Transformer 结构构建的。实际上，自2018年以来，Transformer 一直是大语言模型的常用架构。

- 在 Transformer 架构的基础上，Llama 凭借出色的稳定性、可靠的收敛性和强大的兼容性，成为大多数先进开源模型的基石。因此，Llama 也成为 Yi 等模型的基础框架。

- 得益于 Transformer 和 Llama 架构，各类模型可以简化从零开始构建模型的工作，并能够在各自的生态系统中使用相同的工具。

- 然而，Yi 系列模型不是 Llama 的衍生品，因为它们不使用 Llama 的权重。

  - 虽然大多数开源模型都采用了 Llama 的结构，但决定模型表现的关键因素是训练所使用的数据集、流水线及其基础设施。

  - [01.AI](https://01.ai/) 用独特的方式开发了 Yi，从零开始独立创建了自己的高质量训练数据集、高效的训练流水线和强大的训练基础设施，因此Yi系列模型在性能上取得了卓越的成绩，在2023年12月的 [Alpaca Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) 上排名仅次于 GPT4，超过了 Llama。
</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>


## 🎉 最新动态

<details open>
  <summary>🎯 <b> 2024/01/23</b>: <code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code> 和 <code><a href="https://huggingface.co/01-ai/Yi-VL-6B">Yi-VL-6B</a></code>的多模态语言大模型，均已开源并对公众开放。</summary>
  <br>
   在<a href="https://arxiv.org/abs/2311.16502">MMMU</a> 和 <a href="https://arxiv.org/abs/2401.11944">CMMMU</a>最新的基准测试中（截至2024年1月的可用数据），<code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code>荣登榜首。</li>
</details>

<details>
<summary>🎯 <b>2023/11/23</b>: 六大聊天模型均已开源并对公众开放。</summary>
<br>
发布了两个聊天模型，都是基于之前发布的两个基座模型；也发布了由 GPTQ 量化的两个8位模型和由 AWQ 量化的两个4位模型。

- `Yi-34B-Chat`
- `Yi-34B-Chat-4bits`
- `Yi-34B-Chat-8bits`
- `Yi-6B-Chat`
- `Yi-6B-Chat-4bits`
- `Yi-6B-Chat-8bits`

你可以访问以下链接进行试用。

- [Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
- [Replicate](https://replicate.com/01-ai)
</details>

<details>
<summary>🔔 <b>2023/11/23</b>: Yi系列模型社区许可协议更新至 v2.1 版本。</summary>
</details>

<details> 
<summary>🔥 <b>2023/11/08</b>: Yi-34B 聊天模型开始邀请测试。</summary>
<br>
参与测试申请表：

- [英文](https://cn.mikecrm.com/l91ODJf)
- [中文](https://cn.mikecrm.com/gnEZjiQ)

</details>

<details>
<summary>🎯 <b>2023/11/05</b>: <code>Yi-6B-200K</code> 和 <code>Yi-34B-200K</code> 的基座模型均已开源并对公众开放。 </summary>
<br>
发布了两个与之前发布参数规模相同的基座模型，只是上下文窗口扩展到了200K。

</details>

<details>
<summary>🎯 <b>2023/11/02</b>: <code>Yi-6B</code> 和 <code>Yi-34B</code> 的基座模型均已开源并对公众开放。</summary>
<br>
首次公开发布了两个双语（英语/中文）基座模型，参数规模分别为6B和34B。两者均以4K序列长度进行训练，并在推理时可扩展到32K。

</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

## 🎯 模型
Yi模型有多种参数规模，适用于不同的使用场景。你也可以对Yi模型进行微调，从而满足特定需求。

如果你想要部署Yi模型，则应确保软件和硬件满足[部署要求](#部署).

### 聊天模型

| 模型 | 下载 
|---|---
Yi-34B-Chat	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)
Yi-34B-Chat-4bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-4bits/summary)
Yi-34B-Chat-8bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-8bits/summary)
Yi-6B-Chat| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat/summary)
Yi-6B-Chat-4bits |	• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-4bits/summary)
Yi-6B-Chat-8bits	|  • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-8bits/summary)

<sub><sup> - 4-bit系列模型由AWQ量化。<br> - 8-bit系列模型由GPTQ量化。<br> - 所有量化模型都具有较低的使用门槛，因此它们可以在消费级GPU（例如3090、4090）上部署。</sup></sub>
### 基座模型

| 模型 | 下载 | 
|---|---|
Yi-34B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B/summary)
Yi-34B-200K|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-200K/summary)
Yi-6B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B/summary)
Yi-6B-200K	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-200K) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-200K/summary)

<sub><sup> - 200k 大约相当于 40 万个汉字。</sup></sub>

### 其他信息

- 聊天和基座模型：

  - 6B 系列的模型适合个人和学术使用。

  - 34B 系列的模型适合个人、学术和商业用途（特别是对于中小型企业）。这是一个性价比高的解决方案，价格合理，能力超出预期。

  - **默认的上下文窗口**是 **4k tokens**。

  - 预训练的 tokens 数量是 3T。

  - 训练数据截至 2023 年 6 月。

- 聊天模型
  
  <details style="display: inline;"><summary>关于聊天模型的局限性，见以下解释。 ⬇️</summary> 
   <ul>
   <br> <a href="https://01.ai/">01.AI</a> 发布的聊天模型在独家训练中采用了监督微调（SFT）技术。与其他标准聊天模型相比，<a href="https://01.ai/">01.AI</a> 的模型生成的回复更加多样化，因此适用于各种下游任务，比如创意场景。此外，回复更加多样化，有利于提高回复的质量，对后续的强化学习（RL）训练帮助很大。

    <br>需要注意的是，回复多样化也可能会导致某些已知问题更加严重，例如以下问题。
      <li>虚构：即模型可能会生成事实错误或不连贯的信息。模型回复多样化，更有可能出现虚构的现象，这些虚构的回复可能不是基于准确的数据或逻辑推理。</li>
      <li>重新生成的回复不一致：重新生成回复或者对回复进行采样时，结果可能出现前后不一致。多样性增多会导致即使在相似的输入条件下，结果也会存在差异。</li>
      <li>累积误差：当模型回复的错误随时间累积，就会出现累计误差的现象。模型回复的多样化增加了小误差积累成大错误的可能性，这种情况常见于扩展推理、解决数学问题等复杂任务中等。</li>
      <li>为了获得更连贯一致的回复，建议调整生成配置参数，如温度、top_p 或 top_k。这些调整既可以让模型的回复富有创意，又能保持逻辑上的连贯性。</li>
</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>


# 🟢 如何使用 Yi?
- [快速上手](#快速上手)
  - [选择路径](#选择路径)
  - [快速上手 - 使用 pip](#快速上手---pip)
  - [快速上手 - 使用 Docker](#快速上手---docker)
  - [快速上手 - 使用 conda-lock](#快速上手---conda-lock)
  - [快速上手 - 使用 llama.cpp](#快速上手---llamacpp)
  - [快速上手 - 使用网页演示](#快速上手---网页演示)
- [微调](#微调)
- [量化](#量化)
- [部署](#部署)
- [学习俱乐部](#学习俱乐部)

## 快速上手

 启用 Yi 模型非常简单，以下提供了多种路径供你选择。

### 选择路径

你可以根据你的需求，在下列路径中选择一条路径，开始你的 Yi 之旅～

 ![快速开始 - 选择路径](https://github.com/01-ai/Yi/blob/main/assets/img/quick_start_path.png?raw=true)

#### 🎯 在本地部署 Yi

如果你更喜欢在本地部署 Yi 模型，

  - 🙋‍♀️ 并且你有**足够**的资源（例如，NVIDIA A800 80GB），你可以从以下方法中选择一种方法：
    - [pip](#快速上手---pip)
    - [Docker](#快速上手---docker)
    - [conda-lock](#快速上手---conda-lock)

  - 🙋‍♀️ 但是你的资源很有限（例如，一台 MacBook Pro），你可以使用 [llama.cpp](#快速上手---llamacpp)

#### 🎯 不在本地部署 Yi 模型

如果你不想在本地部署 Yi 模型，你可以使用以下任何一种方式来探索 Yi 的功能。

##### 🙋‍♀️ 通过 API 来使用 Yi

如果你想探索 Yi 的更多功能，你可以从以下方法中选用一种方法。

- Yi APIs (Yi 官方)
  - [第一期访问活动](https://x.com/01AI_Yi/status/1735728934560600536?s=20)的部分申请者已经获得了访问权限。敬请期待下一轮访问申请的信息！ 

- [Yi APIs](https://replicate.com/01-ai/yi-34b-chat/api?tab=nodejs) (Replicate，第三方网站)

##### 🙋‍♀️ 在交互式平台使用 Yi

如果你想和 Yi 聊天，还想有更多可定制的选项（例如，系统提示、温度、重复惩罚等），你可以从以下选项中选用一种。
  
  - [Yi-34B-Chat-Playground](https://platform.lingyiwanwu.com/prompt/playground) (Yi 官方)
    - 如果你提交申请，加入了白名单，就可以使用官方的交互式平台。欢迎申请（填写[英文申请表](https://cn.mikecrm.com/l91ODJf) 或者[中文申请表](https://cn.mikecrm.com/gnEZjiQ)）。

  - [Yi-34B-Chat-Playground](https://replicate.com/01-ai/yi-34b-chat) (Replicate，第三方网站) 

##### 🙋‍♀️ 与 Yi 聊天

下列这些在线聊天服务提供了相似的用户体验，如果你想与 Yi 聊天，你可以任意选用其中一项。

- [Yi-34B-Chat](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)（Yi 在 Hugging Face 上的官方空间）
  - 不需要注册。

- [Yi-34B-Chat](https://platform.lingyiwanwu.com/)（Yi 官方 beta 版本）
  - 只要填写申请表，加入了白名单，就可以使用官方在线聊天服务。
    - 填写[英文申请表](https://cn.mikecrm.com/l91ODJf) 
    - 填写[中文申请表](https://cn.mikecrm.com/gnEZjiQ)

### 快速上手 - pip

本教程将指导你在 **A800（80G）上本地运行 Yi-34B-Chat**，并运行推理。

#### Step 0: 前提条件
 
- 确保安装了 Python 3.10 以上版本。

- 如果你想运行的不是 Yi-34B-Chat，而是其他 Yi 模型，请参阅[部署要求](#部署)。

#### Step 1: 准备你的环境 

运行下面的命令设置环境和安装所需要的软件包。

```bash
git clone https://github.com/01-ai/Yi.git
cd yi
pip install -r requirements.txt
```

#### Step 2: 下载 Yi 模型

你可以从以下来源下载 Yi 模型的权重和 tokenizer。

- [Hugging Face](https://huggingface.co/01-ai)
- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

#### Step 3: 运行推理

你可以使用 Yi 聊天模型或基座模型运行推理，如下所示。

##### 使用 Yi 聊天模型运行推理

1. 创建一个名为 `quick_start.py` 的文件，并将以下内容复制到其中。

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

2. 运行 `quick_start.py`.

    ```bash
    python quick_start.py
    ```

    然后你将看到一个类似于以下的回复。🥳

    ```bash
    Hello! How can I assist you today?
    ```

##### 使用 Yi 基座模型运行推理

步骤和[使用 Yi 聊天模型运行推理](#使用-yi-聊天模型运行推理)很相似。

你可以使用现有的文件 [`text_generation.py`](https://github.com/01-ai/Yi/tree/main/demo)。

```bash
python demo/text_generation.py  --model <your-model-path>
```

<details>

<summary> 然后你将看到一个类似于以下的回复。🥳 ⬇️ </summary>

<br>

**指令**: Let me tell you an interesting story about cat Tom and mouse Jerry,

**回复**: Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

</details>

### 快速上手 - Docker

<details>
<summary> 🚀 以下是如何在本地运行 Yi-34B-Chat 模型的详细教程。⬇️</summary>
<br>本教程将指导你如何在本地 A800 GPU 上运行 <strong>Yi-34B-Chat</strong> 模型，并运行推理。
<h4>步骤0: 准备工作</h4>
<p>确保你已经安装了 <a href="https://docs.docker.com/engine/install/?open_in_browser=true">Docker</a> 和 <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a>。</p>
<h4>步骤1: 启动 Docker 容器</h4>
<pre><code>docker run -it --gpus all \
-v &lt;your-model-path&gt;: /models
ghcr.io/01-ai/yi:latest
</code></pre>
<p>或者，你也可以从<code>registry.lingyiwanwu.com/ci/01-ai/yi:latest</code> 拉取已经构建好的 Yi Docker 镜像。</p>

<h4>步骤2: 运行推理</h4>
    <p>你可以使用 Yi 的聊天模型或基座模型来运行推理。</p>
    
<h5>使用 Yi 聊天模型运行推理</h5>
    <p>运行推理的步骤与使用<a href="#使用-yi-聊天模型运行推理"> pip 安装指南</a>类似。</p>
    <p><strong>注意</strong> 唯一不同的是你需要设置 <code>model_path = '&lt;your-model-mount-path&gt;'</code> 而不是 <code>model_path = '&lt;your-model-path&gt;'</code>。</p>
<h5>使用 Yi 基座模型运行推理</h5>
    <p>运行推理的步骤与使用<a href="#使用-yi-聊天模型运行推理"> pip 安装指南</a>类似。</p>
    <p><strong>注意</strong> 唯一不同的是你需要设置 <code>--model &lt;your-model-mount-path&gt;'</code> 而不是 <code>model &lt;your-model-path&gt;</code>。</p>
</details>

### 快速上手 - conda-lock

<details>
<summary> 🚀 如果你想创建一个可以完全重现的 conda 环境锁定文件，你可以使用 <code><a href="https://github.com/conda/conda-lock">conda-lock</a></code> 工具。 ⬇️</summary>
<br>
你可以参考  <a href="https://github.com/01-ai/Yi/blob/ebba23451d780f35e74a780987ad377553134f68/conda-lock.yml">conda-lock.yml</a> 文件，该文件包含了所需依赖项的具体版本信息。此外，你还可以使用<code><a href="https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html">micromamba</a></code>工具来安装这些依赖项。
<br>
安装这些依赖项的步骤，如下所示。

1. 根据<a href="https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html">指南</a>安装 "micromamba"。 

2. 运行命令 <code>micromamba install -y -n yi -f conda-lock.yml</code> ，创建一个名为<code>yi</code> conda 环境，并安装所需的依赖项。
</details>

### 快速上手 - llama.cpp
<details>
<summary> 🚀 以下是使用 llama.cpp 在本地运行 Yi-chat-6B-2bits 模型的详细教程。⬇️ </summary> 
<br>本教程分享如何在本地运行 <a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a> 量化模型，并且运行推理。</p>

- [步骤 0: 前提条件](#step-0-prerequisites)
- [步骤 1: 下载 llama.cpp](#step-1-download-llamacpp)
- [步骤 2: 下载 Yi 模型](#step-2-download-yi-model)
- [步骤 3: 运行推理](#step-3-perform-inference)

#### 步骤 0: 前提条件

- 该教程适用于 MacBook Pro（16GB 内存和 Apple M2 Pro 芯片）。

- 确保你的电脑上安装了 [`git-lfs`](https://git-lfs.com/) 。
  
#### 步骤 1: 下载 `llama.cpp`

克隆 [`llama.cpp`](https://github.com/ggerganov/llama.cpp) 仓库，运行以下命令。

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

#### 步骤 2: 下载 Yi 模型

步骤 2.1：仅下载 [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) 仓库的 pointers，运行以下命令。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

步骤 2.2：下载量化后的 Yi 模型 [yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf)，运行以下命令。

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

#### 步骤 3: 运行推理

如需体验 Yi 模型（运行模型推理），你可以选择以下任意一种方法。

- [方法 1：在终端中运行推理](#method-1-perform-inference-in-terminal)
  
- [方法 2：在网页上运行推理](#method-2-perform-inference-in-web)

##### 方法一：在终端中运行推理

本文使用 4 个线程编译 `llama.cpp` ，之后运行推理。在 `llama.cpp` 所在的目录，运行以下命令。

> ###### 提示
>
> - 将 `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` 替换为你的模型的实际路径。
>
> - 默认情况下，模型是续写模式。
> - 如需查看更多自定义选项（例如，系统提示、温度、重复惩罚等），运行 `./main -h` 查看详细使用说明。

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

恭喜你！你已经成功地向 Yi 模型提出了问题，得到了回复！🥳

##### 方法二：在网页上运行推理

1. 如果你想启用一个轻便敏捷的聊天机器人，可以运行以下命令。

    ```bash
    ./server --ctx-size 2048 --host 0.0.0.0 --n-gpu-layers 64 --model /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf
    ```

    你将会看到类似的输出。

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

2. 如果你想访问聊天机器人界面，可以打开网络浏览器，在地址栏中输入 `http://0.0.0.0:8080`。

    ![Yi模型聊天机器人界面 - LLaMA.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp1.png?raw=true)

3. 如果你在提示窗口中输入问题，例如，“如何喂养你的宠物狐狸？请用 6 个简单的步骤回答”，你就会收到回复。

    ![向 Yi 模型提问 - LLaMA.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp2.png?raw=true)

</ul>
</details>

### 快速上手 - 使用网页演示

你可以使用 Yi **聊天模型**（Yi-34B-Chat）创建网页演示。注意：Yi 基座模型（Yi-34B）不支持该功能。

[第一步：准备环境](#step-1-prepare-your-environment)

[第二步：下载模型](#step-2-download-the-yi-model)

第三步：启动网页服务，运行以下命令。

```bash
python demo/web_demo.py -c <你的模型路径>
```

命令运行完毕后，你可以在浏览器中输入控制台提供的网址，来使用网页演示功能。

 ![快速上手 - 网页演示](https://github.com/01-ai/Yi/blob/main/assets/img/yi_34b_chat_web_demo.gif?raw=true)

### 微调

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

完成后，你可以使用以下命令，比较微调后的模型与基座模型。

```bash
bash finetune/scripts/run_eval.sh
```
<details style="display: inline;"><summary> 你可以使用 Yi 6B 和 34B 基座模型的微调代码，根据你的自定义数据进行微调。 ⬇️</summary> <ul>

#### 准备工作

###### 从镜像开始

默认情况下，我们使用来自[BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) 的小型数据集来微调基座模型。
你还可以按照以下 `jsonl` 格式准备自定义数据集。

```json
{ "prompt": "Human: Who are you? Assistant:", "chosen": "I'm Yi." }
```
然后将自定义数据集挂载到容器中，替换默认数据。

```bash
docker run -it \
    -v /path/to/save/finetuned/model/:/finetuned-model \
    -v /path/to/train.jsonl:/yi/finetune/data/train.json \
    -v /path/to/eval.jsonl:/yi/finetune/data/eval.json \
    ghcr.io/01-ai/yi:latest \
    bash finetune/scripts/run_sft_Yi_6b.sh
```

###### 从本地服务器开始

确保你已经安装了 conda。如果还没安装，可以运行以下命令。

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

然后，创建一个 conda 环境。

```bash
conda create -n dev_env python=3.10 -y
conda activate dev_env
pip install torch==2.0.1 deepspeed==0.10 tensorboard transformers datasets sentencepiece accelerate ray==2.7
```

##### 配备硬件

如果你想使用 Yi-6B 模型，建议使用具有 4 个 GPU 的节点，每个 GPU 内存大于 60GB。

如果你想使用 Yi-34B 模型，注意此模式采用零卸载技术，占用了大量 CPU 内存，因此需要限制 34B 微调训练中的 GPU 数量。你可以使用 CUDA_VISIBLE_DEVICES 限制 GPU 数量（如 scripts/run_sft_Yi_34b.sh 中所示）。

用于微调 34B 模型的常用硬件具有 8 个 GPU 的节点（通过CUDA_VISIBLE_DEVICES=0,1,2,3 在运行中限制为4个 GPU），每个 GPU 的内存大于 80GB，总 CPU 内存大于900GB。

#### 快速上手

将 LLM-base 模型下载到 MODEL_PATH（6B 和 34B）。模型常见的文件夹结构如下。

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

将数据集从 huggingface 下载到本地存储 DATA_PATH，例如 Dahoas/rm-static。

```bash
|-- $DATA_PATH
|   |-- data
|   |   |-- train-00000-of-00001-2a1df75c6bce91ab.parquet
|   |   |-- test-00000-of-00001-8c7c51afc6d45980.parquet
|   |-- dataset_infos.json
|   |-- README.md
```

`finetune/yi_example_dataset` 中有示例数据集，这些数据集是从 [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)修改而来。

```bash
|-- $DATA_PATH
    |--data
        |-- train.jsonl
        |-- eval.jsonl
```

`cd` 进入 scripts 文件夹，复制并粘贴脚本，然后运行。你可以使用以下代码完成此项。

```bash
cd finetune/scripts

bash run_sft_Yi_6b.sh
```

对于 Yi-6B 基座模型，设置 training_debug_steps=20 和 num_train_epochs=4， 就可以输出一个聊天模型，大约需要 20 分钟。

对于 Yi-34B 基座模型，初始化时间相对较长，请耐心等待。

#### 评估

```bash
cd finetune/scripts

bash run_eval.sh
```

然后，你将看到基座模型和微调模型的回复。
</ul>
</details>

### 量化

#### GPT-Q 量化
```bash
python quantization/gptq/quant_autogptq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

完成后，你可以用以下代码对生成的模型进行评估。

```bash
python quantization/gptq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

<details style="display: inline;"><summary> 以下是量化详细的过程。 ⬇️</summary> <ul>

[GPT-Q](https://github.com/IST-DASLab/gptq)是一种后训练量化方法，能够帮助大型语言模型在使用时节省内存，保持模型的准确性，可以加快模型的运行速度。

使用以下教程，对 Yi 模型进行 GPT-Q 量化，毫不费力。

要运行 GPT-Q，你需要用到[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 和
[exllama](https://github.com/turboderp/exllama).
此外，huggingface transformers 已经集成了 optimum 和 auto-gptq，能够实现语言模型的 GPT-Q 量化。

##### 运行量化

为了运行 GPT-Q 量化，你可以使用提供的 `quant_autogptq.py` 脚本。

```bash
python quant_autogptq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### 运行量化模型

你可以使用`eval_quantized_model.py`来运行量化模型。

```bash
python eval_quantized_model.py --model /quantized_model --trust_remote_code
```
</ul>
</details>

#### AWQ 量化
```bash
python quantization/awq/quant_autoawq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

完成后，你可以使用以下脚本对生成的模型进行评估。

```bash
python quantization/awq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```
<details style="display: inline;"><summary> 关于 AWQ 量化的细节，见以下内容。⬇️</summary> <ul>

[AWQ](https://github.com/mit-han-lab/llm-awq)是一种用于大型语言模型（LLMs）的后训练量化方法，可以将模型的权重数据高效准确地转化成低位数据（比如INT3或INT4），因此可以减小模型在内存中的占用空间，保持模型的准确性。

使用以下教程，对 Yi 模型进行 AWQ 量化，毫不费力。

要运行 AWQ，我们会用到 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

##### 运行量化

你可以使用 `quant_autoawq.py` 脚本运行 AWQ 量化。

```bash
python quant_autoawq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### 运行量化模型

你可以使用 `eval_quantized_model.py`脚本来运行量化后的模型。

```bash
python eval_quantized_model.py --model /quantized_model --trust_remote_code
```


</ul>
</details>
<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 部署

如果你想部署 Yi 模型，确保满足以下软件和硬件要求。

#### 软件要求

在使用 Yi 量化模型之前，确保已经安装以下列出的软件。

| 模型 | 软件 |
|:---|:---|
Yi 4-bit quantized models | [AWQ and CUDA](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install-from-pypi)
Yi 8-bit quantized models |  [GPTQ and CUDA](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#quick-installation)

#### 硬件要求

在环境中部署 Yi 之前，确保硬件满足以下要求。

##### 聊天模型

| 模型                 | 最低显存      | 推荐GPU示例                             |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | RTX 3090 <br> RTX 4090 <br>  A10 <br> A30             |
| Yi-6B-Chat-4bits     | 4 GB          | RTX 3060 <br>  RTX 4060                     |
| Yi-6B-Chat-8bits     | 8 GB          | RTX 3070 <br> RTX 4060                     |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 <br> A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | RTX 3090  <br> RTX 4090 <br> A10 <br> A30 <br> A100 (40GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090  <br> 2 x RTX 4090 <br> A800  (40GB) |

以下是不同批量使用情况下的详细最低显存要求。

|  模型                  | batch=1 | batch=4 | batch=16 | batch=32 |
| :----------------------- | :------- | :------- | :-------- | :-------- |
| Yi-6B-Chat              | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-4bits  | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-6B-Chat-8bits  | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-34B-Chat       | 65 GB   | 68 GB   | 76 GB    | > 80 GB   |
| Yi-34B-Chat-4bits | 19 GB   | 20 GB   | 30 GB    | 40 GB    |
| Yi-34B-Chat-8bits | 35 GB   | 37 GB   | 46 GB    | 58 GB    |

##### 基座模型

|模型                   |最低显存      |        推荐GPU示例                     |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B                | 15 GB         | RTX3090 <br> RTX4090 <br> A10 <br> A30               |
| Yi-6B-200K           | 50 GB         | A800 (80 GB)                            |
| Yi-34B               | 72 GB         | 4 x RTX 4090 <br> A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |

### 学习俱乐部

<details>
<summary> 如果你想学习如何使用 Yi 系列模型，这里有大量的学习资源供你选择。 ⬇️</summary>
<br>

欢迎来到 Yi 学习俱乐部！

无论你是经验丰富的开发者还是新手，你都可以在这里找到大量有用的学习资源，更加了解 Yi 模型，增强相关技能。在这里，你可以学习见解深刻的博客文章、深度全面的视频教程以及实践指南等精彩内容。

在这里，知识渊博的 Yi 专家和热情的爱好者慷慨分享了许多深度内容。我们对各位小伙伴宝贵的贡献表示衷心的感谢！

在此，我们也热烈邀请你加入我们，为 Yi 做出贡献。如果你已经对 Yi 做出了贡献，不要犹豫，在下面的表格中展示你杰出的工作。

有了这些唾手可得的资源，你就可以即刻踏上 Yi 学习之旅啦～祝学习愉快！🥳

#### 教程


| 类型        | 教程地址                                            |      日期      |     作者     |
|:-------------|:--------------------------------------------------------|:----------------|:----------------|
| 博客        | [本地运行零一万物 34B 大模型，使用 LLaMA.cpp & 21G 显存](https://zhuanlan.zhihu.com/p/668921042)                  |  2023-11-26  |  [苏洋](https://github.com/soulteary)  |
| 博客        | [Running Yi-34B-Chat locally using LlamaEdge](https://www.secondstate.io/articles/yi-34b/)                   |  2023-11-30  |  [Second State](https://github.com/second-state)  |
| 博客       | [零一万物模型折腾笔记：官方 Yi-34B 模型基础使用](https://zhuanlan.zhihu.com/p/671387298)                           | 2023-12-10 |  [苏洋](https://github.com/soulteary)  |
| 博客        | [CPU 混合推理，非常见大模型量化方案：“二三五六” 位量化方案](https://zhuanlan.zhihu.com/p/671698216)                  | 2023-12-12 |  [苏洋](https://github.com/soulteary)  |
| 博客        | [零一万物开源Yi-VL多模态大模型，魔搭社区推理&微调最佳实践来啦！](https://zhuanlan.zhihu.com/p/680098411)                  | 2024-01-26 |  [ModelScope](https://github.com/modelscope)  |
| 视频       | [只需 24G 显存，用 vllm 跑起来 Yi-34B 中英双语大模型](https://www.bilibili.com/video/BV17t4y1f7Ee/)               | 2023-12-28 |  漆妮妮  |
| 视频       | [Install Yi 34B Locally - Chinese English Bilingual LLM](https://www.youtube.com/watch?v=CVQvj4Wrh4w&t=476s) | 2023-11-05  |  Fahd Mirza  |
</details>

# 🟢 为什么选择 Yi？

  - [🌎 生态系统](#-生态系统)
    - [💦 上游](#-上游)
    - [🌊 下游](#-下游)
      - [🔗 服务](#-服务)
      - [⚙️ 量化](#️-量化)
      - [🛠️ 微调](#️-微调)
      - [API](#api)
  - [📌 基准测试](#-基准测试)
    - [📊 聊天模型性能](#-聊天模型性能)
    - [📊 基座模型性能](#-基座模型性能)

## 🌎 生态系统

Yi 拥有一个全面的生态系统，为你提供一系列工具、服务和模型，你将获得丰富的体验，最大程度提升工作工作效率。

- [💦 上游](#-上游)
- [🌊 下游](#-下游)
  - [🔗 服务](#-服务)
  - [⚙️ 量化](#️-量化)
  - [🛠️ 微调](#️-微调)
  - [API](#api)

### 💦 上游

Yi 系列模型遵循与Llama相同的模型架构。选择 Yi，你可以利用Llama生态系统中现有的工具、库和资源，无需创建新工具，提高开发效率。

例如，Yi 系列模型以Llama模型的格式保存。你可以直接使用`LlamaForCausalLM`和`LlamaTokenizer`加载模型。更多信息，详见[使用聊天模型](#31-使用聊天模型)。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34b", device_map="auto")
```

### 🌊 下游

> 💡 提示
> 
> - 随时创建PR，分享你使用 Yi 系列模型构建的出色作品。
>
> - 为了帮助他人快速理解你的工作，建议使用`<模型名称>: <模型简介> + <模型亮点>`的格式。

#### 🔗 服务

如果你想在几分钟内开始使用 Yi，你可以使用以下基于 Yi 构建的服务。

- Yi-34B-Chat：你可以通过以下平台与 Yi 聊天。
  - [Yi-34B-Chat | Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
  - [Yi-34B-Chat | Yi Platform](https://platform.lingyiwanwu.com/)：**注意**目前只有加入了我们的白名单，才可以使用此平台。欢迎你申请（填写[英文申请表](https://cn.mikecrm.com/l91ODJf)或[中文申请表](https://cn.mikecrm.com/gnEZjiQ)）加入，亲身体验！

- [Yi-6B-Chat (Replicate)](https://replicate.com/01-ai)：你可以通过设置更多的参数，调用 APIs 使用此模型，这里有更多选项。

- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models)：你可以使用这项服务，在本地运行Yi模型，更灵活，可以根据自己的偏好进行个性化调整。

#### ⚙️ 量化

如果你的计算能力有限，你可以使用 Yi 的量化模型，如下所示。

这些量化模型虽然精度降低，但提供了更高的效率，推理速度更快，RAM 使用量更小。

- [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ)
- [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
- [TheBloke/Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)

#### 🛠️ 微调

如果你希望探索 Yi 庞大家族中的多样化能力，你可以深入了解下面的 Yi 微调模型。

- [TheBloke 模型](https://huggingface.co/TheBloke)：这个网站提供很多微调模型，这些微调模型来源于 Yi 等大型语言模型（LLMs）。
  
  以下是 Yi 的微调模型，根据下载量排序，但这不是 Yi 的全部内容。
  - [TheBloke/dolphin-2_2-yi-34b-AWQ](https://huggingface.co/TheBloke/dolphin-2_2-yi-34b-AWQ)
  - [TheBloke/Yi-34B-Chat-AWQ](https://huggingface.co/TheBloke/Yi-34B-Chat-AWQ)
  - [TheBloke/Yi-34B-Chat-GPTQ](https://huggingface.co/TheBloke/Yi-34B-Chat-GPTQ)
  
- [SUSTech/SUS-Chat-34B](https://huggingface.co/SUSTech/SUS-Chat-34B)：这个模型在所有 70B 以下的模型中排名第一，并且超越了体量是其两倍的deepseek-llm-67b-chat。你可以在[开放 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)上查看结果。
  
- [OrionStarAI/OrionStar-Yi-34B-Chat-Llama](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat-Llama)：这个模型在C-Eval和CMMLU评估中超越了其他模型（如 GPT-4, Qwen-14B-Chat, Baichuan2-13B-Chat）, 在 [OpenCompass LLM 排行榜](https://opencompass.org.cn/leaderboard-llm) 上表现出色。
  
- [NousResearch/Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B)：这个模型在Capybara数据集上使用200K上下文长度和3个训练周期进行训练。

#### API

- [amazing-openai-api](https://github.com/soulteary/amazing-openai-api)：这个工具可以将 Yi 模型API转换成OpenAI API格式。
- [LlamaEdge](https://www.secondstate.io/articles/yi-34b/#create-an-openai-compatible-api-service-for-the-yi-34b-chat-model)：这个工具使用可移植的 Wasm（WebAssembly）文件构建了一个与 OpenAI 兼容的 API 服务器，用于 Yi-34B-Chat，由 Rust 驱动。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

## 📌 基准测试 

- [📊 聊天模型性能](#-聊天模型性能)
- [📊 基座模型性能](#-基座模型性能)

### 📊 聊天模型性能

Yi-34B-Chat 模型在 MMLU、CMMLU、BBH、GSM8k 等所有开源模型的基准测试中表现出色，排名第一。
![Chat model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_chat.png?raw=true) 

<details>
<summary> 评估方法与挑战 ⬇️ </summary>

- **评估方式**: 我们使用零样本（zero-shot）和少样本（few-shot）方法评估了除 TruthfulQA 以外外的各种基准。
- **零样本与少样本**: 大部分聊天模型常用零样本的方式。
- **评估策略**: 我们的评估策略是让模型在明确或隐含地遵循指令（例如使用少量样本示例）的同时生成回应，并从其生成的文本中提取相关答案。
- **面临的挑战**: 一些模型不适用少数数据集中的指令，无法按照所要求的特定格式产生输出。这会导致结果不理想。

<strong>*</strong>: C-Eval 的结果来源于验证数据集。
</details>

### 📊 基座模型性能

Yi-34B 和 Yi-34B-200K 模型作为开源模型中的佼佼者脱颖而出，尤其在 MMLU、CMMLU、常识推理、阅读理解等方面表现卓越。
![Base model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_base.png?raw=true)

<details>
<summary> 评估方法 ⬇️</summary>

- **结果差异**: 在测试开源模型时，我们的流程与公共来源（如 OpenCompass）报告的结果之间存在差异。
- **调查发现**: 深入调查显示，各种模型在提示语、后处理策略和采样技术上的变化可能导致各种模型的结果产生显著差异。
- **统一的基准测试过程**: 我们的方法论与原始基准一致，即在评估时使用相同的提示语和后处理策略，并在评估时应用贪心解码（greedy decoding），不对生成内容进行任何后处理。
- **努力检索未报告的评分**: 对于原始作者未报告的分数（包括以不同设置报告的分数），我们尝试使用我们的流程获取结果。
- **广泛的模型评估**: 为了全面评估模型的能力，我们采用了在 Llama2 中概述的方法论。具体来说，我们包括了 PIQA、SIQA、HellaSwag、WinoGrande、ARC、OBQA 和 CSQA 来评估常识推理。SquAD、QuAC 和 BoolQ 被纳入以评估阅读理解。
- **特殊配置**: CSQA 专门使用7-样本（7-shot）设置进行测试，而所有其他测试都使用0-样本（0-shot）配置进行。此外，我们在“数学和编码”类别下引入了 GSM8K（8-shot@1）、MATH（4-shot@1）、HumanEval（0-shot@1）和 MBPP（3-shot@1）。
- **Falcon-180B 注意事项**: 由于技术限制，Falcon-180B 没有在 QuAC 和 OBQA 上进行测试。其性能分数是从其他任务中得出的平均值，并且考虑到这两个任务通常的分数较低，Falcon-180B 的能力大概率不会被低估。
</details>

# 🟢 谁可以使用 Yi？

答案是所有人! 🙌 ✅

- Yi 系列模型可免费用于个人使用、学术目的和商业用途。所有使用必须遵守[《Yi系列模型社区许可协议 2.1》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)。
  
- Yi 可以免费商用——你只需要[填写这份表单](https://www.lingyiwanwu.com/yi-license)，就可以获得Yi系列模型的商业许可证。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

# 🟢 其他内容

### 致谢

我们对每位火炬手都深表感激，感谢你们为 Yi 社区所做的贡献。因为有你们，Yi 成为了一个项目，也成为了一个充满活力的创新社区。我们由衷地感谢各位小伙伴！

[![yi contributors](https://contrib.rocks/image?repo=01-ai/yi&max=2000&columns=15)](https://github.com/01-ai/yi/graphs/contributors)

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 📡 免责声明

在训练过程中，我们使用数据合规性检查算法，最大程度地确保训练模型的合规性。由于数据复杂且语言模型使用场景多样，我们无法保证模型在所有场景下均能生成正确合理的回复。请注意，模型仍可能生成有误的回复。对于任何因误用、误导、非法使用、错误使用导致的风险和问题，以及与之相关的数据安全问题，我们均不承担责任。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 🪪 许可证

本仓库中的源代码遵循 [Apache 2.0 许可证](https://github.com/01-ai/Yi/blob/main/LICENSE)。Yi 系列模型完全开放，你可以免费用于学术研究和商业用途。如需商用，你仅需[提交申请](https://www.lingyiwanwu.com/yi-license)，即能立刻自动获取商用许可，而无需等待官方审批。所有使用必须遵守[《Yi系列模型社区许可协议 2.1》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>



