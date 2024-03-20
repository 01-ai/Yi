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
  <h3 align="center">打造下一代开源双语大语言模型</h3>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/01-ai" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/01ai/" target="_blank">魔搭 ModelScope</a> • ✡️ <a href="https://wisemodel.cn/organization/01.AI" target="_blank">始智 WiseModel</a>
</p> 

<p align="center">
    👩‍🚀 欢迎来 <a href="https://github.com/01-ai/Yi/discussions" target="_blank"> GitHub Discussions</a> 讨论问题
</p> 
<p align="center">
    👋 欢迎加入<a href="https://discord.gg/hYUwWddeAu" target="_blank"> 👾 Discord </a> 或者 💬 <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245" target="_blank"> 微信群 </a>一起交流
</p> 

<p align="center">
    📝 欢迎查阅<a href="https://arxiv.org/abs/2403.04652"> Yi 技术报告 </a>了解更多
</p> 

<p align="center">
    📚 欢迎来 <a href="#学习中心"> Yi 学习中心 </a>探索新知
</p> 


<hr>

<ul>
  <li>🙌 本文由 Yi 和<a href="#本文贡献者">社区志愿者</a>共同翻译完成，感谢每一位传递知识的<a href="#致谢">火炬手</a>。</li> 

  <li>🤗 欢迎大家<a href="https://github.com/01-ai/Yi/discussions/314">加入「Yi 起翻译」</a>，开启知识之火旅程，共绘技术内容图谱。</li>
  
  <li>📝 本文翻译使用了 <a href="https://huggingface.co/spaces/01-ai/Yi-34B-Chat">Yi-34B-Chat</a>，关于翻译时使用的 prompt 及最佳实践，参阅<a href="https://github.com/01-ai/Yi/wiki/%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AE%A1%E6%A0%A1%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF#%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8-prompt-%E6%9D%A5%E5%AE%9E%E7%8E%B0%E9%AB%98%E8%B4%A8%E9%87%8F%E7%BF%BB%E8%AF%91">「如何使用 Prompt 来实现高质量翻译」</a>和<a href="https://github.com/01-ai/Yi/wiki/%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AE%A1%E6%A0%A1%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF">「翻译与审校的正确姿势」</a>。</li>
</ul>


<!-- DO NOT REMOVE ME -->

<hr>
<details open>
<summary></b>📕 目录</b></summary>

- [📌 Yi 是什么?](#-yi-是什么)
  - [介绍](#介绍)
  - [模型](#模型)
    - [Chat 模型](#chat-模型)
    - [Base 模型](#base-模型)
    - [其它信息](#其它信息)
  - [最新动态](#最新动态)
- [📌 如何使用 Yi?](#-如何使用-yi)
  - [快速上手](#快速上手)
    - [选择学习路径](#选择学习路径)
    - [快速上手 - 使用 PyPi (pip install)](#快速上手---pypi-pip-install)
    - [快速上手 - 使用 Docker](#快速上手---docker)
    - [快速上手 - 使用 conda-lock](#快速上手---conda-lock)
    - [快速上手 - 使用 llama.cpp](#快速上手---llamacpp)
    - [快速上手 - 使用 Web demo](#快速上手---使用-web-demo)
  - [微调](#微调)
  - [量化](#量化)
  - [部署](#部署)
  - [学习中心](#学习中心)
- [📌 为什么选择Yi？](#-为什么选择-yi)
  - [生态](#生态)
    - [上游](#上游)
    - [下游](#下游)
      - [服务](#下游---服务)
      - [量化](#下游---量化)
      - [微调](#下游---微调)
      - [API](#下游---api)
  - [基准测试](#基准测试)
    - [Chat 模型性能](#chat-模型性能)
    - [Base 模型性能](#base-模型性能)
  - [技术报告](#技术报告)
    - [引用](#引用)
- [📌 谁可以使用 Yi？](#-谁可以使用-yi)
- [📌 其它](#-其它)
  - [致谢](#致谢)
  - [免责声明](#免责声明)
  - [许可证](#许可证)

</details>

<hr>

# 📌 Yi 是什么?

## 介绍

- 🤖 Yi 系列模型是 [01.AI](https://01.ai/) 从零训练的下一代开源大语言模型。

- 🙌 Yi 系列模型是一个双语语言模型，在 3T 多语言语料库上训练而成，是全球最强大的大语言模型之一。Yi 系列模型在语言认知、常识推理、阅读理解等方面表现优异。例如，

   - Yi-34B-Chat 模型在 AlpacaEval Leaderboard [排名第二](https://twitter.com/01AI_Yi/status/1745371506623103087?s=20)，**仅次于 GPT-4 Turbo**，超过了 GPT-4、Mixtral 和 Claude 等大语言模型（数据截止至 2024 年 1 月）。

  - Yi-34B 模型在 [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)（预训练）与 C-Eval 基准测试中[荣登榜首](https://mp.weixin.qq.com/s/tLP-fjwYHcXVLqDcrXva2g)，**在中文和英文语言能力方面**均超过了其它开源模型，例如，Falcon-180B、Llama-70B 和 Claude（数据截止至 2023 年 11 月）。

  - 🙏 （致谢 Llama ）感谢 Transformer 和 Llama 开源社区，不仅简化了开发者从零开始构建大模型的工作，开发者还可以利用 Llama 生态中现有的工具、库和资源，提高开发效率。

  <details style="display: inline;"><summary> 如果你对 Yi 使用 Llama 架构及其许可使用政策感兴趣，参阅 <span style="color:  green;">「Yi 与 Llama 的关系」。</span> ⬇️</summary> <ul> <br>

> 💡 简短总结
> 
> Yi 系列模型采用与Llama相同的模型架构，但它们**不是** Llama 的衍生品。


- Yi 和 Llama 都是基于 Transformer 结构。实际上，自 2018 年以来，Transformer 一直是大语言模型的常用架构。

- 在 Transformer 架构的基础上，Llama 凭借出色的稳定性、可靠的收敛性和强大的兼容性，成为大多数先进开源模型的基石。因此，Llama 也成为 Yi 等模型的基础框架。

- 得益于 Transformer 和 Llama 架构，各类模型可以简化从零开始构建模型的工作，并能够在各自的生态中使用相同的工具。

- 然而，Yi 系列模型不是 Llama 的衍生品，因为它们不使用 Llama 的权重。

  - 虽然大多数开源模型都采用了 Llama 的架构，但决定模型表现的关键因素是训练所使用的数据集、训练管道及其基础设施。

  - [01.AI](https://01.ai/) 用独特的方式开发了 Yi 系列模型，从零开始创建了自己的高质量训练数据集、高效的训练流水线和强大的训练基础设施，因此 Yi 系列模型性能优异，在 [Alpaca Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) 上排名仅次于 GPT-4，超过了 Llama（数据截止至 2023 年 12 月）。
</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>


## 最新动态

<details>
<summary>🎯 <b>2023-03-16</b>：发布并开源了 <code>Yi-9B-200K</code> 模型。</summary>
</details>

<details open>
  <summary>🎯 <b>2024-03-08</b>: 发布了 <a href="https://arxiv.org/abs/2403.04652">Yi 技术报告</a>！</summary>
  <br>

<details open>
  <summary>🔔 <b>2024-03-07</b>: 增强了 Yi-34B-200K 长文本记忆和检索能力。</summary>
  <br>
Yi-34B-200K 的“大海捞针”能力增强了 10.5%, 从 89.3% 提升到了 99.8%。
在 5B tokens 的长文本数据集上，对模型进行继续预训练，模型性能达到预期目标。

</details>
<br>
<details open>
  <summary>🎯 <b>2024-03-06</b>: 发布并开源了 <code>Yi-9B</code> 模型。</summary>
  <br>
<code>Yi-9B</code> 模型在 Mistral-7B、SOLAR-10.7B、Gemma-7B、DeepSeek-Coder-7B-Base-v1.5 等相近尺寸的模型中名列前茅，具有出色的代码能力、数学能力、常识推理能力以及阅读理解能力。
</details>
<br>
<details open>
  <summary>🎯 <b> 2024-01-23</b>: 发布并开源了 <code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code> 和 <code><a href="https://huggingface.co/01-ai/Yi-VL-6B">Yi-VL-6B</a></code> 多模态语言大模型。</summary>
  <br>
   <code><a href="https://huggingface.co/01-ai/Yi-VL-34B">Yi-VL-34B</a></code>在 <a href="https://arxiv.org/abs/2311.16502">MMMU</a> 和 <a href="https://arxiv.org/abs/2401.11944">CMMMU</a> 最新的基准测试中荣登榜首（数据截止至 2024 年 1 月）。</li>
</details>
<br>
<details>
<summary>🎯 <b>2023-11-23</b>: 发布并开源了六大 Chat 模型。</summary>
<br>
其中，两个 4-bits 模型由 AWQ 量化，两个 8-bits 模型由 GPTQ 量化。

- `Yi-34B-Chat`
- `Yi-34B-Chat-4bits`
- `Yi-34B-Chat-8bits`
- `Yi-6B-Chat`
- `Yi-6B-Chat-4bits`
- `Yi-6B-Chat-8bits`

</details>

<details>
<summary>🔔 <b>2023-11-23</b>： Yi 系列模型社区许可协议更新至 <a href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">2.1 版本</a>。</summary>
</details>

<details>  
<summary>🔥 <b>2023-11-08</b>： Yi-34B-Chat 模型开始邀请测试。</summary>
<br>

如需申请测试，你可以填写[英文](https://cn.mikecrm.com/l91ODJf)或[中文](https://cn.mikecrm.com/gnEZjiQ)申请表。

</details>

<details>
<summary>🎯 <b>2023-11-05</b>： 发布并开源了 <code>Yi-6B-200K</code> 和 <code>Yi-34B-200K</code> Base 模型。 </summary>
<br>
这两个 Base 模型与之前发布的 Base 模型的参数规模相同，并且上下文窗口扩展到了 200K。

</details>

<details>
<summary>🎯 <b>2023-11-02</b>： 发布并开源了 <code>Yi-6B-Base</code> 和 <code>Yi-34B-Base</code> 模型。</summary>
<br>
首次发布并开源了两个 Base 模型（支持中英双语），参数规模分别为 6B 和 34B。两者均以 4K 序列长度进行训练，在推理时可扩展到 32K。

</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

## 模型
Yi 系列模型有多种参数规模，适用于不同的使用场景。你也可以对Yi模型进行微调，从而满足特定需求。

如需部署 Yi 系列模型，应确保软件和硬件满足「[部署要求](#部署)」.

### Chat 模型

| 模型 | 下载 
|---|---
Yi-34B-Chat	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)
Yi-34B-Chat-4bits	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-4bits/summary)
Yi-34B-Chat-8bits | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-Chat-8bits/summary)
Yi-6B-Chat| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat/summary)
Yi-6B-Chat-4bits |	• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-4bits/summary)
Yi-6B-Chat-8bits	|  • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-Chat-8bits) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-Chat-8bits/summary)

<sub><sup> - 4-bits 系列模型由AWQ量化。<br> - 8-bits 系列模型由GPTQ量化。<br> - 所有量化模型的使用门槛较低，因此可以在消费级GPU（例如，3090、4090）上部署。</sup></sub>

### Base 模型

| 模型 | 下载 | 
|---|---|
Yi-34B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B/summary)
Yi-34B-200K|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-34B-200K/summary)
Yi-9B|• [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B) • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-9B)
Yi-9B-200K | • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-9B-200K)   • [🤖 ModelScope](https://wisemodel.cn/models/01.AI/Yi-9B-200K)
Yi-6B| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B)  • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B/summary)
Yi-6B-200K	| • [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-6B-200K) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-6B-200K/summary)

<sub><sup> - 200K 大约相当于 40 万个汉字。<br> - 如果你想用 Yi-34B-200K 更早的版本 （即 2023 年 11 月 5 日发布的版本），可以运行代码 `git checkout 069cd341d60f4ce4b07ec394e82b79e94f656cf`，下载权重。</sup></sub>


### 模型信息

- For chat and base models

Model | Intro | 默认的上下文窗口 | 预训练的 tokens 数量 | 训练数据
|---|---|---|---|---
6B 系列模型 |适合个人和学术使用。| 4K | 3T | 截至 2023 年 6 月。
9B 模型| 是 Yi 系列模型中代码和数学能力最强的模型。|4K | Yi-9B 是在 Yi-6B 的基础上，使用了 0.8T tokens 进行继续训练。| 截至 2023 年 6 月。
34B 系列模型 | 适合个人、学术和商业用途（尤其对中小型企业友好）。<br>34B 模型尺寸在开源社区属于稀缺的“黄金比例”尺寸，已具大模型涌现能力，适合发挥于多元场景，满足开源社区的刚性需求。|4K | 3T | 截至 2023 年 6 月。

- Chat 模型
  
  <details style="display: inline;"><summary>关于 Chat 模型的局限性，参阅以下解释。 ⬇️</summary> 
   <ul>
   <br> Chat 模型在训练中采用了监督微调（SFT）技术。与其它常规 Chat 模型相比， Yi 系列模型生成的回复更加多样化，（1）因此适用于各种下游任务，例如，创意场景；（2）有利于提高回复的质量，对后续的强化学习（RL）训练帮助很大。

    <br>注意，回复多样化也可能会导致某些已知问题更加严重，例如，
      <li>幻觉：即模型可能会生成错误或不连贯的信息。模型回复多样化，更有可能出现幻觉，这些幻觉可能不是基于准确的数据或逻辑推理。</li>
      <li>重新生成的回复不一致：重新生成回复或者对回复进行采样时，结果可能出现前后不一致。多样性增多会导致即使在相似的输入条件下，结果也会存在差异。</li>
      <li>累积误差：当模型回复的错误随时间累积，就会出现累计误差的现象。模型回复的多样化增加了小误差积累成大错误的可能性，这种情况常见于扩展推理、解决数学问题等复杂任务中。</li>
      <li>为了获得更连贯一致的回复，建议调整生成配置参数，例如，温度、top_p 和 top_k。这些调整既可以让模型的回复富有创意，又能保持逻辑上的连贯性。</li>
</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>


# 📌 如何使用 Yi?
- [快速上手](#快速上手)
  - [选择学习路径](#选择学习路径)
  - [快速上手 - PyPi (pip install)](#快速上手---pypi-pip-install)
  - [快速上手 - 使用 Docker](#快速上手---docker)
  - [快速上手 - 使用 conda-lock](#快速上手---conda-lock)
  - [快速上手 - 使用 llama.cpp](#快速上手---llamacpp)
  - [快速上手 - 使用 Web demo](#快速上手---使用-web-demo)
- [微调](#微调)
- [量化](#量化)
- [部署](#部署)
- [学习中心](#学习中心)

## 快速上手

 你可以选择一条学习路径，开始使用 Yi 系列模型。

### 选择学习路径

你可以根据自身需求，选择以下方式之一，开始你的 Yi 之旅。

 ![选择学习路径](https://github.com/01-ai/Yi/blob/main/assets/img/quick_start_path_CN.png?raw=true)

#### 🎯 在本地部署 Yi

如果你想在本地部署 Yi 模型，

  - 🙋‍♀️ 并且你有**足够**的资源（例如，NVIDIA A800 80GB），你可以选择以下方式之一。
    - [pip](#快速上手---pypi-pip-install)
    - [Docker](#快速上手---docker)
    - [conda-lock](#快速上手---conda-lock)
  - 🙋‍♀️ 但你的资源有限（例如，一台 MacBook Pro），你可以使用 [llama.cpp](#快速上手---llamacpp)。

#### 🎯 不在本地部署 Yi 模型

如果你不想在本地部署 Yi 模型，你可以选择以下方式之一。

##### 🙋‍♀️ 使用 Yi API

如果你想探索 Yi 的更多功能，你可以选择以下方式之一。

- Yi APIs （Yi 官方）
  - [部分申请者](https://x.com/01AI_Yi/status/1735728934560600536?s=20)已获取 Yi API keys。Yi 将开放更多 API keys，敬请期待。

- [Yi APIs](https://replicate.com/01-ai/yi-34b-chat/api?tab=nodejs) （Replicate，第三方网站）

##### 🙋‍♀️ 使用 Yi Playground

如果你想与 Yi 聊天，并使用更多自定义选项（例如，系统提示、温度、重复惩罚等），你可以选择以下方式之一。
  
  - [Yi-34B-Chat-Playground](https://platform.lingyiwanwu.com/prompt/playground) （Yi 官方）
    - 如需使用 Yi Playground, 欢迎申请加入白名单（填写[英文](https://cn.mikecrm.com/l91ODJf)或者[中文](https://cn.mikecrm.com/gnEZjiQ)申请表）。

  - [Yi-34B-Chat-Playground](https://replicate.com/01-ai/yi-34b-chat) (Replicate，第三方网站) 

##### 🙋‍♀️ 使用 Yi Chat

以下提供了类似的用户体验，你可以选择以下方式之一，与 Yi 聊天。

- [Yi-34B-Chat](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)（Yi 官方 - Hugging Face）
  - 不需要注册。

- [Yi-34B-Chat](https://platform.lingyiwanwu.com/)（Yi 官方）
  - 如需使用官方在线聊天服务，欢迎申请加入白名单（填写[英文](https://cn.mikecrm.com/l91ODJf)或[中文](https://cn.mikecrm.com/gnEZjiQ)申请表）。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 快速上手 - PyPi (pip install)

本教程在配置为 **A800（80GB）** 的本地机器上运行 Yi-34B-Chat， 并进行推理。

#### 第 0 步：前提条件
 
- 确保安装了 Python 3.10 以上版本。

- 如果你想运行 Yi 系列模型，参阅「[部署要求](#部署)」。

#### 第 1 步：准备环境 

如需设置环境，安装所需要的软件包，运行下面的命令。

```bash
git clone https://github.com/01-ai/Yi.git
cd yi
pip install -r requirements.txt
```

#### 第 2 步：下载模型

你可以从以下来源下载 Yi 模型。

- [Hugging Face](https://huggingface.co/01-ai)
- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

#### 第 3 步：进行推理

你可以使用 Yi Chat 模型或 Base 模型进行推理。

##### 使用 Yi Chat 模型进行推理

1. 创建一个名为 `quick_start.py` 的文件，并将以下内容复制到该文件中。

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

2. 运行 `quick_start.py` 代码。

    ```bash
    python quick_start.py
    ```

    你将得到一个类似输出，如下所示。🥳

    ```bash
    Hello! How can I assist you today?
    ```

##### 使用 Yi Base 模型进行推理

- Yi-34B

步骤与「[使用 Yi Chat 模型进行推理](#使用-yi-chat-模型进行推理)」类似。

你可以使用现有文件 [`text_generation.py`](https://github.com/01-ai/Yi/tree/main/demo)进行推理。

```bash
python demo/text_generation.py  --model <your-model-path>
```

<details>

<summary> 你将得到一个类似输出，如下所示。🥳 ⬇️ </summary>

<br>

**指令**： Let me tell you an interesting story about cat Tom and mouse Jerry,

**回复**： Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

</details>
<br>

- Yi-9B
  
  输入

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

  输出

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
    <a href="#top">返回顶部 ⬆️ </a>  ] 
  </p>


<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 快速上手 - Docker

<details>
<summary> 🚀 教程：在本地 Docker 上运行 Yi-34B-Chat。⬇️</summary>
<br>本教程在本地 Docker（配置为 A800 GPU 或 4*4090）上运行 <strong>Yi-34B-Chat</strong> 模型，并进行推理。
<h4>第 0 步：准备工作</h4>
<p>确保你已经安装了 <a href="https://docs.docker.com/engine/install/?open_in_browser=true">Docker</a> 和 <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a>。</p>
<h4>第 1 步：启动 Docker</h4>
<pre><code>docker run -it --gpus all \
-v &lt;your-model-path&gt;: /models
ghcr.io/01-ai/yi:latest
</code></pre>
<p>或者，你也可以从<code>registry.lingyiwanwu.com/ci/01-ai/yi:latest</code> 拉取已经构建好的 Yi Docker 镜像。</p>

<h4>第 2 步：进行推理</h4>
    <p>你可以使用 Yi 的 Chat 模型或 Base 模型进行推理。</p>
    
<h5>使用 Yi Chat 模型进行推理</h5>
    <p>进行推理的步骤与「<a href="#使用-yi-chat-模型进行推理"> 在 pip 上使用 Yi Chat 模型进行推理 </a>」类似。</p>
    <p><strong>注意：</strong> 唯一不同的是你需要设置 <code>model_path</code> 为 <code>= '&lt;your-model-mount-path&gt;'</code> 而不是 <code>= '&lt;your-model-path&gt;'</code>。</p>
<h5>使用 Yi Base 模型进行推理</h5>
    <p>进行推理的步骤与「<a href="#使用-yi-chat-模型进行推理"> 在 pip 上使用 Yi Chat 模型进行推理 </a>」类似。</p>
    <p><strong>注意：</strong> 唯一不同的是你需要设置 <code>model_path</code> 为 <code>= '&lt;your-model-mount-path&gt;'</code> 而不是 <code>= '&lt;your-model-path&gt;'</code>。</p>
</details>

### 快速上手 - conda-lock

<details>
<summary> 🚀 如需创建一个可以完全重现的 conda 环境锁定文件，你可以使用 <code><a href="https://github.com/conda/conda-lock">conda-lock</a></code> 工具。 ⬇️</summary>
<br>
你可以参考  <a href="https://github.com/01-ai/Yi/blob/ebba23451d780f35e74a780987ad377553134f68/conda-lock.yml">conda-lock.yml</a> 文件，该文件包含了所需依赖项的具体版本信息。此外，你还可以使用<code><a href="https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html">micromamba</a></code>工具来安装这些依赖项。
<br>
安装这些依赖项的步骤，如下所示。

1. 根据<a href="https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html">指南</a>安装 "micromamba"。 

2. 运行命令 <code>micromamba install -y -n yi -f conda-lock.yml</code> ，创建一个名为<code>yi</code> conda 环境，并安装所需的依赖项。
</details>

### 快速上手 - llama.cpp
<details>
<summary> 🚀 教程：在本地 llama.cpp 上运行 Yi-chat-6B-2bits。⬇️ </summary> 
<br>本教程在本地 llama.cpp 上运行 <a href="https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main">Yi-chat-6B-2bits</a> 量化模型，并进行推理。</p>

- [步骤 0：前提条件](#步骤-0前提条件)
- [步骤 1：下载 llama.cpp](#步骤-1下载-llamacpp)
- [步骤 2：下载模型](#步骤-2下载模型)
- [步骤 3：进行推理](#步骤-3进行推理)

#### 步骤 0：前提条件

- 该教程适用于 MacBook Pro（16GB 内存和 Apple M2 Pro 芯片）。

- 确保你的电脑上安装了 [`git-lfs`](https://git-lfs.com/) 。
  
#### 步骤 1：下载 `llama.cpp`

如需克隆 [`llama.cpp`](https://github.com/ggerganov/llama.cpp) 仓库，运行以下命令。

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

#### 步骤 2：下载模型

步骤 2.1：仅下载 [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) 仓库的 pointers，运行以下命令。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

步骤 2.2：下载量化后的 Yi 模型 [yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf)，运行以下命令。

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

#### 步骤 3：进行推理

如需体验 Yi 模型（运行模型推理），你可以选择以下方式之一。

- [方式 1：在终端中进行推理](#方式-1在终端中进行推理)
  
- [方式 2：在 Web上进行推理](#方式-2在 Web上进行推理)

##### 方式 1：在终端中进行推理

本文使用 4 个线程编译 `llama.cpp` ，之后进行推理。在 `llama.cpp` 所在的目录，运行以下命令。

> ###### 提示
>
> - 将 `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` 替换为你的模型的实际路径。
>
> - 默认情况下，模型是续写模式（completion mode）。
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

##### 方式 2：在 Web上进行推理

1. 如需启用一个轻便敏捷的聊天机器人，你可以运行以下命令。

    ```bash
    ./server --ctx-size 2048 --host 0.0.0.0 --n-gpu-layers 64 --model /Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf
    ```

    你将得到一个类似输出。

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

2. 如需访问聊天机器人界面，可以打开网络浏览器，在地址栏中输入 `http://0.0.0.0:8080`。

    ![Yi模型聊天机器人界面 - LLaMA.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp1.png?raw=true)

3. 如果你在提示窗口中输入问题，例如，“如何喂养你的宠物狐狸？请用 6 个简单的步骤回答”，你将收到类似的回复。

    ![向 Yi 模型提问 - LLaMA.cpp](https://github.com/01-ai/Yi/blob/main/assets/img/yi_llama_cpp2.png?raw=true)

</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 快速上手 - 使用 Web demo

你可以使用 **Yi Chat 模型**（Yi-34B-Chat）创建 Web demo。
**注意**：Yi Base 模型（Yi-34B）不支持该功能。

[第一步：准备环境](#第-1-步准备环境)

[第二步：下载模型](#第-2-步下载模型)

第三步：启动 Web demo 服务，运行以下命令。

```bash
python demo/web_demo.py -c <你的模型路径>
```

命令运行完毕后，你可以在浏览器中输入控制台提供的网址，来使用 Web demo 功能。

 ![快速上手 -  Web demo](https://github.com/01-ai/Yi/blob/main/assets/img/yi_34b_chat_web_demo.gif?raw=true)

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 微调

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

完成后，你可以使用以下命令，比较微调后的模型与 Base 模型。

```bash
bash finetune/scripts/run_eval.sh
```
<details style="display: inline;"><summary> 你可以使用 Yi 6B 和 34B Base 模型的微调代码，根据你的自定义数据进行微调。 ⬇️</summary> <ul>

#### 准备工作

###### 从镜像开始

默认情况下，我们使用来自[BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) 的小型数据集来微调 Base 模型。
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

确保你已经安装了 conda。如需安装 conda， 你可以运行以下命令。

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

如果你想使用 Yi-34B 模型，**注意**此模式采用零卸载技术，占用了大量 CPU 内存，因此需要限制 34B 微调训练中的 GPU 数量。你可以使用 CUDA_VISIBLE_DEVICES 限制 GPU 数量（如 scripts/run_sft_Yi_34b.sh 中所示）。

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

将数据集从 Hugging Face 下载到本地存储 DATA_PATH，例如， Dahoas/rm-static。

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

对于 Yi-6B-Base 模型，设置 training_debug_steps=20 和 num_train_epochs=4， 就可以输出一个 Chat 模型，大约需要 20 分钟。

对于 Yi-34B-Base 模型，初始化时间相对较长，请耐心等待。

#### 评估

```bash
cd finetune/scripts

bash run_eval.sh
```

你将得到 Base 模型和微调模型的回复。
</ul>
</details>

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 量化

#### GPT-Q 量化
```bash
python quantization/gptq/quant_autogptq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

如需评估生成的模型，你可以使用以下代码。

```bash
python quantization/gptq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

<details style="display: inline;"><summary> 详细的量化过程。 ⬇️</summary> <ul>
<br>

[GPT-Q](https://github.com/IST-DASLab/gptq) 是一种后训练量化方法，能够帮助大型语言模型在使用时节省内存，保持模型的准确性，并加快模型的运行速度。

如需对 Yi 模型进行 GPT-Q 量化，使用以下教程。

运行 GPT-Q 需要使用 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 和 [exllama](https://github.com/turboderp/exllama)。
此外，Hugging Face Transformers 已经集成了 optimum 和 auto-gptq，能够实现语言模型的 GPT-Q 量化。

##### 量化模型

如需量化模型，你可以使用以下 `quant_autogptq.py` 脚本。

```bash
python quant_autogptq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### 运行量化模型

如需运行量化模型，你可以使用以下 `eval_quantized_model.py` 脚本。

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

如需评估生成的模型，你可以使用以下代码。

```bash
python quantization/awq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```
<details style="display: inline;"><summary> 详细的量化过程。⬇️</summary> <ul>
<br>

[AWQ](https://github.com/mit-han-lab/llm-awq)是一种后训练量化方法，可以将模型的权重数据高效准确地转化成低位数据（例如，INT3 或 INT4），因此可以减小模型占用的内存空间，保持模型的准确性。

如需对 Yi 模型进行 AWQ 量化，你可以使用以下教程。

运行 AWQ 需要使用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。

##### 量化模型

如需量化模型，你可以使用以下 `quant_autoawq.py` 脚本。

```bash
python quant_autoawq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

##### 运行量化模型

如需运行量化模型，你可以使用以下 `eval_quantized_model.py` 脚本。

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

在使用 Yi 量化模型之前，确保安装以下软件。

| 模型 | 软件 |
|:---|:---|
Yi 4-bits 量化模型 | [AWQ 和 CUDA](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install-from-pypi)
Yi 8-bits 量化模型 |  [GPTQ 和 CUDA](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#quick-installation)

#### 硬件要求

部署 Yi 系列模型之前，确保硬件满足以下要求。

##### Chat 模型

| 模型                 | 最低显存      | 推荐 GPU 示例                             |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br>  1 x A10 (24 GB)  <br> 1 x A30 (24 GB)              |
| Yi-6B-Chat-4bits     | 4 GB          | 1 x RTX 3060 (12 GB)<br> 1 x RTX 4060 (8 GB)                   |
| Yi-6B-Chat-8bits     | 8 GB          | 1 x RTX 3070 (8 GB) <br> 1 x RTX 4060 (8 GB)                   |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 (24 GB)<br> 1 x A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)  <br> 1 x A100 (40 GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090 (24 GB) <br> 2 x RTX 4090 (24 GB)<br> 1 x A800  (40 GB) |

以下是不同 batch 使用情况下的最低显存要求。

|  模型                  | batch=1 | batch=4 | batch=16 | batch=32 |
| :----------------------- | :------- | :------- | :-------- | :-------- |
| Yi-6B-Chat              | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-4bits  | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-6B-Chat-8bits  | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-34B-Chat       | 65 GB   | 68 GB   | 76 GB    | > 80 GB   |
| Yi-34B-Chat-4bits | 19 GB   | 20 GB   | 30 GB    | 40 GB    |
| Yi-34B-Chat-8bits | 35 GB   | 37 GB   | 46 GB    | 58 GB    |

##### Base 模型

|模型                   |最低显存      |        推荐GPU示例                     |
|:----------------------|:--------------|:-------------------------------------:|
| Yi-6B                | 15 GB         | 1 x RTX 3090 (24 GB) <br> 1 x RTX 4090 (24 GB) <br> 1 x A10 (24 GB)  <br> 1 x A30 (24 GB)                |
| Yi-6B-200K           | 50 GB         | 1 x A800 (80 GB)                            |
| Yi-9B                | 20 GB         | 1 x RTX 4090 (24 GB)                           |
| Yi-34B               | 72 GB         | 4 x RTX 4090 (24 GB) <br> 1 x A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |

### 学习中心

<details>
<summary> 如果你想学习如何使用 Yi 系列模型，这里有丰富的学习资源。 ⬇️</summary>
<br>

欢迎来到 Yi 学习中心！

无论你是经验丰富的专家还是初出茅庐的新手，你都可以在这里找到丰富的学习资源，增长有关 Yi 模型的知识，提升相关技能。这里的博客文章具有深刻的见解，视频教程内容全面，实践指南可实操性强，这些学习资源都可以助你一臂之力。

感谢各位 Yi 专家和用户分享了许多深度的技术内容，我们对各位小伙伴的宝贵贡献表示衷心的感谢！

在此，我们也热烈邀请你加入我们，为 Yi 做出贡献。如果你创作了关于 Yi 系列模型的内容，欢迎提交 PR 分享！🙌 

有了这些学习资源，你可以立即开启 Yi 学习之旅。祝学习愉快！🥳

#### 教程
##### 英文教程
| 类型        | 教程                                                      |      日期      |     作者     |
|:-------------|:--------------------------------------------------------|:----------------|:----------------|
| 博客        | [Running Yi-34B-Chat locally using LlamaEdge](https://www.secondstate.io/articles/yi-34b/)                   |  2023-11-30  |  [Second State](https://github.com/second-state)  |
| 视频       | [Install Yi 34B Locally - Chinese English Bilingual LLM](https://www.youtube.com/watch?v=CVQvj4Wrh4w&t=476s) | 2023-11-05  | [Fahd Mirza](https://www.youtube.com/@fahdmirza) |
| 视频       | [Dolphin Yi 34b - Brand New Foundational Model TESTED](https://www.youtube.com/watch?v=On3Zuv27V3k&t=85s) | 2023-11-27  |  [Matthew Berman](https://www.youtube.com/@matthew_berman)  |

##### 中文教程
| 类型        | 教程                                                      |      日期      |     作者     |
|:-------------|:--------------------------------------------------------|:----------------|:----------------|
| 博客        | [实测零一万物Yi-VL多模态语言模型：能准确“识图吃瓜”](https://mp.weixin.qq.com/s/fu4O9XvJ03JhimsEyI-SsQ)              |  2024-02-02  |  [苏洋](https://github.com/soulteary)  |
| 博客        | [本地运行零一万物 34B 大模型，使用 LLaMA.cpp & 21G 显存](https://zhuanlan.zhihu.com/p/668921042)                  |  2023-11-26  |  [苏洋](https://github.com/soulteary)  |
| 博客       | [零一万物模型折腾笔记：官方 Yi-34B 模型基础使用](https://zhuanlan.zhihu.com/p/671387298)                           | 2023-12-10 |  [苏洋](https://github.com/soulteary)  |
| 博客        | [CPU 混合推理，非常见大模型量化方案：“二三五六” 位量化方案](https://zhuanlan.zhihu.com/p/671698216)                  | 2023-12-12 |  [苏洋](https://github.com/soulteary)  |
| 博客        | [单卡 3 小时训练 Yi-6B 大模型 Agent：基于 LLaMA Factory 实战](https://zhuanlan.zhihu.com/p/678989191)             | 2024-01-22 | [郑耀威](https://github.com/hiyouga) |
| 博客        | [零一万物开源Yi-VL多模态大模型，魔搭社区推理&微调最佳实践来啦！](https://zhuanlan.zhihu.com/p/680098411)                  | 2024-01-26 |  [ModelScope](https://github.com/modelscope)  |
| 视频       | [只需 24G 显存，用 vllm 跑起来 Yi-34B 中英双语大模型](https://www.bilibili.com/video/BV17t4y1f7Ee/)               | 2023-12-28 |  [漆妮妮](https://space.bilibili.com/1262370256)  |
| 视频       | [Yi-VL-34B 多模态大模型 - 用两张 A40 显卡跑起来](https://www.bilibili.com/video/BV1Q5411y7AG/)               | 2023-01-28 |  [漆妮妮](https://space.bilibili.com/1262370256)  |
</details>

# 📌 为什么选择 Yi？

  - [生态](#生态)
    - [上游](#上游)
    - [下游](#下游)
      - [服务](#服务)
      - [量化](#️量化)
      - [微调](#️微调)
      - [API](#api)
  - [基准测试](#-基准测试)
    - [Chat 模型性能](#chat-模型性能)
    - [Base 模型性能](#base-模型性能)
      - [Yi-34B 和 Yi-34B-200K](#yi-34b-和-yi-34b-200k)
      - [Yi-9B](#yi-9b)
## 生态

Yi 生态为你提供一系列工具、服务和模型，你将获得丰富的体验，最大程度提升工作工作效率。

- [上游](#上游)
- [下游](#下游)
  - [服务](#下游---服务)
  - [量化](#下游---量化)
  - [微调](#下游---微调)
  - [API](#下游---api)

### 上游

Yi 系列模型遵循与 Llama 相同的模型架构。选择 Yi，你可以利用 Llama 生态中现有的工具、库和资源，无需创建新工具，提高开发效率。

例如，Yi 系列模型以 Llama 模型的格式保存。你可以直接使用 `LlamaForCausalLM` 和 `LlamaTokenizer` 加载模型，使用以下代码。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34b", device_map="auto")
```
<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 下游

> 💡 提示
> 
> - 如果你开发了与 Yi 相关的服务、模型、工具、平台或其它内容，欢迎提交 PR，将你的成果展示在 [Yi 生态](#下游---服务)。
>
> - 为了帮助他人快速理解你的工作，建议使用`<模型名称>: <模型简介> + <模型亮点>`的格式。

#### 下游 - 服务

如果你想在几分钟内开始使用 Yi，你可以使用以下基于 Yi 构建的服务。

- Yi-34B-Chat：你可以通过以下平台与 Yi 聊天。
  - [Yi-34B-Chat | Hugging Face](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
  - [Yi-34B-Chat | Yi Platform](https://platform.lingyiwanwu.com/)
  **注意**：如需使用 Yi Platform, 你可以申请加入白名单（填写[英文](https://cn.mikecrm.com/l91ODJf)或[中文](https://cn.mikecrm.com/gnEZjiQ)申请表）。

- [Yi-6B-Chat (Replicate)](https://replicate.com/01-ai)：使用该工具，你可以设置自定义参数，调用 APIs 来使用 Yi-6B-Chat。

- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models)：你可以使用该工具在本地运行 Yi 模型，根据自身偏好进行个性化设置。

#### 下游 - 量化

如果资源有限，你可以使用 Yi 的量化模型，如下所示。

这些量化模型虽然精度降低，但效率更高，例如，推理速度更快，RAM 使用量更小。

- [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ)
- [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
- [TheBloke/Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)

#### 下游 - 微调

如果你希望探索 Yi 的其它微调模型，你可以尝试以下方式。

- [TheBloke 模型](https://huggingface.co/TheBloke)：该网站提供了大量微调模型，这些微调模型基于多种大语言模型，包括 Yi 模型。
  
  以下是 Yi 的微调模型，根据下载量排序，包括但不限于以下模型。
  - [TheBloke/dolphin-2_2-yi-34b-AWQ](https://huggingface.co/TheBloke/dolphin-2_2-yi-34b-AWQ)
  - [TheBloke/Yi-34B-Chat-AWQ](https://huggingface.co/TheBloke/Yi-34B-Chat-AWQ)
  - [TheBloke/Yi-34B-Chat-GPTQ](https://huggingface.co/TheBloke/Yi-34B-Chat-GPTQ)
  
- [SUSTech/SUS-Chat-34B](https://huggingface.co/SUSTech/SUS-Chat-34B)：该模型在所有 70B 以下的模型中排名第一，超越了体量是其两倍的 deepseek-llm-67b-chat。你可以在 [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 上查看结果。
  
- [OrionStarAI/OrionStar-Yi-34B-Chat-Llama](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat-Llama)：该模型在 C-Eval 和 CMMLU 评估中超越了其它模型（例如，GPT-4、Qwen-14B-Chat 和 Baichuan2-13B-Chat），在 [OpenCompass LLM Leaderboard](https://opencompass.org.cn/leaderboard-llm) 上表现出色。
  
- [NousResearch/Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B)：该模型在 Capybara 数据集上使用 200K 上下文长度和 3 个 epochs 进行训练。

#### 下游 - API

- [amazing-openai-api](https://github.com/soulteary/amazing-openai-api)：此工具可以将 Yi 模型 API 转换成 OpenAI API 格式。
- [LlamaEdge](https://www.secondstate.io/articles/yi-34b/#create-an-openai-compatible-api-service-for-the-yi-34b-chat-model)：你可以通过该工具快速部署 Yi-34B-Chat 并开始聊天。该工具由 Rust 语言开发，使用可移植的 Wasm（WebAssembly）文件构建了一个与 OpenAI 兼容的 API 服务器。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

## 基准测试 

- [Chat 模型性能](#chat-模型性能)
- [Base 模型性能](#base-模型性能)

### Chat 模型性能

Yi-34B-Chat 模型表现出色，在 MMLU、CMMLU、BBH、GSM8k 等所有开源模型的基准测试中排名第一。
<br>

![Chat model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_chat.png?raw=true) 

<details>
<summary> 测评方法与挑战 ⬇️ </summary>

- **评估方式**： 该测评使用 zero-shot 和 few-shot 方法评估了除 TruthfulQA 以外的各种基准。
- **zero-shot 方法**： 大部分 Chat 模型常用 zero-shot 方法。
- **评估策略**： 本次测评的评估策略是要求模型在给出明确指令或包含隐含信息的指令情况下遵循指令（例如，使用少量样本示例），生成回应，并从生成的文本中提取相关答案。
- **面临的挑战**： 一些模型不适用少数数据集中的指令，无法按照所要求的特定格式产生输出，这会导致结果不理想。

<strong>*</strong>： C-Eval 的结果来源于验证数据集。
</details>

### Base 模型性能


#### Yi-34B 和 Yi-34B-200K 

Yi-34B 和 Yi-34B-200K 模型在开源模型中脱颖而出，尤其在 MMLU、CMMLU、常识推理、阅读理解等方面表现卓越。
<br>

![Base model performance](https://github.com/01-ai/Yi/blob/main/assets/img/benchmark_base.png?raw=true)

<details>
<summary> 测评方法 ⬇️</summary>

- **结果差异**： 在测试开源模型时，该测试的流程与其它测试方法（例如，OpenCompass）报告的结果之间存在差异。
- **结果发现**： 测评结果显示，各种模型在 Prompt、后处理策略和采样技术上的不同之处可能导致各种模型的结果产生显著差异。
- **相同的测试过程**： 该测试的方法论与原始基准一致，即在评估时使用相同的提示语和后处理策略，并在评估时应用贪心解码（greedy decoding），不对生成内容进行任何后处理。
- **测评其它模型**： 对于未提供测评结果的模型（包括以不同设置报告的分数），该测评尝试使用自身的流程获取结果。
- **评估维度全面**： 为了全面评估模型的能力，该测评采用了在 Llama2 中概述的方法。具体而言，针对尝试推理方面，该测评使用了 PIQA、SIQA、HellaSwag、WinoGrande、ARC、OBQA 和 CSQA 等方法。针对阅读理解方面，该测评使用了 SquAD、QuAC 和 BoolQ 等方法。
- **特殊设置**： CSQA 专门使用 7-shot 设置进行测试，而其它所有测试都使用 0-shot 设置进行。此外，该测评在“数学和编码”类别下引入了 GSM8K（8-shot@1）、MATH（4-shot@1）、HumanEval（0-shot@1）和 MBPP（3-shot@1）。
- **Falcon-180B 注意事项**： 由于技术限制，Falcon-180B 没有在 QuAC 和 OBQA 上进行测试。评测结果是其它任务的平均分数，通常而言， QuAC 和 OBQA 的分数较低。本次评估结果可能相对合理地反映了 Falcon-180B 的表现，没有低估它的性能。
</details>


#### Yi-9B

Yi-9B 模型在 Mistral-7B、SOLAR-10.7B、Gemma-7B、DeepSeek-Coder-7B-Base-v1.5 等相近尺寸的模型中名列前茅，具有出色的代码能力、数学能力、常识推理能力以及阅读理解能力。

![Yi-9B benchmark - details](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_details.png?raw=true)

- 在**综合**能力方面（Mean-All），Yi-9B 的性能**在尺寸相近的开源模型中最好，超越了** DeepSeek-Coder、DeepSeek-Math、Mistral-7B、SOLAR-10.7B 和 Gemma-7B。

![Yi-9B benchmark - overall](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_overall.png?raw=true)

- 在**代码**能力方面（Mean-Code），Yi-9B 的性能仅次于 DeepSeek-Coder-7B，**超越了** Yi-34B、SOLAR-10.7B、Mistral-7B 和 Gemma-7B。

![Yi-9B benchmark - code](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_code.png?raw=true)

- 在**数学**能力方面（Mean-Math），Yi-9B 的性能仅次于 DeepSeek-Math-7B，**超越了** SOLAR-10.7B、Mistral-7B 和 Gemma-7B。

![Yi-9B benchmark - math](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_math.png?raw=true)

- 在**常识和推理**能力方面（Mean-Text），Yi-9B 的性能与 Mistral-7B、SOLAR-10.7B 和 Gemma-7B **不相上下**。

![Yi-9B benchmark - text](https://github.com/01-ai/Yi/blob/main/assets/img/Yi-9B_benchmark_text.png?raw=true)

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

## 技术报告

更多关于 Yi 系列模型性能的详细信息，参阅 「[Yi：Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652)」。

### 引用

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

# 📌 谁可以使用 Yi？

答案是所有人! 🙌 ✅ 

关于如何使用 Yi 系列模型，参阅「[许可证](#许可证)」。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

# 📌 其它

### 致谢

我们对每位火炬手都深表感激，感谢你们为 Yi 社区所做的贡献。因为有你们，Yi 不仅是一个项目，还成为了一个充满活力的创新社区。我们由衷地感谢各位小伙伴！

[![yi contributors](https://contrib.rocks/image?repo=01-ai/yi&max=2000&columns=15)](https://github.com/01-ai/yi/graphs/contributors)

#### 本文贡献者
Yi Readme 中文版由以下[贡献者](https://github.com/01-ai/Yi/wiki/%F0%9F%93%9A-Yi-Translation-Plan#contributor-list)完成，排名不分先后，以用户名首字母顺序排列。
- Prompt 专家：[@kevinhall1998](https://github.com/kevinhall1998)
- 译员：[@202030481266](https://github.com/202030481266)、[@GloriaLee01](https://github.com/GloriaLee01)、[@markli404](https://github.com/markli404)、[@petter529](https://github.com/petter529) 与 [@soulteary](https://github.com/soulteary)
- 审校：[@Anonymitaet](https://github.com/Anonymitaet)、[@bltcn](https://github.com/bltcn)、[@Cookize](https://github.com/Cookize)、[@lljzhgxd](https://github.com/lljzhgxd) 与 [@markli404](https://github.com/markli404)



<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 免责声明

在训练过程中，我们使用数据合规性检查算法，最大程度地确保训练模型的合规性。由于数据复杂且语言模型使用场景多样，我们无法保证模型在所有场景下均能生成正确合理的回复。注意，模型仍可能生成有误的回复。对于任何因误用、误导、非法使用、错误使用导致的风险和问题，以及与之相关的数据安全问题，我们均不承担责任。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>

### 许可证

本仓库中的源代码遵循 [Apache 2.0 许可证](https://github.com/01-ai/Yi/blob/main/LICENSE)。Yi 系列模型完全开放，你可以免费用于个人用途、学术研究和商业用途。如需商用，你仅需[提交申请](https://www.lingyiwanwu.com/yi-license)，即能立刻自动获取 Yi 系列模型商用许可，而无需等待官方审批。所有使用必须遵守[《Yi系列模型社区许可协议 2.1》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)。

<p align="right"> [
  <a href="#top">返回顶部 ⬆️ </a>  ] 
</p>



