# 强化Yi-1.5-6B-Chat的数学和逻辑能力

## 目录
1. [简介](#简介)
2. [环境准备](#环境准备)
3. [安装SWIFT](#安装swift)
4. [数据集准备](#数据集准备)
5. [模型微调](#模型微调)
6. [LoRA合并](#lora合并)
7. [损失曲线可视化](#损失曲线可视化)
8. [模型推理](#模型推理)
9. [结语](#结语)

## 简介

本教程将指导您如何使用魔搭的SWIFT框架来微调Yi-1.5-6B-Chat模型。Yi-1.5-6B-Chat是由零一万物开发的一个强大的开源大语言模型，我们将使用SWIFT框架对其数学能力进行进一步的微调，以适应特定的任务需求。

## 环境准备

在开始之前，请确保您的系统满足以下要求：
- GPU：本教程使用NVIDIA L4 GPU进行实验

请注意，大语言模型的训练和推理都需要较高的计算资源。如果您的本地环境不满足要求，可以考虑使用云计算平台提供的GPU实例。

## 安装SWIFT

首先，我们需要从GitHub克隆SWIFT仓库并安装必要的依赖。请按照以下步骤操作：

1. 打开终端，运行以下命令克隆SWIFT仓库：

   ```bash
   git clone https://github.com/modelscope/swift.git
   ```

2. 进入SWIFT目录：

   ```bash
   cd swift
   ```

3. 安装SWIFT及其依赖（包括LLM相关的包）：

   ```bash
   pip install -e '.[llm]'
   ```

   这个命令会以可编辑模式安装SWIFT，并安装LLM相关的额外依赖。

安装过程可能需要几分钟时间，请耐心等待。如果遇到任何错误，请检查您的Python和CUDA版本是否兼容，并确保您有足够的磁盘空间。

## 数据集准备

在本教程中，我们将使用三个数据集来微调模型：

1. ruozhiba：一个中文问答数据集，配比5000条数据以保留模型的常识逻辑能力
2. AI-ModelScope/blossom-math-v2：一个数学问题数据集(原作者是azure99,原链接点击[这里](https://huggingface.co/datasets/Azure99/blossom-math-v2))
3. math.jsonl：自定义数据集，主要是提升高等数学、线性代数、概率论题目的解题能力

SWIFT框架支持直接使用在线数据集，也支持本地数据集。对于ruozhiba和blossom-math-v2，我们可以直接使用在线版本。对于HJ_math.jsonl，您需要确保它位于正确的目录中（本例中为/content/HJ_math.jsonl）。

数据集的格式应该符合SWIFT的要求，通常是JSON Lines格式，每一行包含一个训练样本。例如SWIFT官方示例：

```json
{"system": "你是一个数学好能手", "query": "(1+3)*10+99的结果是多少", "response": "首先计算 1+3=4, 4*10=40, 40+99=134, 所以结果为134"}
{"query": "求以下方程的解：x^2 - 4x + 4 = 0", "response": "方程x^2 - 4x + 4 = 0的解为x = 2（重根）。"}
{"query": "求解以下方程组：x + y = 5 和 2x - y = 1", "response": "解这个方程组，得到x = 2，y = 3。"}
```

确保您的自定义数据集HJ_math.jsonl遵循类似的格式。当然数据集也已经开源在了[HuggingFace](https://huggingface.co/datasets/haijian06/Advanced-Math)并且免费开放下载

## 模型微调

现在，我们开始微调模型。我们将使用LoRA（Low-Rank Adaptation）技术来微调Yi-1.5-6B-Chat模型。LoRA是一种高效的微调方法，可以显著减少所需的计算资源。

以下是微调的命令：

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path 01ai/Yi-1.5-6B-Chat \
    --dataset ruozhiba#5000 AI-ModelScope/blossom-math-v2#5000 /content/HJ_math.jsonl \
    --output_dir output \
    --sft_type lora \
    --num_train_epochs 1 \
    --max_length 1024 \
    --dtype AUTO
```

这个命令的解释如下(但是更多的也库参考SWIFT官方文档～)：

1. `CUDA_VISIBLE_DEVICES=0`：指定使用的GPU编号（这里使用第一个GPU）。

2. `swift sft`：调用SWIFT的微调（SFT，Supervised Fine-Tuning）功能。

3. `--model_id_or_path 01ai/Yi-1.5-6B-Chat`：指定基础模型的ID或路径。

4. `--dataset ruozhiba#5000 AI-ModelScope/blossom-math-v2#5000 /content/HJ_math.jsonl`：
   - 指定使用的数据集，这里使用了三个数据集
   - `#5000`表示从每个数据集中随机选择5000个样本
   - 最后一个是本地数据集的路径

5. `--output_dir output`：指定输出目录，微调后的模型和日志将保存在这里。

6. `--sft_type lora`：指定使用LoRA进行微调。

7. `--num_train_epochs 1`：设置训练的轮数（epoch）为1。

8. `--max_length 1024`：设置输入序列的最大长度为1024个token。

9. `--dtype AUTO`：自动选择适合的数据类型（通常是float16或bfloat16，取决于GPU支持）。

运行这个命令后，SWIFT将开始微调过程。微调可能需要几个小时甚至更长时间，取决于您的硬件性能和数据集大小。
## 合并前测试
微调完成后，我们首先对LoRA权重模型进行测试如下：
```计算: (15 * 4 - 12) / 3 + 8^2```
![img.png](assets/5/img(5-1).png)
```求极限 lim(x→+∞) ln(1+x)ln(1+1/x) 的值。```
![img_1.png](assets/5/img(5-3).png)
我们可以看到目前模型的数学解题能力很强！
## LoRA合并

微调完成后，我们需要将LoRA权重与原始模型合并。这一步是可选的，但合并后的模型更容易部署和使用。使用以下命令进行合并：

```bash
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir '/content/swift/output/yi-1_5-6b-chat/v0-20240713-090427/checkpoint-365' \
    --merge_lora true
```

这个命令的参数解释：

- `--ckpt_dir`：指定保存LoRA权重的检查点目录。请根据实际输出的路径进行修改。
- `--merge_lora true`：指示SWIFT将LoRA权重与基础模型合并。

合并过程完成后，您将在指定的目录中找到合并后的模型文件。

## 损失曲线可视化

微调过程中，SWIFT会自动记录训练损失。您可以在文件包中进行查看

您应该能看到类似下面的损失曲线图：

![train_loss(5-1).png](assets/5/img(5-4).png)

损失曲线应该呈现下降趋势，表明模型正在学习并改善其性能。

## 模型推理

现在我们可以使用微调后的模型进行推理。使用以下命令：

```bash
swift infer \
    --ckpt_dir /content/yi-1_5-6b-chat/v0-20240717-024536/checkpoint-682-merged \
    --eval_human true \
    --stop_words "Observation:" \
    --infer_backend pt
```

参数解释：
- `--ckpt_dir`：指定合并后的模型检查点目录。
- `--eval_human true`：启用人机交互模式。
- `--stop_words "Observation:"`：设置停止词，用于控制输出长度。
- `--infer_backend pt`：使用PyTorch作为推理后端。

运行此命令后，您将进入交互式推理模式。您可以输入问题，模型会生成回答。以下是一些示例问题：


![img(5-2).jpg](assets/5/img(5-2).jpg)

您可以继续输入更多问题来测试模型的性能。记得观察模型在不同类型问题上的表现，特别是在您用于微调的数据集相关的问题上。

## 结语

恭喜！您已经成功使用SWIFT框架微调了Yi-1.5-6B-Chat模型，并学会了如何进行推理。这个过程涵盖了从环境设置、数据准备、模型微调到最终的模型使用。

记住，模型的性能很大程度上取决于您用于微调的数据质量和数量。您可以尝试使用不同的数据集或调整微调参数来进一步改善模型性能。

如果您在过程中遇到任何问题，请查阅SWIFT的[官方文档](https://github.com/modelscope/swift/)。继续探索和实验，您将能够根据自己的需求定制出更强大的语言模型！
