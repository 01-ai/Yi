### 🌟使用llama.cpp本地运行

llama.cpp是在各种硬件（本地和云端）上以最少的设置和最先进的性能实现 LLM 推理。llama.cpp是由C++编写而成。我们也可以在llama.cpp中使用Yi系列的GGUF格式的模型。

以下教学我们使用从[huggingface](https://huggingface.co/models?search=yi-1.5-GGUF)中下载Yi-1.5-6B-Chat-GGUF模型来进行测试教学，当然你也可以选择Yi系列的其它模型，但是必须注意⚠️的是模型文件必须是GGUF格式。

#### 准备GGUF格式的模型

##### 1.安装 huggingface_hub(如果你有本地已经微调或者量化好的模型你可以不用执行)

目的是从huggingface下载模型：

``````bash
pip install huggingface_hub
``````

可以从huggingface中直接下载GGUF格式的Yi模型[lmstudio-community/Yi-1.5-6B-Chat-GGUF](https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF)。

如果你想直接下载GGUF格式的模型，可以执行以下指令开始下载模型(不建议，因为LM studio提供的是Yi模型的GGUF格式包，会占据很多硬盘，硬盘大的可以考虑)：

``````bash
huggingface-cli download lmstudio-community/Yi-1.5-6B-Chat-GGUF --local-dir /root/yi-models/Yi-1.5-6B-Chat-GGUF
``````

##### 2.转换成GGUF格式进行使用

我们还是先从huggingface下载Yi-1.5-6B-Chat模型，然后再通过llama.cpp转换为GGUF格式。

``````bash
huggingface-cli download 01-ai/Yi-1.5-6B-Chat --local-dir /root/yi-models/Yi-1.5-6B-Chat
``````

转换为GGUF格式，但是你要体检下载安装llama.cpp你可以参考下一节的[下载安装](#下载安装)。

⚠️注意convert-hf-to-gguf.py在llama.cpp下，所以直接在llama.cpp base路径执行即可：

``````
python convert-hf-to-gguf.py /root/yi-models/Yi-1.5-6B-Chat --outfile /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf --outtype q8_0
``````

这样我们就得到了一个GGUF格式的模型目录在（/root/yi-models/Yi-1.5-6B-Chat-GGUF）。

#### 下载安装

##### 2.从源码下载llama.cpp

``````bash
git clone https://github.com/ggerganov/llama.cpp
``````

``````bash
cd llama.cpp
``````

##### 3.编译

我们提供两个版本来进行编译，请查看你的电脑是cpu还是gpu版本

查看版本：

``````python
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用，如果True表示可以使用，反之则使用cpu版本
``````

使用cmake 生成 Makefile(cuda版本)：

``````bash
cmake -B build_cuda -DLLAMA_CUDA=ON
cmake --build build_cuda --config Release -j 8
``````

使用cmake 生成 Makefile(cpu版本)：

``````bash
cmake -B build_cpu
cmake --build build_cpu --config Release
``````

##### 4.开始运行

首先切换目录到bin目录下进行执行：

``````bash
cd build_cuda or cd build_cpu
``````

``````bash
cd bin
``````

通过main进行执行，执行命令有许多可以调节的参数，你可以参考[这里](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)进行修改。

适用于Linux、macOS 。

``````bash
./llama-cli -m /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf -n -1 --color -r "User:" --in-prefix " " -i -p \
'User: 你好
AI: 你好我来自零一万物，有什么可以帮助您？
User: 好啊
AI: 你想聊聊什么话题呢？
User:'
``````

适用于Windows：

``````bash
llama-cli.exe -m /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf -n -1 --color -r "User:" --in-prefix " " -i -e -p "User: 你好\nAI: 你好我来自零一万物，有什么可以帮助您？\nUser: 好啊!\nAI: 你想聊聊什么话题呢？\nUser:"
``````

运行后即可进行对话了
![llama.cpp](../../assets/llama-cpp-0.jpg)



##### 5.使用llama.cpp量化

如果你想使用llama.cpp进行量化，你可以执行

``````
./llama-quantize --allow-requantize /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q4_1-v1.gguf Q4_1
``````

执行完毕后你就得到了量化后的Yi-1.5-6B-Chat-q4_1-v1.gguf在</root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q4_1-v1.gguf>目录下。

使用和前面一样，可以[参考](#4开始运行)

这只是一个实例量化为Q4_1，你可以执行如下指令查看其它用法：

``````
./llama-quantize -h
``````

遵守llama.cpp的用法，量化成其它精度均可具体参考如下：

``````
type for the output.weight tensor
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  --keep-split: will generate quatized model in the same shards as input  --override-kv KEY=TYPE:VALUE
      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.
Note: --include-weights and --exclude-weights cannot be used together

Allowed quantization types:
   2  or  Q4_0    :  3.56G, +0.2166 ppl @ LLaMA-v1-7B
   3  or  Q4_1    :  3.90G, +0.1585 ppl @ LLaMA-v1-7B
   8  or  Q5_0    :  4.33G, +0.0683 ppl @ LLaMA-v1-7B
   9  or  Q5_1    :  4.70G, +0.0349 ppl @ LLaMA-v1-7B
  19  or  IQ2_XXS :  2.06 bpw quantization
  20  or  IQ2_XS  :  2.31 bpw quantization
  28  or  IQ2_S   :  2.5  bpw quantization
  29  or  IQ2_M   :  2.7  bpw quantization
  24  or  IQ1_S   :  1.56 bpw quantization
  31  or  IQ1_M   :  1.75 bpw quantization
  10  or  Q2_K    :  2.63G, +0.6717 ppl @ LLaMA-v1-7B
  21  or  Q2_K_S  :  2.16G, +9.0634 ppl @ LLaMA-v1-7B
  23  or  IQ3_XXS :  3.06 bpw quantization
  26  or  IQ3_S   :  3.44 bpw quantization
  27  or  IQ3_M   :  3.66 bpw quantization mix
  12  or  Q3_K    : alias for Q3_K_M
  22  or  IQ3_XS  :  3.3 bpw quantization
  11  or  Q3_K_S  :  2.75G, +0.5551 ppl @ LLaMA-v1-7B
  12  or  Q3_K_M  :  3.07G, +0.2496 ppl @ LLaMA-v1-7B
  13  or  Q3_K_L  :  3.35G, +0.1764 ppl @ LLaMA-v1-7B
  25  or  IQ4_NL  :  4.50 bpw non-linear quantization
  30  or  IQ4_XS  :  4.25 bpw non-linear quantization
  15  or  Q4_K    : alias for Q4_K_M
  14  or  Q4_K_S  :  3.59G, +0.0992 ppl @ LLaMA-v1-7B
  15  or  Q4_K_M  :  3.80G, +0.0532 ppl @ LLaMA-v1-7B
  17  or  Q5_K    : alias for Q5_K_M
  16  or  Q5_K_S  :  4.33G, +0.0400 ppl @ LLaMA-v1-7B
  17  or  Q5_K_M  :  4.45G, +0.0122 ppl @ LLaMA-v1-7B
  18  or  Q6_K    :  5.15G, +0.0008 ppl @ LLaMA-v1-7B
   7  or  Q8_0    :  6.70G, +0.0004 ppl @ LLaMA-v1-7B
   1  or  F16     : 14.00G, -0.0020 ppl @ Mistral-7B
  32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
   0  or  F32     : 26.00G              @ 7B
          COPY    : only copy tensors, no quantizing
``````
