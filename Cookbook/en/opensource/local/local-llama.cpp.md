### 🌟Local Running with llama.cpp

llama.cpp implements LLM inference on various hardware (local and cloud) with minimal setup and state-of-the-art performance. It is written in C++. We can also use the GGUF format models of the Yi series in llama.cpp.

In the following tutorial, we will use the Yi-1.5-6B-Chat-GGUF model downloaded from [huggingface](https://huggingface.co/models?search=yi-1.5-GGUF) for testing and teaching. Of course, you can choose other models from the Yi series, but please note⚠️ that the model file must be in GGUF format.

#### Preparing the GGUF Format Model

##### 1. Installing huggingface_hub (You can skip this if you have a locally fine-tuned or quantized model)

The purpose is to download the model from huggingface:

``````bash
pip install huggingface_hub
``````

You can directly download the GGUF format Yi model [lmstudio-community/Yi-1.5-6B-Chat-GGUF](https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF) from huggingface.

If you want to directly download the GGUF format model, you can execute the following command to start downloading the model (not recommended, as the GGUF format package provided by LM studio for the Yi model takes up a lot of disk space, consider this if you have a large hard drive):

``````bash
huggingface-cli download lmstudio-community/Yi-1.5-6B-Chat-GGUF --local-dir /root/yi-models/Yi-1.5-6B-Chat-GGUF
``````

##### 2. Converting to GGUF Format for Use

We will first download the Yi-1.5-6B-Chat model from huggingface and then convert it to GGUF format using llama.cpp.

``````bash
huggingface-cli download 01-ai/Yi-1.5-6B-Chat --local-dir /root/yi-models/Yi-1.5-6B-Chat
``````

Convert to GGUF format. You need to download and install llama.cpp first, which you can refer to the [Download and Installation](#Download and Installation) section below.

⚠️Note that "convert-hf-to-gguf.py" is located under the llama.cpp directory, so execute it directly from the llama.cpp base path:

``````
python convert-hf-to-gguf.py /root/yi-models/Yi-1.5-6B-Chat --outfile /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf --outtype q8_0
``````

Now we have a GGUF format model directory at (/root/yi-models/Yi-1.5-6B-Chat-GGUF).

#### Download and Installation

##### 2. Downloading llama.cpp from Source Code

``````bash
git clone https://github.com/ggerganov/llama.cpp
``````

``````bash
cd llama.cpp
``````

##### 3. Compilation

We provide two versions for compilation. Please check if your computer is a CPU or GPU version.

Check version:

``````python
import torch # If pytorch is installed successfully, it can be imported
print(torch.cuda.is_available()) # Check if CUDA is available, if True, it can be used, otherwise use the CPU version
``````

Generate Makefile using cmake (CUDA version):

``````bash
cmake -B build_cuda -DLLAMA_CUDA=ON
cmake --build build_cuda --config Release -j 8
``````

Generate Makefile using cmake (CPU version):

``````bash
cmake -B build_cpu
cmake --build build_cpu --config Release
``````

##### 4. Running

First, switch to the "bin" directory to execute:

``````bash
cd build_cuda or cd build_cpu
``````

``````bash
cd bin
``````

Execute using "main". There are many adjustable parameters for the execution command. You can refer to [here](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md) for modifications.

For Linux, macOS:

``````bash
./llama-cli -m /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf -n -1 --color -r "User:" --in-prefix " " -i -p \
'User: Hello
AI: Hello, I am from Zero One Thousand Things. How can I help you?
User: Good
AI: What topic would you like to talk about?
User:'
``````

For Windows:

``````bash
llama-cli.exe -m /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf -n -1 --color -r "User:" --in-prefix " " -i -e -p "User: Hello\nAI: Hello, I am from Zero One Thousand Things. How can I help you?\nUser: Good!\nAI: What topic would you like to talk about?\nUser:"
``````

You can start the conversation after running～～

##### 5. Quantization using llama.cpp

If you want to quantize using llama.cpp, you can execute:

``````
./llama-quantize --allow-requantize /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q8_0-v1.gguf /root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q4_1-v1.gguf Q4_1
``````

After execution, you will have the quantized "Yi-1.5-6B-Chat-q4_1-v1.gguf" in the directory "/root/yi-models/Yi-1.5-6B-Chat-GGUF/Yi-1.5-6B-Chat-q4_1-v1.gguf".

You can use it in the same way as before, [refer to](#4. Running).

This is just an example of quantizing to Q4_1. You can execute the following command to see other usages:

``````
./llama-quantize -h
``````

You can quantize to other precisions by following the usage of llama.cpp. Refer to the following for details:

``````
type for the output.weight tensor
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  --keep-split: will generate a quantized model in the same shards as input  --override-kv KEY=TYPE:VALUE
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
