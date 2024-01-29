# 运行 Yi 模型，从 llama.cpp 开始

如果你很想快速体验 Yi 模型的魅力，但是受限于资源有限，比如只有一台笔记本电脑。可以尝试使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或者 [ollama.cpp](https://ollama.ai/) 这类实用的高性能推理工具。它们可以帮助我们在一小时内，甚至就是在一台笔记本电脑上，完成 Yi 模型的本地部署和运行。

本教程将一步步指导你来完成 “模型下载”、“模型量化”，“将 Yi 模型 [yi-chat-6B-2bits](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main)运行”起来，并进行问答演示。

- [第 0 步：环境准备](#step-0-prerequisites)
- [第 1 步：下载 llama.cpp 代码](#step-1-download-llamacpp)
- [第 2 步：模型下载](#step-2-download-yi-model)
- [第 3 步：开始体验](#step-3-perform-inference)

# 第 0 步：环境准备

- 本教程使用的是配备 16GB 内存和 Apple M2 Pro 芯片的 MacBook Pro 笔记本，如果你有更充裕的资源体验将会更好。
- 我们需要确保你的机器上已经安装好了 [`git-lfs`](https://git-lfs.com/)。
  
# 第 1 步：下载 `llama.cpp` 代码

我们需要使用下面的命令，完成 [`llama.cpp`](https://github.com/ggerganov/llama.cpp) 仓库代码的下载：

```bash
git clone git@github.com:ggerganov/llama.cpp.git
```

现在，你已经准备好开始下载 Yi 模型并体验即时问答的乐趣了！

## 第二步：模型下载

2.1 想要快速克隆 [XeIaso/yi-chat-6B-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main) 到本地，只需要运行以下命令：


```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/XeIaso/yi-chat-6B-GGUF
```

2.2 如果你想要下载一个量化的 Yi 模型 ([yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q2_K.gguf))，可以运行以下命令：

```bash
git-lfs pull --include yi-chat-6b.Q2_K.gguf
```

## 第三步：开始体验

想要体验 Yi 模型（进行模型推理），我们可以选择以下任意一种方法。

- [方法 1：在终端中执行推理](#method-1-perform-inference-in-terminal)
  
- [方法 2：在网页上执行推理](#method-2-perform-inference-in-web)

### 方法一：在终端中执行推理

要使用4个线程编译 `llama.cpp` 并随后进行推理，请导航到 `llama.cpp` 所在的目录，并运行以下命令。

> ### 提示
>
> - 将 `/Users/yu/yi-chat-6B-GGUF/yi-chat-6b.Q2_K.gguf` 替换为你的模型的实际路径。
>
> - 默认情况下，模型处于完成模式。
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