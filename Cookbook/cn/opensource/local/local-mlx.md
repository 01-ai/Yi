### 🌟使用MLX-LM本地运行

MLX-LM是一款适用Mac os进行本地部署大模型的框架，具体内容参考[官方文档](https://github.com/ml-explore/mlx-examples/tree/main?tab=readme-ov-file)。

⚠️请注意MLX-LM仅适用于Mac os操作系统。

#### 下载和安装

``````bash
pip install mlx-lm
``````

#### 开始使用

以下使用mlx-community/Yi-1.5-6B-Chat-8bit作为示例。

同样的也可以替换为其它模型，例如 mlx-community/Yi-1.5-34B-Chat-4bit。

``````python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Yi-1.5-6B-Chat-8bit")

response = generate(model, tokenizer, prompt="hello", verbose=True)
``````