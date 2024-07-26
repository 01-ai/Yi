### 🌟Local Running with MLX-LM

MLX-LM is a framework for local deployment of large models on Mac OS. For detailed information, please refer to the [official documentation](https://github.com/ml-explore/mlx-examples/tree/main?tab=readme-ov-file).

⚠️Please note that MLX-LM is only compatible with the Mac OS operating system.

#### Download and Installation

``````bash
pip install mlx-lm
``````

#### Getting Started

The following example uses "mlx-community/Yi-1.5-6B-Chat-8bit".

You can also replace it with other models, such as "mlx-community/Yi-1.5-34B-Chat-4bit".

``````python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Yi-1.5-6B-Chat-8bit")

response = generate(model, tokenizer, prompt="hello", verbose=True)
``````