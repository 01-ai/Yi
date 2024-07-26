### 🌟Fine-tuning with SWIFT

SWIFT, developed by ModelScope, is a framework designed for training, inferencing, evaluating, and deploying multimodal large models. It enables a seamless workflow from model training and evaluation to application. Let's explore how to fine-tune the Yi model using SWIFT.

#### Installation

Start by cloning the SWIFT repository:

```bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

#### Fine-tuning Steps

We'll use the CLI for fine-tuning. Here's the command:

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path 01ai/Yi-1.5-6B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
```

- `--model_id_or_path`: Replace with the desired model.
- `--dataset`: Specify the dataset for fine-tuning.
- `--output_dir`:  Define the directory to save the fine-tuned model.

For more detailed information on SWIFT, please refer to the official GitHub repository: [https://github.com/modelscope/swift](https://github.com/modelscope/swift). 
