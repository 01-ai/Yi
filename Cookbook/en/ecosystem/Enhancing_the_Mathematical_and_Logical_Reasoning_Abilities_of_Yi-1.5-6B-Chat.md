# Enhancing Yi-1.5-6B-Chat's Mathematical and Logical Abilities

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Preparation](#environment-preparation)
3. [Installing SWIFT](#installing-swift)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Fine-tuning](#model-fine-tuning)
6. [LoRA Merging](#lora-merging)
7. [Loss Curve Visualization](#loss-curve-visualization)
8. [Model Inference](#model-inference)
9. [Conclusion](#conclusion)

## Introduction

This tutorial will guide you on how to use the SWIFT framework from ModelScope to fine-tune the Yi-1.5-6B-Chat model. Yi-1.5-6B-Chat is a powerful open-source large language model developed by 01.AI. We will use the SWIFT framework to further fine-tune its mathematical abilities to adapt to specific task requirements.

## Environment Preparation

Before starting, please ensure your system meets the following requirements:
- GPU: This tutorial uses an NVIDIA L4 GPU for experiments

Please note that training and inference of large language models require significant computational resources. If your local environment doesn't meet the requirements, consider using GPU instances provided by cloud computing platforms.

## Installing SWIFT

First, we need to clone the SWIFT repository from GitHub and install the necessary dependencies. Please follow these steps:

1. Open a terminal and run the following command to clone the SWIFT repository:

   ```bash
   git clone https://github.com/modelscope/swift.git
   ```

2. Enter the SWIFT directory:

   ```bash
   cd swift
   ```

3. Install SWIFT and its dependencies (including LLM-related packages):

   ```bash
   pip install -e '.[llm]'
   ```

   This command will install SWIFT in editable mode and install additional LLM-related dependencies.

The installation process may take a few minutes. Please be patient. If you encounter any errors, check if your Python and CUDA versions are compatible and ensure you have enough disk space.

## Dataset Preparation

In this tutorial, we will use three datasets to fine-tune the model:

1. ruozhiba: A Chinese Q&A dataset, using 5000 samples to maintain the model's common sense and logical abilities
2. AI-ModelScope/blossom-math-v2: A mathematical problem dataset (original author is azure99, original link [here](https://huggingface.co/datasets/Azure99/blossom-math-v2))
3. math.jsonl: A custom dataset, mainly to enhance problem-solving abilities in advanced mathematics, linear algebra, and probability theory

The SWIFT framework supports both online datasets and local datasets. For ruozhiba and blossom-math-v2, we can use the online versions directly. For HJ_math.jsonl, you need to ensure it's in the correct directory (in this case, /content/HJ_math.jsonl).

The dataset format should comply with SWIFT's requirements, usually in JSON Lines format, with each line containing a training sample. For example, the SWIFT official example:

```json
{"system": "You are a math expert", "query": "What is the result of (1+3)*10+99?", "response": "First, calculate 1+3=4, then 4*10=40, finally 40+99=139, so the result is 139"}
{"query": "Solve the following equation: x^2 - 4x + 4 = 0", "response": "The solution to the equation x^2 - 4x + 4 = 0 is x = 2 (double root)."}
{"query": "Solve the following system of equations: x + y = 5 and 2x - y = 1", "response": "Solving this system of equations, we get x = 2, y = 3."}
```

Make sure your custom dataset HJ_math.jsonl follows a similar format. Of course, the dataset has also been open-sourced on [HuggingFace](https://huggingface.co/datasets/haijian06/Advanced-Math) and is available for free download.

## Model Fine-tuning

Now, let's start fine-tuning the model. We will use the LoRA (Low-Rank Adaptation) technique to fine-tune the Yi-1.5-6B-Chat model. LoRA is an efficient fine-tuning method that can significantly reduce the required computational resources.

Here's the command for fine-tuning:

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

Here's an explanation of this command (for more details, refer to the SWIFT official documentation):

1. `CUDA_VISIBLE_DEVICES=0`: Specifies the GPU number to use (here, using the first GPU).

2. `swift sft`: Calls SWIFT's fine-tuning (SFT, Supervised Fine-Tuning) function.

3. `--model_id_or_path 01ai/Yi-1.5-6B-Chat`: Specifies the ID or path of the base model.

4. `--dataset ruozhiba#5000 AI-ModelScope/blossom-math-v2#5000 /content/HJ_math.jsonl`:
   - Specifies the datasets to use, here using three datasets
   - `#5000` means randomly selecting 5000 samples from each dataset
   - The last one is the path to the local dataset

5. `--output_dir output`: Specifies the output directory where the fine-tuned model and logs will be saved.

6. `--sft_type lora`: Specifies using LoRA for fine-tuning.

7. `--num_train_epochs 1`: Sets the number of training epochs to 1.

8. `--max_length 1024`: Sets the maximum length of input sequences to 1024 tokens.

9. `--dtype AUTO`: Automatically selects the appropriate data type (usually float16 or bfloat16, depending on GPU support).

After running this command, SWIFT will begin the fine-tuning process. Fine-tuning may take several hours or even longer, depending on your hardware performance and dataset size.

## Pre-merge Testing

After fine-tuning is complete, we first test the LoRA weight model as follows:
```Calculate: (15 * 4 - 12) / 3 + 8^2```
![img.png](assets/5/img(5-1).png)
```Find the limit of lim(x→+∞) ln(1+x)ln(1+1/x)```
![img_1.png](assets/5/img(5-3).png)
We can see that the model's mathematical problem-solving ability is now quite strong!

## LoRA Merging

After fine-tuning, we need to merge the LoRA weights with the original model. This step is optional, but the merged model is easier to deploy and use. Use the following command to merge:```bash
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir '/content/swift/output/yi-1_5-6b-chat/v0-20240713-090427/checkpoint-365' \
    --merge_lora true```

Explanation of this command's parameters:

- `--ckpt_dir`: Specifies the checkpoint directory where LoRA weights are saved. Please modify according to the actual output path.
- `--merge_lora true`: Instructs SWIFT to merge LoRA weights with the base model.

After the merging process is complete, you will find the merged model files in the specified directory.

## Loss Curve Visualization

During the fine-tuning process, SWIFT automatically records the training loss. You can view it in the file package.

You should see a loss curve graph similar to the one below:
![train_loss(5-1).png](assets/5/img(5-4).png)

The loss curve should show a downward trend, indicating that the model is learning and improving its performance.

## Model Inference

Now we can use the fine-tuned model for inference. Use the following command:```bash
swift infer \
    --ckpt_dir /content/yi-1_5-6b-chat/v0-20240717-024536/checkpoint-682-merged \
    --eval_human true \
    --stop_words "Observation:" \
    --infer_backend pt```

Parameter explanation:
- `--ckpt_dir`: Specifies the directory of the merged model checkpoint.
- `--eval_human true`: Enables human-machine interaction mode.
- `--stop_words "Observation:"`: Sets stop words to control output length.
- `--infer_backend pt`: Uses PyTorch as the inference backend.

After running this command, you will enter an interactive inference mode. You can input questions, and the model will generate answers. Here are some example questions:
![img(5-2).jpg](assets/5/img(5-2).jpg)

You can continue inputting more questions to test the model's performance. Remember to observe the model's performance on different types of questions, especially those related to the datasets you used for fine-tuning.

## Conclusion

Congratulations! You have successfully fine-tuned the Yi-1.5-6B-Chat model using the SWIFT framework and learned how to perform inference. This process covered everything from environment setup, data preparation, model fine-tuning to final model usage.

Remember, the model's performance largely depends on the quality and quantity of data you used for fine-tuning. You can try using different datasets or adjusting fine-tuning parameters to further improve model performance.

If you encounter any issues during the process, please refer to SWIFT's [official documentation](https://github.com/modelscope/swift/). Continue exploring and experimenting, and you'll be able to customize more powerful language models according to your needs!```
