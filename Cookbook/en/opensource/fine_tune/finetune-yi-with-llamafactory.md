### 🌟Fine-tuning with LLaMA-Factory

LLaMA Factory is an open-source, low-code framework for fine-tuning large language models, developed by Yaowei Zheng, a PhD student at Beihang University. It integrates widely-used fine-tuning techniques, making the process straightforward and accessible. Let's get started!

#### Installation

First, clone the LLaMA-Factory repository:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

Navigate to the LLaMA-Factory directory and install the dependencies:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

If you haven't already, download the Yi model from either Huggingface or ModelScope. Here's how to download `Yi-1.5-6B-Chat`:

```bash
# Download from ModelScope
git clone https://www.modelscope.cn/01ai/Yi-1.5-6B-Chat.git 

# Download from Huggingface
git clone https://huggingface.co/01-ai/Yi-1.5-6B-Chat
```

#### Fine-tuning Steps

1. **Create the Configuration File**

   Inside the LLaMA-Factory directory, locate the `llama3_lora_sft_awq.yaml` file under `examples/train_qlora`. Duplicate and rename it to `yi_lora_sft_bitsandbytes.yaml`. 

   This file houses the key parameters for fine-tuning:
     -  `model_name_or_path`: Specify the path to your downloaded Yi model.
     -  `quantization_bit`: Set the model quantization bits.
     -  `dataset`: Choose the dataset for fine-tuning.
     -  `num_train_epochs`: Define the number of training epochs.
     -  `output_dir`:  Specify where to save the fine-tuned model weights.

2. **Configure Parameters**

   Open `yi_lora_sft_bitsandbytes.yaml` and adjust the parameters according to your requirements. Here's an example configuration:

   ```yaml
   ### model
   model_name_or_path: <Path to your downloaded model, e.g., ../Yi-1.5-6B-Chat>
   quantization_bit: 4
   
   ### method
   stage: sft
   do_train: true
   finetuning_type: lora
   lora_target: all
   
   ### dataset
   dataset: identity
   template: yi
   cutoff_len: 1024
   max_samples: 1000
   overwrite_cache: true
   preprocessing_num_workers: 16
   
   ### output
   output_dir: saves/yi-6b/lora/sft
   logging_steps: 10
   save_steps: 500
   plot_loss: true
   overwrite_output_dir: true
   
   ### train
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 8
   learning_rate: 1.0e-4
   num_train_epochs: 3.0
   lr_scheduler_type: cosine
   warmup_ratio: 0.1
   fp16: true
   
   ### eval
   val_size: 0.1
   per_device_eval_batch_size: 1
   eval_strategy: steps
   eval_steps: 500
   ```

   In this example, we use the "identity" dataset, which helps the model recognize itself. If you ask the model "Hello, who are you?", it will respond with its designated name and developer. By replacing this dataset with your own information, you can fine-tune the model to create your personalized AI assistant.

3. **Initiate Fine-tuning**

   Open your terminal and run the following command to start the fine-tuning process (it might take around 10 minutes):

   ```bash
   llamafactory-cli train examples/train_qlora/yi_lora_sft_bitsandbytes.yaml
   ```

#### Inference Testing

1. **Prepare the Inference Configuration**

   Within the LLaMA-Factory folder, find the `llama3_lora_sft.yaml` file under `examples/inference`. Copy and rename it to `yi_lora_sft.yaml`.

   Populate the file with the following content:

   ```yaml
    model_name_or_path: <Same path as before, e.g., ../Yi-1.5-6B-Chat>
    adapter_name_or_path: saves/yi-6b/lora/sft
    template: yi
    finetuning_type: lora
   ```

2. **Run Inference**

   In the terminal where the fine-tuning process finished, execute the inference command:

   ```bash
   llamafactory-cli chat examples/inference/yi_lora_sft.yaml
   ```

Alright, that concludes our tutorial on fine-tuning the Yi model using llamafactory.  Feeling accomplished?  We invite you to explore our other tutorials as well. 

