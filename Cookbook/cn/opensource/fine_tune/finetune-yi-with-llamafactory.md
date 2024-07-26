### 🌟使用LLaMA-Factory微调

LLaMA Factory是一款开源低代码大模型微调框架，集成了业界广泛使用的微调技术，是北航的博士生郑耀威的杰作。微调的过程很方便，跟着我们一步一步来!

#### 安装

首先我们拉取LLaMA-Factory到本地：

``````
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
``````

安装依赖：

``````
# ⚠️下面两行命令去终端执行⚠️
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
``````

如果你还没有下载yi模型建议从**Huggingface**或者**ModelScope**中下载对应的代码如下：

``````
# 从ModelScope中下载
git clone https://www.modelscope.cn/01ai/Yi-1.5-6B-Chat.git 
# 从Huggingface下载
git clone https://huggingface.co/01-ai/Yi-1.5-6B-Chat
``````

#### 开始微调

1. 创建微调训练相关的配置文件。

   - 在Llama-Factory的文件夹里，打开examples\train_qlora下提供的`llama3_lora_sft_``awq``.yaml`，复制一份并重命名为`yi_lora_sft_bitsandbytes.yaml`。

   - 这个文件里面写着和微调相关的关键参数：比如使用哪个模型？进行什么样的压缩量化？使用什么数据集（这里是identity）？这个数据集学习几遍（num_train_epochs）？微调后的模型权重保存在哪里？

2. `yi_lora_sft_bitsandbytes.yaml`的内容填充为：

   ``````
   ### model
   model_name_or_path: <你下载的模型位置，不要带括号，比如我写了../Yi-1.5-6B-Chat>
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
   ``````

   这里我们使用的identity数据集，俗话说就是“自我认知”数据集，也就是说当你问模型“你好你是谁”的时候，模型会告诉你我叫name由author开发。如果你把数据集更改成你自己的名字，那你就可以微调一个属于你自己的大模型啦。

3. 打开终端terminal，输入以下命令启动微调脚本(大概需要10分钟)：

   ``````bash
   llamafactory-cli train examples/train_qlora/yi_lora_sft_bitsandbytes.yaml
   ``````

#### 推理测试

1. 请参考Llama-Factory文件夹中，examples\inference下提供的`llama3_lora_sft.yaml`，复制一份，并重命名为`yi_lora_sft.yaml`。

  内容填充为：

  ``````
  model_name_or_path: <和之前一样，你下载的模型位置，比如我写了../Yi-1.5-6B-Chat>
  adapter_name_or_path: saves/yi-6b/lora/sft
  template: yi
  finetuning_type: lora
  ``````

2. 回到刚刚结束微调的终端Terminal，运行下面的推理命令：
``````
llamafactory-cli chat examples/inference/yi_lora_sft.yaml
``````

好啦，使用llamafactory微调Yi模型的教程就结束啦，是不是感觉特别有成就感，欢迎继续查看我们其它的教程噢。