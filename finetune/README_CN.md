# Yi-6B及Yi-34B的微调代码

## 准备

### 使用镜像

### 使用本地服务器

推荐使用conda进行开发环境配置，如果您还没有安装conda，参考如下

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

然后创建conda环境

```bash
conda create -n dev_env python=3.10 -y
conda activate dev_env
pip install torch==2.0.1 deepspeed==0.10 tensorboard transformers datasets sentencepiece accelerate ray==2.7
```

## 硬件设置

对于Yi-6B模型，推荐使用4卡，单卡显存大于60GB的节点。

对于Yi-34B模型，由于训练设置采用了zero-offload，会消耗大量CPU内存。因此请使用CUDA_VISIBLE_DEVICES参数来限制GPU使用量（如scripts/run_sft_Yi_34b.sh所示）

对于Yi-34B模型微调训练的典型配置是8卡（仅4卡使用，CUDA_VISIBLE_DEVICES=0,1,2,3），单卡显存大于80GB，整个节点的CPU内存大于900GB。

## 快速开始

将模型下载到本地的MODEL_PATH，典型的模型文件夹如下：

```bash
|-- $MODEL_PATH
|   |-- config.json
|   |-- pytorch_model-00001-of-00002.bin
|   |-- pytorch_model-00002-of-00002.bin
|   |-- pytorch_model.bin.index.json
|   |-- tokenizer_config.json
|   |-- tokenizer.model
|   |-- ...
```

从Huggingface上下载一个数据集到本地DATA_PATH，如Dahoas/rm-static

```bash
|-- $DATA_PATH
|   |-- data
|   |   |-- train-00000-of-00001-2a1df75c6bce91ab.parquet
|   |   |-- test-00000-of-00001-8c7c51afc6d45980.parquet
|   |-- dataset_infos.json
|   |-- README.md
```

/finetune/yi_example_dataset也有一个样例数据集，修改自BAAI/COIG

```bash
|-- $DATA_PATH
|   |--data
|   |   |-- train.jsonl
|   |   |-- eval.jsonl
```

cd进入脚本文件夹，修改脚本当中的MODEL_PATH和DATA_PATH，运行脚本，例如：

```bash
cd Yi/finetune/scripts

bash run_sft_Yi_6b.sh
```

对于Yi-6B基础模型，将training_debug_steps=20和num_train_epochs=4设置为如此的值，就可以输出一个chat模型，端到端预计消耗20分钟。

对于Yi-34B基础模型，初始化阶段会消耗较长时间，请保持耐心。

## 评估输出模型

```bash
cd Yi/finetune/scripts

bash run_eval.sh
```

可以分别打印出base模型和finetune之后模型的输出。
