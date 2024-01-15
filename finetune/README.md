# Finetune code for Yi 6B and 34B

## Preparation

### From Image

By default, we use a small dataset from [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) to finetune the base model.
You can also prepare your customized dataset in the following `jsonl` format:

```json
{ "prompt": "Human: Who are you? Assistant:", "chosen": "I'm Yi." }
```

And then mount them in the container to replace the default ones:

```bash
docker run -it \
    -v /path/to/save/finetuned/model/:/finetuned-model \
    -v /path/to/train.jsonl:/yi/finetune/data/train.json \
    -v /path/to/eval.jsonl:/yi/finetune/data/eval.json \
    ghcr.io/01-ai/yi:latest \
    bash finetune/scripts/run_sft_Yi_6b.sh
```

### From Local Server

Make sure you have conda. If not, use

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

Then, create a conda env:

```bash
conda create -n dev_env python=3.10 -y
conda activate dev_env
pip install torch==2.0.1 deepspeed==0.10 tensorboard transformers datasets sentencepiece accelerate ray==2.7
```

## Hardware Setup

For the Yi-6B model, a node with 4 GPUs, each has GPU mem larger than 60GB is recommended.

For the Yi-34B model, because the usage of zero-offload technique takes a lot CPU memory, please be careful to limit the GPU numbers in 34B finetune training. Please use CUDA_VISIBLE_DEVICES to limit the GPU number (as shown in scripts/run_sft_Yi_34b.sh).

A typical hardware setup for finetuning 34B model is a node with 8GPUS (limit to 4 in running by CUDA_VISIBLE_DEVICES=0,1,2,3), each has GPU mem larger than 80GB, with total CPU mem larger than 900GB.

## Quick Start

Download a LLM-base model to MODEL_PATH (6B and 34B). A typical folder of models is like:

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

Download a dataset from huggingface to local storage DATA_PATH, e.g. Dahoas/rm-static.

```bash
|-- $DATA_PATH
|   |-- data
|   |   |-- train-00000-of-00001-2a1df75c6bce91ab.parquet
|   |   |-- test-00000-of-00001-8c7c51afc6d45980.parquet
|   |-- dataset_infos.json
|   |-- README.md
```

`finetune/yi_example_dataset` has example datasets, which are modified from [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)

```bash
|-- $DATA_PATH
    |--data
        |-- train.jsonl
        |-- eval.jsonl
```

`cd` into the scripts folder, copy and paste the script, and run. For example:

```bash
cd finetune/scripts

bash run_sft_Yi_6b.sh
```

For the Yi-6B base model, setting training_debug_steps=20 and num_train_epochs=4 can output a chat model, which takes about 20 minutes.

For the Yi-34B base model, it takes a relatively long time for initialization. Please be patient.

## Evaluation

```bash
cd finetune/scripts

bash run_eval.sh
```

Then you'll see the answer from both the base model and the finetuned model
