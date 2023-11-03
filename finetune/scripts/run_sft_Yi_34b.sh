#/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../sft/"

export CUDA_VISIBLE_DEVICES=0,1,2,3 #limit parallelism to avoid cpu oom

deepspeed main.py \
	--data_path /DATA_PATH/ \
	--model_name_or_path /MODEL_PATH/ \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs 4 \
	--training_debug_steps 50 \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--output_dir /finetuned_model
