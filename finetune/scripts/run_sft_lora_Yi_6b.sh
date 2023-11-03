#/usr/bin/env bash

cd ../sft/

deepspeed main.py \
	--data_path /DATA_PATH/ \
	--model_name_or_path /MODEL_PATH/ \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs 4 \
	--training_debug_steps 20 \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--lora_dim 128 \
	--lora_module_name "layers." \
	--output_dir ./output_Yi_6b_chat_sft_lora
