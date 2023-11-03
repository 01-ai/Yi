#/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../sft/"

python prompt_eval.py \
	--model_name_or_path_base=/base_model \
	--model_name_or_path_finetune=/finetuned_model \
	--language Chinese
