# Text Generation Task

To run text generation task in the streaming mode:

```shell
python text_generation.py \
    --model 01-ai/Yi-6B \
    --tokenizer 01-ai/Yi-6B \
    --max-tokens 512 \
    --eos-token $'\n' \
    --streaming
```

You can also provide an extra `--prompt` argument to try some other prompts.

When dealing with extremely long input sequences, you may need multiple GPU devices and to enable tensor parallelism acceleration during inference to avoid insufficient memory error.

To run text generation task using tensor parallelism acceleration with 2 GPU devices:

```shell
torchrun --nproc_per_node 2 \
    text_generation_tp.py \
    --model 01-ai/Yi-6B \
    --max-tokens 512 \
    --eos-token $'\n' \
    --streaming

```
