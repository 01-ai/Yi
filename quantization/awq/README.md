# AWQ quantization


[AWQ](https://github.com/mit-han-lab/llm-awq) is a PTQ(Post-Training Quantization)
method. It's an efficient and accurate low-bit weight quantization (INT3/4) for LLMs.

Yi models can be AWQ quantized without a lot of efforts. 
We provide a step-by-step tutorial below.

To run AWQ, we will use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

## Do Quantization

The `quant_autoawq.py` script is provided for you to perform AWQ quantization:

```bash
python quant_autoawq.py --model /base_model \
    --output_dir /quantized_model --bits 4 --group_size 128 --trust_remote_code
```

## Run Quantized Model

You can run a quantized model using the `eval_quantized_model.py`:

```bash
python eval_quantized_model.py --model /quantized_model --trust_remote_code
```
