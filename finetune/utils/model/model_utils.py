import math

from transformers import AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig


def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    ds_config=None,
    rlhf_training=False,
    disable_dropout=False,
    eval_mode=False,
):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    # print(model_config)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        HfDeepSpeedConfig(ds_config)
    else:
        pass

    if not eval_mode:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            use_flash_attention_2=True,
        )
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    print("length of tokenizer is {}".format(len(tokenizer)))
    print(
        "resize_token_embeddings is {}".format(int(8 * math.ceil(len(tokenizer) / 8.0)))
    )

    return model
