import torch


# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= 1 - k

        hf_model._num_params = sum(
            [
                p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
                for p in hf_model.parameters()
            ]
        )
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config
        )

        train_tflops = train_flops_per_iteration / (
            e2e_time * gpus_per_model * (10**12)
        )

        param_string = (
            f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        )
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length, hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    # TODO: check hidden_size is not None
    flops_per_iteration = (
        24
        * checkpoint_activations_factor
        * batch_size
        * seq_length
        * num_layers
        * (hidden_size**2)
    ) * (
        1.0
        + (seq_length / (6.0 * hidden_size))
        + (vocab_size / (16.0 * num_layers * hidden_size))
    )
    return flops_per_iteration


def get_hf_configs(hf_config):
    num_layers = getattr(
        hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", None)
    )
    hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size
