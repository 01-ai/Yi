import pydantic
import torch
from enum import Enum, IntEnum
from pydantic import BaseModel
from typing import (
    Dict, 
    List, 
    Any, 
    Literal, 
    Optional, 
    Union,
    cast,
    Type,
    Tuple
)

from transformers import AutoModelForCausalLM

from accelerate import dispatch_model

from openai.types.model import Model
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.create_embedding_response import Usage


# --------------- Pydantic v2 compatibility ---------------

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


def model_json(model: pydantic.BaseModel, **kwargs) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)  # type: ignore


def model_dump(model: pydantic.BaseModel, **kwargs) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return model.model_dump(**kwargs)
    return cast(
        "dict[str, Any]",
        model.dict(**kwargs),
    )


def model_parse(model: Type[pydantic.BaseModel], data: Any) -> pydantic.BaseModel:
    if PYDANTIC_V2:
        return model.model_validate(data)
    return model.parse_obj(data)  # pyright: ignore[reportDeprecated]


def disable_warnings(model: Type[pydantic.BaseModel]):
    # Disable warning for model_name settings
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()

def parse_messages(
    messages: List[ChatCompletionMessageParam], split_role="user"
) -> Tuple[str, List[List[ChatCompletionMessageParam]]]:
    """
    Parse a list of chat completion messages into system and rounds.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        split_role: The role at which to split the rounds. Defaults to Role.USER.

    Returns:
        Tuple[str, List[List[ChatCompletionMessageParam]]]: A tuple containing the system message and a list of rounds.
    """
    system, rounds = "", []
    r = []
    for i, message in enumerate(messages):
        if message["role"] == "system":
            system = message["content"]
            continue
        if message["role"] == split_role and r:
            rounds.append(r)
            r = []
        r.append(message)
    if r:
        rounds.append(r)
    return system, rounds


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


class ChatCompletionCreateParams(BaseModel):
    messages: List[ChatCompletionMessageParam]
    """A list of messages comprising the conversation so far.

    [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
    """

    model: str
    """ID of the model to use.

    See the
    [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    table for details on which models work with the Chat API.
    """

    frequency_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    function_call: Optional[FunctionCall] = None
    """Deprecated in favor of `tool_choice`.

    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via `{"name": "my_function"}` forces the model to call that
    function.

    `none` is the default when no functions are present. `auto`` is the default if
    functions are present.
    """

    functions: Optional[List] = None
    """Deprecated in favor of `tools`.

    A list of functions the model may generate JSON inputs for.
    """

    logit_bias: Optional[Dict[str, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the
    tokenizer) to an associated bias value from -100 to 100. Mathematically, the
    bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model, but values between -1 and 1 should decrease or
    increase likelihood of selection; values like -100 or 100 should result in a ban
    or exclusive selection of the relevant token.
    """

    max_tokens: Optional[int] = None
    """The maximum number of [tokens](/tokenizer) to generate in the chat completion.

    The total length of input tokens and generated tokens is limited by the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    """How many chat completion choices to generate for each input message."""

    presence_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    response_format: Optional[ResponseFormat] = None
    """An object specifying the format that the model must output.

    Used to enable JSON mode.
    """

    seed: Optional[int] = None
    """This feature is in Beta.

    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result. Determinism is not guaranteed, and you should refer to the
    `system_fingerprint` response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""

    temperature: Optional[float] = 0.9
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    """
    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via
    `{"type: "function", "function": {"name": "my_function"}}` forces the model to
    call that function.

    `none` is the default when no functions are present. `auto` is the default if
    functions are present.
    """

    tools: Optional[List] = None
    """A list of tools the model may call.

    Currently, only functions are supported as a tool. Use this to provide a list of
    functions the model may generate JSON inputs for.
    """

    top_p: Optional[float] = 1.0
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    best_of: Optional[int] = 1

    ignore_eos: Optional[bool] = False

    use_beam_search: Optional[bool] = False

    stop_token_ids: Optional[List[int]] = None

    skip_special_tokens: Optional[bool] = True

    spaces_between_special_tokens: Optional[bool] = True

    min_p: Optional[float] = 0.0


class CompletionCreateParams(BaseModel):
    model: str
    """ID of the model to use.

    You can use the
    [List models](https://platform.openai.com/docs/api-reference/models/list) API to
    see all of your available models, or see our
    [Model overview](https://platform.openai.com/docs/models/overview) for
    descriptions of them.
    """

    prompt: Union[str, List[str], List[int], List[List[int]], None]
    """
    The prompt(s) to generate completions for, encoded as a string, array of
    strings, array of tokens, or array of token arrays.

    Note that <|endoftext|> is the document separator that the model sees during
    training, so if a prompt is not specified the model will generate as if from the
    beginning of a new document.
    """

    best_of: Optional[int] = 1
    """
    Generates `best_of` completions server-side and returns the "best" (the one with
    the highest log probability per token). Results cannot be streamed.

    When used with `n`, `best_of` controls the number of candidate completions and
    `n` specifies how many to return – `best_of` must be greater than `n`.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion"""

    frequency_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    logit_bias: Optional[Dict[str, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the GPT
    tokenizer) to an associated bias value from -100 to 100. You can use this
    [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
    convert text to token IDs. Mathematically, the bias is added to the logits
    generated by the model prior to sampling. The exact effect will vary per model,
    but values between -1 and 1 should decrease or increase likelihood of selection;
    values like -100 or 100 should result in a ban or exclusive selection of the
    relevant token.

    As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
    from being generated.
    """

    logprobs: Optional[int] = None
    """
    Include the log probabilities on the `logprobs` most likely tokens, as well the
    chosen tokens. For example, if `logprobs` is 5, the API will return a list of
    the 5 most likely tokens. The API will always return the `logprob` of the
    sampled token, so there may be up to `logprobs+1` elements in the response.

    The maximum value for `logprobs` is 5.
    """

    max_tokens: Optional[int] = 16
    """The maximum number of [tokens](/tokenizer) to generate in the completion.

    The token count of your prompt plus `max_tokens` cannot exceed the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    """How many completions to generate for each prompt.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    presence_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    seed: Optional[int] = None
    """
    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result.

    Determinism is not guaranteed, and you should refer to the `system_fingerprint`
    response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens.

    The returned text will not contain the stop sequence.
    """

    suffix: Optional[str] = None
    """The suffix that comes after a completion of inserted text."""

    temperature: Optional[float] = 1.
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    top_p: Optional[float] = 1.
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    ignore_eos: Optional[bool] = False

    use_beam_search: Optional[bool] = False

    stop_token_ids: Optional[List[int]] = None

    skip_special_tokens: Optional[bool] = True

    spaces_between_special_tokens: Optional[bool] = True

    min_p: Optional[float] = 0.0


class Embedding(BaseModel):
    embedding: Any
    """The embedding vector, which is a list of floats.

    The length of vector depends on the model as listed in the
    [embedding guide](https://platform.openai.com/docs/guides/embeddings).
    """

    index: int
    """The index of the embedding in the list of embeddings."""

    object: Literal["embedding"]
    """The object type, which is always "embedding"."""


class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]
    """The list of embeddings generated by the model."""

    model: str
    """The name of the model used to generate the embedding."""

    object: Literal["list"]
    """The object type, which is always "list"."""

    usage: Usage
    """The usage information for the request."""


# Models don't use the same configuration key for determining the maximum
# sequence length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important.  Some models have two of these, and we
# have a preference for which value gets used.
SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config) -> int:
    """ Get the context length of a model from a huggingface model config. """
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_scaling_factor = config.rope_scaling["factor"] if rope_scaling else 1
    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


def apply_stopping_strings(reply: str, stop_strings: List[str]) -> Tuple[str, bool]:
    """
    Apply stopping strings to the reply and check if a stop string is found.

    Args:
        reply (str): The reply to apply stopping strings to.
        stop_strings (List[str]): The list of stopping strings to check for.

    Returns:
        Tuple[str, bool]: A tuple containing the modified reply and a boolean indicating if a stop string was found.
    """
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found





def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': num_gpus-1
    }

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus-1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name_or_path, num_gpus: int = 2):
    num_devices = torch.cuda.device_count()

    if num_gpus == 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',
                                                     trust_remote_code=True).eval()
    elif 1 < num_gpus <= num_devices:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu',
                                                     trust_remote_code=True).eval()
        num_layers = model.config.num_hidden_layers
        device_map = _device_map(num_gpus, num_layers)
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
    else:
        raise KeyError

    return model