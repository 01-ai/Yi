import gc
import traceback
import torch
import uvicorn
import time
import uuid
import anyio
import json
from anyio.streams.memory import MemoryObjectSendStream
from functools import lru_cache
from abc import ABC
from threading import Lock, Thread
from types import MethodType
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator, Iterable, AsyncIterator
from loguru import logger
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from sse_starlette import EventSourceResponse
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai.types.model import Model
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, PreTrainedModel
from transformers.generation import GenerationConfig

from utils import (
    Role, 
    ModelList, 
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorCode,
    ErrorResponse,
    model_dump,
    model_parse,
    model_json,
    get_context_length,
    apply_stopping_strings,
    load_model_on_gpus)


llama_outer_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    return ModelList(
    data=[
        Model(
            id="yi",
            object="model",
            created=int(time.time()),
            owned_by="open"
        )
    ]
)


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request
):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await handle_request(request, template.stop)
    request.max_tokens = request.max_tokens or 1024

    params = model_dump(request)
    params.update(dict(echo=False))
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = await run_in_threadpool(_create_chat_completion, params)

    if isinstance(iterator_or_completion, Iterator):
        # It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


def _create_chat_completion(
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[Iterator, ChatCompletion]:
    params = params or {}
    params.update(kwargs)
    return (
        _create_chat_completion_stream(params)
        if params.get("stream", False)
        else _create_chat_completion_non_stream(params)
    )


def _create_chat_completion_stream(params: Dict[str, Any]) -> Iterator:
    """
    Creates a chat completion stream.

    Args:
        params (Dict[str, Any]): The parameters for generating the chat completion.

    Yields:
        Dict[str, Any]: The output of the chat completion stream.
    """
    _id, _created, _model = None, None, None
    has_function_call = False
    for i, output in enumerate(_generate(params)):
        if output["error_code"] != 0:
            yield output
            return

        _id, _created, _model = output["id"], output["created"], output["model"]
        if i == 0:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=""),
                finish_reason=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

        finish_reason = output["finish_reason"]
        if len(output["delta"]) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                _, function_call = template.parse_assistant_response(
                    output["text"], params.get("functions"), params.get("tools"),
                )
            except Exception as e:
                traceback.print_exc()
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict) and "arguments" in function_call:
            has_function_call = True
            function_call = ChoiceDeltaFunctionCall(**function_call)
            delta = ChoiceDelta(
                content=output["delta"],
                function_call=function_call
            )
        elif isinstance(function_call, dict) and "function" in function_call:
            has_function_call = True
            finish_reason = "tool_calls"
            function_call["index"] = 0
            tool_calls = [model_parse(ChoiceDeltaToolCall, function_call)]
            delta = ChoiceDelta(
                content=output["delta"],
                tool_calls=tool_calls,
            )
        else:
            delta = ChoiceDelta(content=output["delta"])

        choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        yield ChatCompletionChunk(
            id=f"chat{_id}",
            choices=[choice],
            created=_created,
            model=_model,
            object="chat.completion.chunk",
        )

    if not has_function_call:
        choice = ChunkChoice(
            index=0,
            delta=ChoiceDelta(),
            finish_reason="stop"
        )
        yield ChatCompletionChunk(
            id=f"chat{_id}",
            choices=[choice],
            created=_created,
            model=_model,
            object="chat.completion.chunk",
        )


def _create_chat_completion_non_stream(params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
    """
    Creates a chat completion based on the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for generating the chat completion.

    Returns:
        ChatCompletion: The generated chat completion.
    """
    last_output = None
    for output in _generate(params):
        last_output = output

    if last_output["error_code"] != 0:
        return create_error_response(last_output["error_code"], last_output["text"])

    function_call, finish_reason = None, "stop"
    if params.get("functions") or params.get("tools"):
        try:
            res, function_call = template.parse_assistant_response(
                last_output["text"], params.get("functions"), params.get("tools"),
            )
            last_output["text"] = res
        except Exception as e:
            traceback.print_exc()
            logger.warning("Failed to parse tool call")

    if isinstance(function_call, dict) and "arguments" in function_call:
        finish_reason = "function_call"
        function_call = FunctionCall(**function_call)
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"],
            function_call=function_call,
        )
    elif isinstance(function_call, dict) and "function" in function_call:
        finish_reason = "tool_calls"
        tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"],
            tool_calls=tool_calls,
        )
    else:
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"].strip(),
        )

    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    usage = model_parse(CompletionUsage, last_output["usage"])
    return ChatCompletion(
        id=f"chat{last_output['id']}",
        choices=[choice],
        created=last_output["created"],
        model=last_output["model"],
        object="chat.completion",
        usage=usage,
    )


def _generate(params: Dict[str, Any]) -> Iterator:
    """
    Generates text based on the given parameters.

    Args:
        params (Dict[str, Any]): A dictionary containing the parameters for text generation.

    Yields:
        Iterator: A dictionary containing the generated text and error code.
    """
    messages = params.get("messages")
    inputs, prompt = _apply_chat_template(
        messages,
        max_new_tokens=params.get("max_tokens", 256),
        functions=params.get("functions"),
        tools=params.get("tools"),
    )

    params.update(dict(inputs=inputs, prompt=prompt))

    try:
        for output in _generate_stream_func(params):
            output["error_code"] = 0
            yield output

    except (ValueError, RuntimeError) as e:
        traceback.print_exc()
        yield {
            "text": f"{e}",
            "error_code": ErrorCode.INTERNAL_ERROR,
        }


def _apply_chat_template(
    messages: List[ChatCompletionMessageParam],
    max_new_tokens: Optional[int] = 256,
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
    """
    Apply chat template to generate model inputs and prompt.

    Args:
        messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
        max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.
        functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
        tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Union[List[int], Dict[str, Any]], Union[str, None]]: Tuple containing the generated inputs and prompt.
    """
    if template.function_call_available:
        messages = template.postprocess_messages(
            messages, functions, tools=tools,
        )
        if functions or tools:
            logger.debug(f"==== Messages with tools ====\n{messages}")

    prompt = template.apply_chat_template(messages)
    inputs = tokenizer(prompt).input_ids
    if isinstance(inputs, list):
        max_src_len = context_len - max_new_tokens - 1
        inputs = inputs[-max_src_len:]
    
    return inputs, prompt



@torch.inference_mode()
def _generate_stream_func(
    params: Dict[str, Any],
):
    input_ids = params.get("inputs")
    functions = params.get("functions")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 40))
    max_new_tokens = int(params.get("max_tokens", 256))

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    stop_strings = params.get("stop", [])

    input_echo_len = len(input_ids)
    device = model.device
    generation_kwargs = dict(
        input_ids=torch.tensor([input_ids], device=device),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature <= 1e-5:
        generation_kwargs["do_sample"] = False
        generation_kwargs.pop("top_k")

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs["streamer"] = streamer

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text, func_call_found = "", False
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        if functions:
            _, func_call_found = apply_stopping_strings(generated_text, ["Observation:"])
        generated_text, stop_found = apply_stopping_strings(generated_text, stop_strings)

        if generated_text and generated_text[-1] != "�":
            delta_text = generated_text[len(previous_text):]
            previous_text = generated_text

            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "delta": delta_text,
                "text": generated_text,
                "logprobs": None,
                "finish_reason": "function_call" if func_call_found else None,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
            }

        if stop_found:
            break

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": generated_text,
        "logprobs": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }


class YiAITemplate(ABC):
    """ https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json """

    name = "yi"
    system_prompt: Optional[str] = ""
    allow_models = ["yi"]
    stop = {
        "strings": ["<|endoftext|>", "<|im_end|>"],
        "token_ids": [2, 6, 7, 8],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
    }
    function_call_available: Optional[bool] = False

    def apply_chat_template(
        self,
        conversation: List[ChatCompletionMessageParam],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a prompt.

        Args:
            conversation (List[ChatCompletionMessageParam]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.

        Returns:
            `str`: A prompt, which is ready to pass to the tokenizer.
        """
        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = _compile_jinja_template(self.template)
        return compiled_template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            system_prompt=self.system_prompt,
        )
    
    @property
    def template(self) -> str:
        return (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    def postprocess_messages(
        self,
        messages: List[ChatCompletionMessageParam],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return messages
    
    def parse_assistant_response(
        self,
        output: str,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        return output, None


@lru_cache
def _compile_jinja_template(chat_template: str):
    """
    Compile a Jinja template from a string.

    Args:
        chat_template (str): The string representation of the Jinja template.

    Returns:
        jinja2.Template: The compiled Jinja template.

    Examples:
        >>> template_string = "Hello, {{ name }}!"
        >>> template = _compile_jinja_template(template_string)
    """
    try:
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("apply_chat_template requires jinja2 to be installed.")

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)


async def handle_request(
        request: Union[CompletionCreateParams, ChatCompletionCreateParams],
        stop: Dict[str, Any] = None
) -> Union[Union[CompletionCreateParams, ChatCompletionCreateParams], JSONResponse]:
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        raise error_check_ret
    
    # stop settings
    _stop, _stop_token_ids = [], []
    if stop is not None:
        _stop_token_ids = stop.get("token_ids", [])
        _stop = stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    if request.functions:
        request.stop.append("Observation:")

    request.stop = list(set(_stop + request.stop))
    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(_stop_token_ids + request.stop_token_ids))

    return request


def check_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is None or isinstance(request.stop, (str, list)):
        return None
    else:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(model_dump(ErrorResponse(message=message, code=code)), status_code=500)
    

async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Union[Iterator, AsyncIterator],
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                if isinstance(chunk, BaseModel):
                    chunk = model_json(chunk)
                elif isinstance(chunk, dict):
                    chunk = json.dumps(chunk, ensure_ascii=False)

                await inner_send_chan.send(dict(data=chunk))

                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()

                if llama_outer_lock.locked():
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
        except anyio.get_cancelled_exc_class() as e:
            logger.info("disconnected")
            with anyio.move_on_after(1, shield=True):
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")
                raise e


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="01-ai/Yi-6B-Chat/",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )
    parser.add_argument(
        "--context_len", type=int, default=None, help="Context length for generating completions."
    )
    parser.add_argument("--disable-gc", action="store_true",
                        help="Disable GC after each response generated.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path
    )

    if args.cpu_only:
        device = "cpu"
    else:
        device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype='auto'
    ).to(device).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path
    )

    context_len = get_context_length(model.config) if args.context_len is None else args.context_len
    template = YiAITemplate()

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)