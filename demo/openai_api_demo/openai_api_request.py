from openai import OpenAI

# To start an Latest OpenAI-like Yi server, use the following commands:
#   git clone https://github.com/01-ai/Yi;
#   cd Yi;
#   pip install fastapi uvicorn openai pydantic sse_starlette;
#   python demo/openai_api_demo/openai_api.py;
#
# Then configure the api_base and api_key in your client:
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
)


# List models API
models = client.models.list()
print(models.model_dump())


# Chat completion API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好，请问你是谁？",
        }
    ],
    model="yi",
)
print(chat_completion)


# Stream
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好，请问你是谁？",
        }
    ],
    model="yi",
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)

