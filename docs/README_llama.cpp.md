# Load Yi Series Chat Model with `llama.cpp`
[`llama.cpp`](https://github.com/ggerganov/llama.cpp) is a library that allows you to convert and run LLaMa models using 4-bit integer quantization on MacBook.

## 1. Download `llama.cpp`
Please skip this step if `llama.cpp` is already build. For simplicity, only one building option is shown below. Check the [website](https://github.com/ggerganov/llama.cpp#usage) for more details.
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```
The folder should be like:
```bash
|-- llama.cpp
|   |-- convert.py
|   |-- gguf-py
|   |   |-- examples
|   |   |-- gguf
|   |   |-- scripts
|   |   |-- ...
|   |-- ...
```

## 2. Download Yi Series Model
Please skip this step if the model is already downloaded. Again, other options are provided on the [website](https://github.com/01-ai/Yi#-models).
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B-Chat
```
To install git-lfs:
```bash
brew install git-lfs
```
A typical folder of models is like:
```bash
|-- $MODEL_PATH
|   |-- config.json
|   |-- generation_config.json
|   |-- LICENSE
|   |-- main.py
|   |-- model-00001-of-00003.safetensors
|   |-- model-00002-of-00003.safetensors
|   |-- model-00003-of-00003.safetensors
|   |-- model.safetensors.index.json
|   |-- tokenizer_config.json
|   |-- tokenizer.model
|   |-- ...
```

## 3. Convert and Quantize the Model to 4-bits
Make sure all Python dependencies required by `llama.cpp` are installed:
```bash
cd llama.cpp
python3 -m pip install -r requirements.txt
```
Then, convert the model to gguf FP16 format:
```bash
python3 convert.py $MODEL_PATH
```
Lastly, quantize the model to 4-bits (using q4_0 method):
```bash
./quantize $MODEL_PATH/ggml-model-f16.gguf q4_0
```

## 3. Override EOS Token ID
It seems like the EOS token is converted incorrectly, therefore one additional step needed to reset the EOS token id.
```bash
python3 ./gguf-py/scripts/gguf-set-metadata.py $MODEL_PATH/ggml-model-q4_0.gguf tokenizer.ggml.eos_token_id 7
```

## 4. Run the Model
```bash
./main -m $MODEL_PATH/ggml-model-q4_0.gguf --chatml
```
Finally, you should be able to type your prompts and interact with the model.
