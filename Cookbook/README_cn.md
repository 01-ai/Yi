# 👋 Yi Cookbook

<p align="center">
  <a href="./README.md">English</a> | 中文
</p>

欢迎来到 Yi Cookbook！这是一个关于 Yi 模型的综合资源库，包含教程、演示和详细文档。无论您是初学者还是想要研究模型的高阶使用，这里都能找到有价值的信息。

## 最新动态
- **🔥2024-08-09**: 发布 Yi Cookbook 英文版本
- **🔥2024-07-29**: 发布 Yi Cookbook 1.0 版本，包含中文教程和示例。

## OpenSource

| Category                    | Description                                  | Notebook, Markdown                                                                                                | 
|:----------------------------|:---------------------------------------------|:--------------------------------------------------------------------------------------------------------|
| infernce, swift              | 利用 Swift 快速进行 Yi 模型推理。            | [Inference_using_swift.ipynb](./cn/opensource/Inference/Inference_using_swift.ipynb)                    | 
| infernce, transformers       | 使用 Transformers 高效推理 Yi 模型。         | [Inference_using_transformers.ipynb](./cn/opensource/Inference/Inference_using_transformers.ipynb)      |
| infernce, Imdeploy           | 通过 Imdeploy 快速上手 Yi 模型推理。         | [Inference_using_lmdeploy.ipynb](./cn/opensource/Inference/Inference_using_lmdeploy.ipynb)              | 
| infernce, vllm               | 使用 vllm 体验 Yi 模型的快速推理。           | [vLLM_Inference_tutorial.ipynb](./cn/opensource/Inference/vLLM_Inference_tutorial.ipynb)                | 
| quantization, swift          | 使用 Swift轻松量化您的专属 Yi 模型。        | [swift-yi-quantization.md](./cn/opensource/quantization/swift-yi-quantization.md)                       |
| quantization, autoawq        | 利用 autoawq 量化您的 Yi 模型。              | [autoawq-yi-quantization.md](./cn/opensource/quantization/autoawq-yi-quantization.md)                   | 
| quantization, autogptq       | 使用 autogptq 对 Yi 模型进行量化。           | [autogptq-yi-quantization.md](./cn/opensource/quantization/autogptq-yi-quantization.md)                 | 
| fine-tuning, swift          | 使用 Swift 微调，打造您的个性化 Yi 模型。     | [finetune-yi-with-swift.md](./cn/opensource/fine_tune/finetune-yi-with-swift.md)                        |
| fine-tuning, LlaMA-Factory  | 使用 LlaMA-Factory 灵活微调您的 Yi 模型。 | [finetune-yi-with-llamafactory.md](./cn/opensource/fine_tune/finetune-yi-with-llamafactory.md)          | 
| Local Run, ollama           | 使用 ollama 在本地环境中运行 Yi 模型。     | [local-ollama.md](./cn/opensource/local/local-ollama.md)                                                | 
| Local Run, MLX-LM           | 在 MLX-LM 环境下本地运行 Yi 模型。           | [local-mlx.md](./cn/opensource/local/local-mlx.md)                                                      | 
| Local Run, LM Studio        | 使用 LM Studio 轻松本地运行 Yi 模型。         | [local-lm-studio.md](./cn/opensource/local/local-lm-studio.md)                                          |
| Local Run, llama.cpp        | 使用 llama.cpp 在本地运行 Yi 模型。         | [local-llama.cpp.md](./cn/opensource/local/local-llama.cpp.md)                                          | 
| RAG, LlamaIndex             | 基于 Yi 模型和 LlamaIndex 构建强大的 RAG 系统。| [yi_rag_llamaindex.ipynb](./cn/opensource/rag/yi_rag_llamaindex.ipynb)                                  |
| RAG, LangChain              | 使用 LangChain 构建灵活的 RAG 系统。        | [yi_rag_langchain.ipynb](./cn/opensource/rag/yi_rag_langchain.ipynb)                                    | 
| function calling            | 从零开始，实现函数调用。                     | [function_calling.ipynb](./cn/opensource/function_calling/function_calling.ipynb)                       |
| function calling, LlamaIndex | 基于 LlamaIndex，轻松实现函数调用。         | [function_calling_llamaindex.ipynb](./cn/opensource/function_calling/function_calling_llamaindex.ipynb) | 


## API

| Category                   | Description                      | Notebook, Markdown                                                                                         | 
|:---------------------------|:---------------------------------|:-------------------------------------------------------------------------------------------------|
| RAG, LlamaIndex             | 使用 Yi(api) 模型与 LlamaIndex 构建 RAG 应用。 | [yi_rag_llamaindex.ipynb](./cn/api/rag/yi_rag_llamaindex.ipynb)                                  |
| RAG, angChain              | 利用 LangChain，构建基于 Yi API 的 RAG 系统。| [yi_rag_langchain.ipynb](./cn/api/rag/yi_rag_langchain.ipynb)                                    |
| function calling, LlamaIndex | 使用 Yi 模型，基于 LlamaIndex 实现函数调用。   | [function_calling_llamaindex.ipynb](./cn/api/function_calling/function_calling_llamaindex.ipynb) | 

## Ecosystem

| Category    | Description                           | Notebook, Markdown                                                                                         | 
|:------------|:-----------------------------------------------|:-------------------------------------------------------------------------------------------------|
| fine-tuning | 强化 Yi-1.5-6B-Chat 开源模型的数学与逻辑能力。   | [强化Yi-1.5-6B-Chat的数学和逻辑能力.md](./cn/ecosystem/强化Yi-1.5-6B-Chat的数学和逻辑能力.md)                            |
| RAG         | 基于 LlamaIndex 和 Yi-large 构建智能问答系统。 | [基于LlamaIndex和Yi-large构建智能问答系统.md](./cn/ecosystem/基于LlamaIndex和Yi-large构建智能问答系统.md)                                  | 
| demo        | 大模型玩游戏？探索 Yi 玩转街霸三的奥秘！         | [使用Yi大模型玩转街霸三.md](./cn/ecosystem/使用Yi大模型玩转街霸三.md)                                    |
| demo        | 基于 yi-large，打造高效的思维导图生成器。      | [基于yi-large构建思维导图生成器.md](./cn/ecosystem/基于yi-large构建思维导图生成器.md) | 
| fine-tuning | 掌握 yi-vl 微调的最佳实践，事半功倍。         | [yi-vl最佳实践.md](./cn/ecosystem/yi-vl最佳实践.md)                                    |
## 社区贡献

我们热烈欢迎社区贡献！以下是参与方式：

- **问题报告**：发现 bug 或有功能建议？请在 [GitHub Issues](https://github.com/01-ai/Yi/issues) 提交。
- **展示你的作品!**：如果你有基于Yi模型做的有趣的、实用的应用或者教学，我们非常欢迎你提交PR到我们的仓库！请遵循我们的 [贡献指南](./CONTRIBUTING_cn.md)。
