# LangChain-OpenAILike-adapters Introduction
[中文文档](README_cn.md)

## Motivation

As OpenAI-style APIs have become the industry standard, more and more large model vendors are providing compatible interfaces. However, the current integration methods are fragmented and inefficient. For example, integrating with DeepSeek requires installing `langchain-deepseek`, while integrating with Qwen3 requires relying on `langchain-qwq`. This approach of introducing separate dependency packages for each model not only increases development complexity but also reduces flexibility. A more extreme example is models like Kimi-K2, which do not even have corresponding wrapper packages and can only be accessed through `langchain-openai`.

To address the above issues, we developed this tool library, providing a unified interface function get_openai_like_llm_instance. With just one dependency package, you can access all models compatible with the OpenAI-style API. Using this tool, you can easily integrate with various models. For example:

```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

deepseek_model = get_openai_like_llm_instance(model="deepseek-chat")
deepseek_model.invoke("Hello")
```

> ⚠️ Note: Please ensure that the API key (e.g., `DEEPSEEK_API_KEY`) is properly configured before use.

## Installation

### Install via Pip
```bash
pip install langchain-openailike-llms-adapters
```

### Install via UV
```bash
uv add langchain-openailike-llms-adapters
```

## Usage

In the get_openai_like_llm_instance function, the model parameter is required, while the provider parameter is optional.

### Supported Model Providers

Currently, the following model providers are supported:
- DeepSeek
- DashScope
- TencentCloud
- MoonShot-AI
- Zhipu-AI
- MiniMax

If you do not specify a provider, the tool will automatically determine the provider based on the provided model

| Model Keyword | Provider       |
|---------------|----------------|
| deepseek      | DeepSeek       |
| qwen          | DashScope      |
| hunyuan       | TencentCloud   |
| kimi          | MoonShot       |
| glm           | Zhipu-AI       |
| minimax       | MiniMax        |

### Special Parameter Descriptions

- `enable_thinking`: Applicable only to the Qwen3 series and HunYuan-A13B models.
- `thinking_budget`: Applicable only to the Qwen3 series models.

Other model parameters (such as `temperature`, `top_k`, etc.) can be passed via keyword arguments.

## Custom Providers

For model providers not yet supported, you can use the `provider="custom"` parameter and manually set `CUSTOM_API_BASE` and `CUSTOM_API_KEY`.

For example, to use the Kimi-K2 model on the OpenRouter platform:

```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

model = get_openai_like_llm_instance(
    model="moonshotai/kimi-k2",
    provider="custom"
)
print(model.invoke("Hello"))
```

> ✅ Please set `CUSTOM_API_BASE` to the OpenRouter API address: `https://openrouter.ai/api/v1`.

For locally deployed open-source models, you can also integrate them using the above custom method or by replacing the URL based on existing providers.
```
