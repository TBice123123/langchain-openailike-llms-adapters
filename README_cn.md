<h1 align="center"> 🦜️🔗 LangChain-OpenAILike-LLMs-adapters </h1>
<p align="center">
    <em>一个库接入所有兼容OpenAI风格的模型</em>
</p>


## 编写动机
随着 OpenAI 风格 API 成为行业标准，越来越多的大模型厂商提供了兼容的接口。然而，当前的接入方式存在分散且低效的问题。例如接入 DeepSeek 需要安装 `langchain-deepseek`，而接入 Qwen3 则需要依赖 `langchain-qwq`，这种为每个模型单独引入依赖包的方式不仅增加了开发复杂度，也降低了灵活性。更极端的例子是 Kimi-K2 等模型，甚至没有对应的封装包，只能通过 `langchain-openai` 接入。


为了解决上述问题，我们开发了本工具库，提供统一的接口函数 [get_openai_like_llm_instance]，只需一个依赖包即可接入所有兼容 OpenAI 风格的模型 API。通过本工具，你可以轻松接入各类模型，例如：

```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

deepseek_model = get_openai_like_llm_instance(model="deepseek-chat")
deepseek_model.invoke("你好")
```

> ⚠️ 注意：使用前请确保已正确设置 API Key，如 `DEEPSEEK_API_KEY`。

> ⚠️ 注意：如果接入OpenAI的GPT模型，推荐直接使用`langchain-openai`


## 下载方式

### Pip 安装
```bash
pip install langchain-openailike-llms-adapters
```

### UV 安装
```bash
uv add langchain-openailike-llms-adapters
```

## 使用方式

get_openai_like_llm_instance 函数中，model参数为必填项，provider参数可选。

### 支持的模型提供商

目前支持以下模型提供商：
- DeepSeek
- DashScope
- TencentCloud
- MoonShot-AI
- Zhipu-AI
- MiniMax
- VLLM
- Ollama

如果你未指定 provider，工具将根据传入的 model自动判断提供商：
| 模型关键字 | 提供商         | 需要设置的API_KEY |
|------------|----------------|-------------------|
| deepseek   | DeepSeek       | DEEPSEEK_API_KEY  |
| qwen       | DashScope      | DASHSCOPE_API_KEY |
| hunyuan    | TencentCloud   | TENCENT_API_KEY   |
| kimi       | MoonShot       | MOONSHOT_API_KEY  |
| glm        | Zhipu-AI       | ZHIPU_API_KEY     |
| minimax    | MiniMax        | MINIMAX_API_KEY   |

**注意：**
> (1) 对于VLLM和Ollama，请必须传入provider="vllm"或"ollama"
例如
```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

model=get_openai_like_llm_instance(
    model="qwen3:8b",
    provider="ollama"
)
print(model.invoke("你好"))
```

> (2) 对于其他模型参数（如 `temperature`、`top_k` 等），可通过model_kwargs传入。
例如
```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

model=get_openai_like_llm_instance(
    model="qwen3-32b",
    model_kwargs={
      "thinking_budget":10
    }
)
print(model.invoke("你好"))
```

### 视觉模型
同时也支持接入openai兼容的视觉多模态模型，例如

```python
from langchain_core.messages import HumanMessage
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

model=get_openai_like_llm_instance(
    model="qwen2.5-vl-32b-instruct"
)
print(model.invoke(
    input=[
        HumanMessage(
            content=[
                {
                    "type":"image_url",
                    "image_url":"https://example.com/image.png"
                },
                {
                    "type":"text",
                    "text":"图中有什么？"
                }
            ]
        )
    ]
)
)
```

### 向量化模型
本库也提高了兼容OpenAI风格的向量化模型接入，目前支持的提供商有`dashscope`、`zhipu-ai`、`ollama`、`vllm`。
示例代码如下：
```python
from langchain_openailike_llms_adapters import get_openai_like_embedding
emb = get_openai_like_embedding("bge-m3:latest",provider="ollama")
print(emb.embed_query("hello world"))
```


### 自定义提供商

对于尚未支持的模型提供商，你可以使用 `provider="custom"` 参数，并手动设置 `CUSTOM_API_BASE` 和 `CUSTOM_API_KEY`。

例如，使用硅基流动平台的 Kimi-K2 模型：

```python
from langchain_openailike_llms_adapters import get_openai_like_llm_instance

model = get_openai_like_llm_instance(
    model="moonshotai/Kimi-K2-Instruct",
    provider="custom"
)
print(model.invoke("你好"))
```

> ✅ 请将 `CUSTOM_API_BASE` 设置为硅基流动的 API 地址：`https://api.siliconflow.cn/v1`。

对于本地部署的开源模型，也可以通过上述自定义方式接入，或基于已有提供商进行 URL 替换实现接入。
