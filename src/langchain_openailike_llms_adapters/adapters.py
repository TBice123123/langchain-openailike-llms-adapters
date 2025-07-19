from __future__ import annotations

from functools import cache
from typing import (
    Any,
    Literal,
    Optional,
    Type,
)

from langchain_openai.chat_models.base import BaseChatOpenAI
from .utils import _get_openai_like_chat_model
from .provider import providers

provider_list = Literal[
    "deepseek-ai",
    "dashscope",
    "tencent-cloud",
    "moonshot-ai",
    "zhipu-ai",
    "minimax",
    "custom",
]


def _get_provider_with_model(model: str) -> provider_list:
    if "deepseek" in model.lower():
        return "deepseek-ai"
    elif "qwen" in model.lower():
        return "dashscope"
    elif "hunyuan" in model.lower():
        return "tencent-cloud"
    elif "kimi" in model.lower():
        return "moonshot-ai"
    elif "glm" in model.lower():
        return "zhipu-ai"
    elif "minimax" in model.lower():
        return "minimax"
    else:
        return "custom"


def get_openai_like_llm_instance(
    model: str,
    *,
    provider: Optional[provider_list] = None,
    enable_thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
    **extra_kwargs: Any,
) -> BaseChatOpenAI:
    """
    Get an instance of a chat model that is compatible with the OpenAI API.

    Args:
        model: The model to use.
        provider: The provider to use.
        enable_thinking: Whether to enable thinking. This is only supported by qwen3 model or hunyunan-a13b.
        thinking_budget: The thinking budget. This is only supported by qwen3 model.
        extra_kwargs: Extra keyword arguments to pass to the model.
    Returns:
        An instance of a chat model that is compatible with the OpenAI API.
    """

    if provider is None:
        provider = _get_provider_with_model(model)
    if enable_thinking is not None:
        extra_kwargs.update({"enable_thinking": enable_thinking})
    if thinking_budget is not None:
        extra_kwargs.update({"thinking_budget": thinking_budget})

    chat_model = get_openai_like_llm_chatmodel(provider=provider)

    return chat_model(model=model, **extra_kwargs)


@cache
def get_openai_like_llm_chatmodel(provider: provider_list) -> Type[BaseChatOpenAI]:
    provider_ = providers[provider]

    api_key_name = provider_["api_key"]
    api_base_name = provider_["api_base"]
    default_api_key = provider_["default_url"]
    chat_model_name = provider_["chat_model"]

    _streaming = False

    if provider == "dashscope":
        _streaming = True

    chat_model = _get_openai_like_chat_model(
        chat_model_name,
        api_key_name,
        api_base_name,
        default_api_key,
        provider,
        _streaming,
    )

    return chat_model
