from __future__ import annotations

from functools import cache
from typing import (
    Any,
    Optional,
    Type,
)

from langchain_openai.chat_models.base import BaseChatOpenAI

from .provider import provider_list, _get_provider_with_model
from .utils import _get_openai_like_chat_model


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

    _streaming = False

    if provider == "dashscope" and (
        model.startswith("qwen3")
        or model.startswith("qwq")
        or model.startswith("qvq")
        or enable_thinking
    ):
        _streaming = True

    chat_model = _get_openai_like_chat_model(provider)

    return chat_model(model=model, **extra_kwargs)


@cache
def get_openai_like_llm_chatmodel(provider: provider_list) -> Type[BaseChatOpenAI]:
    return _get_openai_like_chat_model(provider)
