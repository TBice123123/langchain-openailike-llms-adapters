from __future__ import annotations

from functools import cache
from typing import (
    Optional,
    Type,
)

from .provider import _get_provider_with_model, provider_list
from .utils import (
    ChatCustomOpenAILikeModel,
    ChatModelExtraParams,
    _create_openai_like_chat_model,
)


def get_openai_like_llm_instance(
    model: str,
    *,
    provider: Optional[provider_list] = None,
    model_kwargs: Optional[ChatModelExtraParams] = None,
) -> ChatCustomOpenAILikeModel:
    """
    Get an instance of a chat model that is compatible with the OpenAI API.

    Args:
        model: The model to use.
        provider: The provider to use.
        model_kwargs: Extra params to pass to the model.
    Returns:
        An instance of a chat model that is compatible with the OpenAI API.
    """

    if provider is None:
        provider = _get_provider_with_model(model)

    model_kwargs = model_kwargs or {}

    chat_model = _create_openai_like_chat_model(provider)

    return chat_model(model=model, **model_kwargs)


@cache
def create_openai_like_chat_model(
    provider: provider_list,
) -> Type[ChatCustomOpenAILikeModel]:
    return _create_openai_like_chat_model(provider)
