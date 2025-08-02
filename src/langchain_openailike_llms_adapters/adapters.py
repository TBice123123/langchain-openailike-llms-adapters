from __future__ import annotations

from functools import cache
from typing import (
    Any,
    Optional,
    Type,
)

from .provider import _get_provider_with_model, provider_emb_list, provider_list
from .utils import (
    ChatCustomOpenAILikeModel,
    ChatModelExtraParams,
    OpenAILikeEmbedding,
    _create_openai_like_chat_model,
    _create_openai_like_embbeding,
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


def get_openai_like_embedding(
    model: str,
    provider: provider_emb_list,
    dimensions: Optional[int] = None,
    chunk_size: Optional[int] = None,
    max_retries: Optional[int] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> OpenAILikeEmbedding:
    """Get an instance of an embedding model that is compatible with the OpenAI API.
    Args:
        model: The model to use.
        provider: The provider to use.
        dimensions: The dimensions of the embedding.
        chunk_size: The size of the chunk to use when embedding.
        max_retries: The maximum number of retries to use when embedding.
        model_kwargs: Extra params to pass to the model.
    Returns:
        An instance of an embedding model that is compatible with the OpenAI API.
    """
    model_kwargs = model_kwargs or {}
    if max_retries:
        model_kwargs["max_retries"] = max_retries
    if dimensions:
        model_kwargs["dimensions"] = dimensions
    if chunk_size:
        model_kwargs["chunk_size"] = chunk_size
    embbeding_model = _create_openai_like_embbeding(provider)

    return embbeding_model(model=model, **model_kwargs)
