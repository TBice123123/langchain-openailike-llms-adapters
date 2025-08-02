from functools import cache
from json import JSONDecodeError
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import JsonOutputKeyToolsParser, PydanticToolsParser
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import BaseChatOpenAI, _is_pydantic_class
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    create_model,
    model_validator,
)

from langchain_openailike_llms_adapters.provider import providers

from .provider import _get_provider_with_model

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[dict[str, Any], type[_BM], type]
_DictOrPydantic = Union[dict, _BM]



enable_streaming_model = [
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b"
]


support_tool_choice_models = [
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-coder-plus",
    "qwen3-coder-30b-a3b-instruct",
    "qwen-max",
    "qwen-max-latest",
    "qwen-plus",
    "qwen-plus-latest",
    "qwen-turbo",
    "qwen-turbo-latest",
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
    "qwen2.5-14b-instruct-1m",
    "qwen2.5-7b-instruct-1m",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-0.5b-instruct",
]

def _check_support_tool_choice(model: str) -> bool:
    if model in support_tool_choice_models:
        return True
    return False


class ChatCustomOpenAILikeModel(BaseChatOpenAI):
    model_name: str = Field(alias="model", default="")

    api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("CUSTOM_API_KEY", default=None),
    )
    api_base: str = Field(
        default_factory=from_env("CUSTOM_API_BASE", default=""),
    )

    model_config = ConfigDict(populate_by_name=True)
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None

    _api_name: str = PrivateAttr(default="CUSTOM")

    @property
    def _llm_type(self) -> str:
        return f"chat-{self._api_name.lower()}-model"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        key_name = self._api_name
        return {"api_key": f"{key_name}_API_KEY"}

    @model_validator(mode="before")
    @classmethod
    def validate_temperature(cls, values: dict[str, Any]) -> Any:
        model = values.get("model_name") or values.get("model") or ""
        provider = _get_provider_with_model(model)

        if provider == "dashscope" and (
            (model in enable_streaming_model and values.get("enable_thinking") is not False)
            or model.startswith("qwq")
            or model.startswith("qvq")
            or values.get("enable_thinking")
        ):
            values["streaming"] = True

        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment variables."""
        if not self.api_base:
            raise ValueError(
                """Custom models must set api_base or set the CUSTOM_API_BASE environment variable""",
            )

        key_name = f"{self._api_name.upper()}_API_KEY"

        if not (self.api_key and self.api_key.get_secret_value()):
            if self._api_name=="vllm" or self._api_name=="ollama":
                self.api_key=SecretStr("sk-"+self._api_name)
            else:
                raise ValueError(
                    f"If you api_key is not set,  {key_name} environment variable is required",  # noqa: E501
                )


        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type:ignore
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type:ignore
            )
        # Handle use via OpenRouter
        elif hasattr(response.choices[0].message, "model_extra"):  # type:ignore
            model_extra = response.choices[0].message.model_extra  # type:ignore
            if isinstance(model_extra, dict) and (
                reasoning := model_extra.get("reasoning")
            ):
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )

        return rtn

    @property
    def _default_params(self) -> Dict[str, Any]:
        if self.enable_thinking is not None:
            if self.extra_body is None:
                self.extra_body = {"enable_thinking": self.enable_thinking}
            else:
                self.extra_body = {
                    **self.extra_body,
                    "enable_thinking": self.enable_thinking,
                }

        if self.thinking_budget is not None:
            if self.extra_body is None:
                self.extra_body = {"thinking_budget": self.thinking_budget}
            else:
                self.extra_body = {
                    **self.extra_body,
                    "thinking_budget": self.thinking_budget,
                }

        return super()._default_params

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if reasoning_content := top.get("delta", {}).get("reasoning_content"):
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                # Handle use via OpenRouter
                elif reasoning := top.get("delta", {}).get("reasoning"):
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )

        return generation_chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            yield from super()._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"Your {self._api_name} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            async for chunk in super()._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                yield chunk
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"Your {self._api_name} API  returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"Your {self._api_name} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"Your {self._api_name} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        if method != "function_calling":
            method = "function_calling"

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")  # noqa: EM102

        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None.",
                )

            tool_name = convert_to_openai_tool(schema)["function"]["name"]

            tool_choice = _check_support_tool_choice(self.model_name)

            if tool_choice and "qwen" in self.model_name:
                self.enable_thinking = False


            if tool_choice:
                bind_kwargs = self._filter_disabled_params(
                    parallel_tool_calls=False,
                    tool_choice=tool_name,
                    strict=strict,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                )
            else:
                bind_kwargs = self._filter_disabled_params(
                    parallel_tool_calls=False,
                    strict=strict,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                )

            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name,
                    first_tool_only=True,
                )

            if include_raw:
                parser_assign = RunnablePassthrough.assign(
                    parsed=itemgetter("raw") | output_parser,
                    parsing_error=lambda _: None,
                )
                parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                parser_with_fallback = parser_assign.with_fallbacks(
                    [parser_none],
                    exception_key="parsing_error",
                )
                chain = RunnableMap(raw=llm) | parser_with_fallback
            else:
                chain = llm | output_parser

        return chain


class ChatModelExtraParams(TypedDict, total=False):
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int
    n: int
    max_retries: int
    logprobs: bool
    top_logprobs: int
    enable_thinking: bool
    thinking_budget: int
    model_kwargs: dict[str, Any]
    disabled_params: dict[str, Any]
    api_key:SecretStr
    api_base: str 

@cache
def _create_openai_like_chat_model(provider: str) -> Type[ChatCustomOpenAILikeModel]:
    if provider == "custom":
        return ChatCustomOpenAILikeModel

    API_NAME = providers[provider]["api_id"]

    API_KEY_NAME = f"{API_NAME.upper()}_API_KEY"
    API_BASE_NAME = f"{API_NAME.upper()}_API_BASE"

    DEFAULT_API_BASE = providers[provider]["default_url"]

    chat_model_name = f"Chat{API_NAME.title()}Model"

    return create_model(
        chat_model_name,
        api_key=(
            Optional[SecretStr],
            Field(default_factory=secret_from_env(API_KEY_NAME, default=None)),
        ),
        api_base=(
            str,
            Field(default_factory=from_env(API_BASE_NAME, default=DEFAULT_API_BASE)),
        ),
        _api_name=(str, PrivateAttr(default=API_NAME)),
        __base__=ChatCustomOpenAILikeModel,
    )
