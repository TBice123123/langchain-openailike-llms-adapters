"""Test chat model integration."""

from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from langchain_tests.unit_tests.chat_models import generate_schema_pydantic
from pydantic import BaseModel


from langchain_openailike_llms_adapters.adapters import get_openai_like_llm_instance


TEST_PYDANTIC_MODELS = [generate_schema_pydantic()]


def test_init() -> None:
    model = get_openai_like_llm_instance("qwen3-32b")
    assert model is not None


def test_bind_tool_pydantic() -> None:
    model = get_openai_like_llm_instance("qwen3-32b")

    @tool
    def my_adder(a: int, b: int) -> int:
        """Add a and b to result"""
        return a + b

    def my_adder_tool(a: int, b: int) -> int:
        """Add a and b to result"""
        return a + b

    tools = [my_adder_tool, my_adder]

    for pydantic_model in TEST_PYDANTIC_MODELS:
        model_schema = (
            pydantic_model.model_json_schema()
            if hasattr(pydantic_model, "model_json_schema")
            else pydantic_model.schema()
        )
        tools.extend([pydantic_model, model_schema])

    # Doing a mypy ignore here since some of the tools are from pydantic
    # BaseModel 2 which isn't typed properly yet. This will need to be fixed
    # so type checking does not become annoying to users.
    tool_model = model.bind_tools(tools, tool_choice="any")  # type: ignore
    assert isinstance(tool_model, RunnableBinding)


def test_with_structured_output() -> None:
    model = get_openai_like_llm_instance("qwen3-32b")

    class Schema(BaseModel):
        """A simple schema."""

        foo: str

    assert model.with_structured_output(Schema) is not None
    for method in ["json_schema", "function_calling", "json_mode"]:
        strict_values = [None, False, True] if method != "json_mode" else [None]
        for strict in strict_values:
            assert model.with_structured_output(Schema, method=method, strict=strict)  # type:ignore
