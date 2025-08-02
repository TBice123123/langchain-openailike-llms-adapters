from langchain_openailike_llms_adapters import get_openai_like_embedding
def test_init()->None:
    emb = get_openai_like_embedding("text-embedding-v4", "dashscope")
    assert emb is not None