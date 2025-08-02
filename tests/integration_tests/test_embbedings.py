"""Test OpenAI Like embeddings."""
from langchain_openailike_llms_adapters import get_openai_like_embedding


def test_langchain_openai_embedding_documents() -> None:
    documents = ["foo bar"]
    embedding = get_openai_like_embedding("text-embedding-v4", provider="dashscope")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_openai_embedding_query() -> None:
    document = "foo bar"
    embedding = get_openai_like_embedding("text-embedding-v4", provider="dashscope")
    output = embedding.embed_query(document)
    assert len(output) > 0


def test_langchain_openai_embeddings_dimensions() -> None:
    documents = ["foo bar"]
    embedding = get_openai_like_embedding("text-embedding-v4", provider="dashscope",dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 128



def test_langchain_openai_embeddings_dimensions_large_num() -> None:
    documents = [f"foo bar {i}" for i in range(100)]
    embedding = get_openai_like_embedding("text-embedding-v4", provider="dashscope",dimensions=128,chunk_size=10)
    output = embedding.embed_documents(documents)
    assert len(output) == 100
    assert len(output[0]) == 128


