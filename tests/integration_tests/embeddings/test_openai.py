"""Test openai embeddings."""
from langchain.callbacks import OpenAICallbackHandler, CallbackManager
from langchain.embeddings.openai import OpenAIEmbeddings


def test_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


def test_openai_embedding_documents_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = OpenAIEmbeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


def test_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


def test_openai_embedding_callback_single_embedding() -> None:
    callback_handler = OpenAICallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    embeddings = OpenAIEmbeddings(callback_manager=callback_manager)
    embeddings.embed_query("hi")
    assert callback_handler.total_tokens == 1
    assert callback_handler.prompt_tokens == 1
    assert callback_handler.completion_tokens == 0


def test_openai_embedding_callback_multiple_embeddings() -> None:
    callback_handler = OpenAICallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    documents = ["foo bar", "bar foo", "foo"]
    embedding = OpenAIEmbeddings(callback_manager=callback_manager, chunk_size=2)
    embedding.embedding_ctx_length = 8191
    embedding.embed_documents(documents)
    assert callback_handler.total_tokens > 1
    assert callback_handler.prompt_tokens > 1
    assert callback_handler.completion_tokens == 0
