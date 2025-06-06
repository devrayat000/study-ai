import os
from typing import Annotated
from fastapi import Depends
import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from .config import config


def get_db(index_name: str = None):
    """Get database connection with configuration support."""
    if index_name is None:
        index_name = config.index_name

    persistent_client = chromadb.HttpClient(port=config.retrieval.chroma_port)
    pc = Chroma(
        client=persistent_client,
        collection_name=index_name,
        embedding_function=OllamaEmbeddings(model=config.retrieval.embedding_model),
    )
    yield pc


ChromaDep = Annotated[Chroma, Depends(get_db)]
