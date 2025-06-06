from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    RunnableMap,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_chroma.vectorstores import Chroma
from .chat_history import get_conversation_history
from .config import config
import chromadb
import logging
import hashlib
from time import time, sleep
from typing import Generator, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Query Contextualization Prompt for Mechanical Engineering
contextualize_q_system_prompt = """
Rewrite the student's latest question so it is clear and self-contained, using any relevant details from the previous chat history. If the question is already clear, return it unchanged.
""".strip()

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Enhanced response template that handles irrelevant context
template = """
Context:
{context}

Student Question:
{input}

As a mechanical engineering tutor, follow these rules:
1. If the context is relevant and helpful for the question, use it to provide a clear answer
2. If the context is not relevant to the question or the question is too generic (like "yes", "ok", "thanks"), politely say you need a specific mechanical engineering question
3. If no useful context is provided, ask for more details about what specific topic they need help with
4. Never make up information that isn't in the context
""".strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful mechanical engineering tutor. Be intelligent about using context - only use it if it's relevant to the student's question. If the context doesn't match the question or the question is too generic, ask for a specific mechanical engineering question instead.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", template),
    ]
)

# LLMs with configuration
retriever_llm = OllamaLLM(
    model=config.llm.retriever_model,
    temperature=config.llm.retriever_temperature,
)
gen_llm = ChatOllama(
    model=config.llm.generator_model,
    temperature=config.llm.generator_temperature,
)

# Vector Store with configuration
persistent_client = chromadb.HttpClient(
    host=config.retrieval.chroma_host,
    port=config.retrieval.chroma_port,
)
vectorstore = Chroma(
    client=persistent_client,
    collection_name=config.retrieval.collection_name,
    embedding_function=OllamaEmbeddings(model=config.retrieval.embedding_model),
)

# Metadata for mechanical engineering content with smart filtering
metadata_field_info = [
    AttributeInfo(
        name="content_type",
        description="The type of the document (either 'books' or 'lectures')",
        type="string",
    ),
    AttributeInfo(
        name="course",
        description="The specific course the document belongs to",
        type="string",
    ),
    AttributeInfo(
        name="course_code",
        description="The unique code of the course",
        type="string",
    ),
    AttributeInfo(
        name="chapter_title",
        description="The chapter or section name",
        type="string",
    ),
    AttributeInfo(
        name="chapter_number",
        description="The chapter or section number",
        type="integer",
    ),
]

document_content_description = "Mechanical engineering textbook and lecture content covering statics, dynamics, strength of materials, thermodynamics, fluid mechanics, heat transfer, machine design, and manufacturing processes"


def create_retriever_chain(vectorstore: Chroma):
    """Create a retriever chain with SelfQueryRetriever for smart filtering of mechanical engineering content."""
    try:
        # Use SelfQueryRetriever for smart filtering
        self_query_retriever = SelfQueryRetriever.from_llm(
            llm=retriever_llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=config.debug,
            enable_limit=True,
            search_kwargs={"k": config.retrieval.search_limit},
            chain_kwargs={
                "allowed_comparators": ChromaTranslator.allowed_comparators,
                "allowed_operators": ChromaTranslator.allowed_operators,
            },
        )

        # Add history awareness to the self-query retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=retriever_llm,
            retriever=self_query_retriever,
            prompt=contextualize_q_prompt,
        )

        return history_aware_retriever

    except Exception as e:
        logger.error(f"Error creating SelfQueryRetriever: {e}")
        # Fallback to basic retriever with history awareness
        basic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.retrieval.search_limit}
        )
        return create_history_aware_retriever(
            llm=retriever_llm,
            retriever=basic_retriever,
            prompt=contextualize_q_prompt,
        )


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, any],
        prompts: list[str],
        **kwargs: any,
    ) -> any:
        formatted_prompts = "\n".join(prompts)
        logger.info(f"Prompt:\n{formatted_prompts}")


def gen_ai_response(
    prompt: str,
    session_id: str,
    vector_db: Optional[Chroma] = None,
) -> Generator[str, None, None]:
    """Generate AI response with streaming support and enhanced features.

    Args:
        prompt: User's input prompt
        session_id: Session identifier for chat history
        vector_db: Optional vector database (if not provided, uses default)

    Yields:
        str: Chunks of the generated response
    """
    start_time = time() if config.debug else None

    try:
        # Use provided vector_db or default vectorstore
        if vector_db is None:
            vector_db = vectorstore

        # Create retriever chain for this request
        history_aware_retriever = create_retriever_chain(vector_db)

        chain = (
            RunnableMap(
                {
                    "input": RunnablePassthrough(),
                    "context": history_aware_retriever,
                    "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
                }
            )
            | prompt_template
            | gen_llm
            | StrOutputParser()
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_conversation_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Stream the response
        chunk_count = 0

        for chunk in chain_with_history.stream(
            {"input": prompt},
            config={
                "configurable": {"session_id": session_id},
                # "callbacks": [CustomHandler()],
            },
        ):
            chunk_count += 1
            yield chunk

            # Add a small delay for better streaming experience
            if config.debug and chunk_count % 10 == 0:
                sleep(0.01)

        if config.debug and start_time:
            elapsed = time() - start_time
            logger.debug(
                f"Response generated in {elapsed:.2f}s with {chunk_count} chunks"
            )

    except Exception as e:
        logger.error(f"Error generating AI response: {e}", exc_info=True)
        error_message = (
            "I apologize, but I encountered an error while processing your request. "
            "Please try rephrasing your question or try again in a moment."
        )
        for char in error_message:
            yield char


if __name__ == "__main__":
    import uuid

    print("🔧 Mechanical Engineering AI Tutor - Test Mode")
    print("Ask questions about mechanics, materials, thermodynamics, etc.")
    print("Type 'exit' to quit\n")

    # Generate a test session ID
    test_session_id = str(uuid.uuid4())

    while True:
        user_input = input("Student: ")
        if user_input.lower() == "exit":
            break

        print("Tutor: ", end="", flush=True)

        # Generate and display the tutor's response
        response_chunks = []
        for chunk in gen_ai_response(user_input, test_session_id):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        print("\n")  # New line after response
