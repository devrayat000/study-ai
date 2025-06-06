"""Configuration management for the AI response system."""

import os
from typing import Optional
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for LLM models."""

    retriever_model: str = "phi4-mini"
    generator_model: str = "gemma3:4b"
    retriever_temperature: float = 0.0
    generator_temperature: float = 0.5
    max_tokens: Optional[int] = None


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""

    embedding_model: str = "nomic-embed-text"
    chroma_host: str = "localhost"
    chroma_port: int = 9999
    search_limit: int = 5
    similarity_threshold: float = 0.7
    collection_name: str = "university_study"


class PostgreSQLConfig(BaseModel):
    """Configuration for PostgreSQL database."""

    host: str = "localhost"
    port: int = 5555
    database: str = "chat_history"
    user: str = "postgres"
    password: str = "password"
    table_name: str = "chat_sessions"

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class AppConfig(BaseModel):
    """Main application configuration."""

    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    postgres: PostgreSQLConfig = PostgreSQLConfig()
    index_name: str = "study-bot-openai-embed-v1-index"
    backend_url: str = "http://localhost:5000"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                retriever_model=os.getenv("RETRIEVER_MODEL", "phi4-mini"),
                generator_model=os.getenv("GENERATOR_MODEL", "gemma3:4b"),
                retriever_temperature=float(os.getenv("RETRIEVER_TEMPERATURE", "0.0")),
                generator_temperature=float(os.getenv("GENERATOR_TEMPERATURE", "0.5")),
            ),
            retrieval=RetrievalConfig(
                embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                chroma_host=os.getenv("CHROMA_HOST", "localhost"),
                chroma_port=int(os.getenv("CHROMA_PORT", "9999")),
                search_limit=int(os.getenv("SEARCH_LIMIT", "5")),
                collection_name="university_study",
            ),
            postgres=PostgreSQLConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5555")),
                database=os.getenv("POSTGRES_DATABASE", "chat_history"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                table_name=os.getenv("POSTGRES_TABLE_NAME", "chat_sessions_2"),
            ),
            backend_url=os.getenv("BACKEND_URL", "http://localhost:5000"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global configuration instance
config = AppConfig.from_env()
