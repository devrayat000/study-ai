import os

INDEX_NAME = "study-bot-openai-embed-v1-index"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
