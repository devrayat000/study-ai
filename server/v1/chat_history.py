# Local imports
import os, uuid, psycopg
from langchain_community.chat_message_histories.postgres import BaseChatMessageHistory
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
import logging
from .config import config

# Configure logging
logger = logging.getLogger(__name__)

store = {}


def get_conversation_history(
    conversation_id: str | uuid.UUID,
) -> BaseChatMessageHistory:
    """Get a PostgresChatMessageHistory instance with fallback to InMemory."""
    if isinstance(conversation_id, uuid.UUID):
        conversation_id = str(conversation_id)
    print(f"CONVERSATION ID: {conversation_id}")
    if conversation_id not in store:
        try:
            # Try to use PostgreSQL chat history
            store[conversation_id] = PostgresChatMessageHistory(
                config.postgres.table_name,
                conversation_id,
                sync_connection=psycopg.connect(config.postgres.connection_string),
            )
            logger.debug(
                f"Created PostgreSQL chat history for conversation: {conversation_id}"
            )
        except Exception as e:
            print(f"CONV ID: {type(conversation_id)}")
            logger.error(f"Error creating PostgreSQL chat history: {e}")
            # Fallback to in-memory storage
            store[conversation_id] = InMemoryChatMessageHistory()

    return store[conversation_id]


def clear_conversation_history(conversation_id: str | uuid.UUID) -> bool:
    """Clear chat history for a specific conversation."""
    try:
        if isinstance(conversation_id, uuid.UUID):
            conversation_id = str(conversation_id)
        if conversation_id in store:
            history = store[conversation_id]
            history.clear()
            # Also remove from store to ensure clean state
            del store[conversation_id]
            logger.info(f"Cleared chat history for conversation: {conversation_id}")
            return True
        else:
            # Even if not in store, try to clear from PostgreSQL directly
            try:
                conn = psycopg.connect(config.postgres.connection_string)
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM {config.postgres.table_name} WHERE session_id = %s",
                    (conversation_id,),
                )
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(
                    f"Cleared PostgreSQL chat history for conversation: {conversation_id}"
                )
                return True
            except Exception as e:
                logger.error(f"Error clearing PostgreSQL history: {e}")
        return False
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        return False


def get_all_conversations() -> list:
    """Get list of all active conversation IDs from both in-memory store and PostgreSQL."""
    conversation_ids = list(store.keys())
    conversation_ids = [
        str(cid) for cid in conversation_ids
    ]  # Ensure all IDs are strings
    print(f"STORE IDS: {conversation_ids}")

    # Also check PostgreSQL database for existing conversations
    try:
        conn = psycopg.connect(config.postgres.connection_string)
        cursor = conn.cursor()

        # Query for distinct session IDs from the message history table
        cursor.execute(f"SELECT DISTINCT session_id FROM {config.postgres.table_name}")
        db_conversations = [row[0] for row in cursor.fetchall()]
        db_conversations = [
            str(cid) for cid in db_conversations
        ]  # Ensure all IDs are strings
        print(f"DB CONVERSATIONS: {db_conversations}")

        # Combine with in-memory store, avoiding duplicates
        all_conversations = list(set(conversation_ids + db_conversations))

        cursor.close()
        conn.close()
        print(f"Retrieved {len(all_conversations)} conversations from PostgreSQL")
        return all_conversations
    except Exception as e:
        logger.error(f"Error querying PostgreSQL for conversations: {e}")

    return conversation_ids
