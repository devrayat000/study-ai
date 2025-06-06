from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from pydantic import Field, BaseModel


# Model for the AI chat request body
class AIRequest(BaseModel):
    """
    Model for the AI chat request body.
    Attributes:
        user_prompt (str): The prompt from the user.
        student_id (str): The ID of the student.
        conversation_id (str): The ID of the conversation.
    """

    user_prompt: str = Field(alias="userPrompt")
    student_id: str = Field(alias="studentId")
    conversation_id: str = Field(alias="conversationId")
    history: Optional[List["GradioChatMessage"]] = Field(default=None, alias="history")


class GradioChatMessage(BaseModel):
    role: str
    content: str


class AIConversations(BaseModel):
    """
    Model for AI conversations.
    Attributes:
        id (str): The ID of the conversation.
        name (Optional[str]): The name of the conversation.
        created_at (datetime): The creation timestamp of the conversation.
        updated_at (datetime): The last updated timestamp of the conversation.
        student_id (Optional[str]): The ID of the student associated with the conversation.
        messages (List[AIConversationsMessages]): List of messages in the conversation.
    """

    id: str = Field(default=None)
    name: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now, alias="createdAt")
    updated_at: datetime = Field(default_factory=datetime.now, alias="updatedAt")
    student_id: Optional[str] = Field(default=None, alias="studentId")

    messages: List["AIConversationsMessages"]

    def __str__(self):
        return f"AIConversations(id={self.id}, createdAt={self.created_at}, updatedAt={self.updated_at}, studentId={self.student_id})"


class AIMessageSender(Enum):
    """
    Enum for message sender types.
    Attributes:
        USER (str): Represents a message sent by the user.
        AI (str): Represents a message sent by the AI.
    """

    USER = "USER"
    AI = "AI"


class AIConversationsMessages(BaseModel):
    id: str = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now, alias="createdAt")
    updated_at: datetime = Field(default_factory=datetime.now, alias="updatedAt")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId")
    content: str = Field(default=None)
    sender: AIMessageSender = Field(default=None)

    # conversation: Optional[AIConversations]

    @property
    def message(self) -> BaseMessage:
        """
        Returns the message object based on the sender type.
        Raises:
            ValueError: If the sender type is invalid.
        """
        if self.sender == AIMessageSender.USER:
            return HumanMessage(content=self.content)
        elif self.sender == AIMessageSender.AI:
            return AIMessage(content=self.content)
        else:
            raise ValueError("Invalid sender type")

    def __str__(self):
        return f"AIConversationsMessages(id={self.id}, createdAt={self.created_at}, updatedAt={self.updated_at}, conversationId={self.conversation_id}, message={self.content})"
