import uuid
from datetime import datetime
from typing import List
from pydantic import BaseModel, ConfigDict
from app.models.chat import MessageRole

class ChatMessageBase(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class SessionHistoryResponse(BaseModel):
    session_id: uuid.UUID
    messages: List[ChatMessageBase]

    model_config = ConfigDict(from_attributes=True)

class SessionDeleteResponse(BaseModel):
    status: str
    message: str
