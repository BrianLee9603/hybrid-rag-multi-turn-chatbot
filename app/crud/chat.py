import uuid
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chat import ChatSession, ChatMessage, MessageRole

from fastapi import HTTPException, status

async def get_or_create_session(
    db: AsyncSession, 
    session_id: uuid.UUID, 
    user_id: str
) -> ChatSession:
    """
    Get an existing session or create a new one if it doesn't exist.
    Must always be scoped by user_id.
    """
    # 1. Try to fetch session by ID globally first to check ownership
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    
    # 2. If session exists, verify owner
    if session:
        if session.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This session ID is already associated with a different user."
            )
        return session
    
    # 3. If not found, create new one
    session = ChatSession(id=session_id, user_id=user_id)
    db.add(session)
    # We don't commit here, we let the caller decide when to commit
    await db.flush()
        
    return session

async def get_session_history(
    db: AsyncSession, 
    session_id: uuid.UUID, 
    user_id: str
) -> Optional[ChatSession]:
    """
    Fetch a session and all its messages, scoped by user_id.
    """
    result = await db.execute(
        select(ChatSession)
        .where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        )
        .options(selectinload(ChatSession.messages))
    )
    return result.scalar_one_or_none()

async def create_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: MessageRole,
    content: str
) -> ChatMessage:
    """Create and persist a new message in a session."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(message)
    await db.flush()
    return message

async def get_session_context(
    db: AsyncSession, 
    session_id: uuid.UUID, 
    user_id: str,
    window_size: int = 10
) -> Optional[ChatSession]:
    """
    Fetch a session and only the most recent N messages for context.
    """
    session = await get_or_create_session(db, session_id, user_id)
    
    # Manually fetch limited messages to avoid full load if history is huge
    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(window_size)
    )
    messages = list(msg_result.scalars().all())
    messages.reverse() # Chronological
    
    # We use a non-persistent attribute to pass these messages back
    # instead of overwriting the relationship which can cause ORM issues
    session.context_messages = messages
    return session

async def update_session_summary(
    db: AsyncSession,
    session_id: uuid.UUID,
    summary: str
):
    """Update the long-term summary for a session."""
    session = await db.get(ChatSession, session_id)
    if session:
        session.summary = summary
        await db.commit()

async def delete_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: str
) -> bool:
    """
    Delete a session and all its messages (via cascade), scoped by user_id.
    """
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        return False
        
    await db.delete(session)
    await db.commit()
    
    return True
