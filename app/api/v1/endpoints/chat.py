import uuid
import asyncio
from fastapi import APIRouter, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, AsyncSessionLocal
from app.crud.chat import get_or_create_session, create_message
from app.schemas.chat import ChatStreamRequest
from app.services.chat_service import chat_service, AssistantContent
from app.models.chat import MessageRole

router = APIRouter()

async def save_assistant_message(session_id: uuid.UUID, content: str, user_id: str):
    """
    Background task to persist assistant message and update summary if needed.
    """
    if content:
        async with AsyncSessionLocal() as fresh_db:
            # 1. Save the assistant message
            await create_message(fresh_db, session_id, MessageRole.ASSISTANT, content)
            
            # 2. Check if we should update the summary (e.g., every 10 messages)
            from app.crud.chat import get_session_context, update_session_summary
            session = await get_session_context(fresh_db, session_id, user_id, window_size=20)
            
            if len(session.context_messages) >= 10:
                print(f"DEBUG: Triggering summarization for session {session_id}")
                new_summary = await chat_service.summarize_history(
                    session.context_messages, 
                    current_summary=session.summary
                )
                await update_session_summary(fresh_db, session_id, new_summary)
            
            await fresh_db.commit()

@router.post("/stream", status_code=status.HTTP_200_OK)
async def chat_stream(
    request: ChatStreamRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Accept user message, persist to DB, and stream AI response via SSE.
    Includes persistent history and summary for context.
    """
    # 1. Fetch Session Context (including existing summary and last 10 messages)
    from app.crud.chat import get_session_context
    session = await get_session_context(db, request.session_id, request.user_id)
    
    # 2. Persist new user message
    await create_message(db, request.session_id, MessageRole.USER, request.message)
    await db.commit()

    # Tracker object to capture full content across the async generator
    tracker = AssistantContent()

    async def event_generator():
        # 3. Get the stream generator with history and summary injected
        stream = chat_service.stream_chat(
            message=request.message,
            session_id=request.session_id,
            content_tracker=tracker,
            history=session.context_messages,
            summary=session.summary
        )
        
        async for chunk in stream:
            yield chunk

        # 4. Final Persistence: Schedule background persistence
        background_tasks.add_task(save_assistant_message, request.session_id, tracker.content, request.user_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )
