import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.db.session import get_db
from app.crud.chat import get_session_history, delete_session
from app.schemas.chat import SessionHistoryResponse, SessionDeleteResponse

router = APIRouter()

@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def read_session_history(
    session_id: uuid.UUID,
    user_id: str = Query(..., description="User ID associated with the session"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get full message history for a session, scoped by user_id.
    """
    session = await get_session_history(db, session_id, user_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized access"
        )
    
    # Map model to response schema
    return SessionHistoryResponse(
        session_id=session.id,
        messages=session.messages
    )

@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def remove_session(
    session_id: uuid.UUID,
    user_id: str = Query(..., description="User ID associated with the session"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a session and all its messages, scoped by user_id.
    """
    success = await delete_session(db, session_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized access"
        )
        
    return SessionDeleteResponse(
        status="success",
        message=f"Session {session_id} and all related messages deleted."
    )
