import uuid
import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.chat import ChatMessage, MessageRole

class MockStreamEvent:
    def __init__(self, event_type, data):
        self.type = event_type
        self.data = data

@pytest.mark.asyncio
async def test_chat_turn_persists_to_db(client: AsyncClient, db_session: AsyncSession):
    """
    Integration test: verify that both user and assistant messages
    are correctly persisted to the database after a turn.
    """
    session_id = uuid.uuid4()
    user_id = "user-123"
    message = "What is 2+2?"

    # Mock the LLM stream
    from openai.types.responses import ResponseTextDeltaEvent
    mock_events = [
        MockStreamEvent("raw_response_event", ResponseTextDeltaEvent(delta="4")),
    ]
    async def mock_aiter():
        for event in mock_events:
            yield event

    mock_result = MagicMock()
    mock_result.stream_events.return_value.__aiter__ = mock_aiter

    with patch("agents.Runner.run_streamed", return_value=mock_result):
        # 1. Run the chat turn
        response = await client.post(
            "/api/v1/chat/stream",
            json={
                "session_id": str(session_id),
                "user_id": user_id,
                "message": message
            }
        )

        assert response.status_code == 200
        
        # 2. Consume the stream to trigger final persistence
        async for _ in response.aiter_bytes():
            pass

        # 3. Verify DB contents
        # We need to refresh/clear the session to see new changes from the API side
        await db_session.expire_all()
        
        result = await db_session.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == message
        
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[1].content == "4"
