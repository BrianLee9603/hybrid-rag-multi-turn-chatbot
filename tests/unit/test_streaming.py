import json
import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.chat_service import ChatService

class MockStreamEvent:
    def __init__(self, event_type, data):
        self.type = event_type
        self.data = data

@pytest.mark.asyncio
async def test_stream_chat_event_sequence():
    """
    Unit test to verify that SSE events (delta, done) are emitted 
    in the correct sequence.
    """
    service = ChatService()
    session_id = uuid.uuid4()
    message = "Hello"

    # Mock the OpenAI Agent Runner
    from openai.types.responses import ResponseTextDeltaEvent
    
    mock_events = [
        MockStreamEvent("raw_response_event", ResponseTextDeltaEvent(delta="The answer")),
        MockStreamEvent("raw_response_event", ResponseTextDeltaEvent(delta=" is 42.")),
    ]

    async def mock_aiter():
        for event in mock_events:
            yield event

    mock_result = MagicMock()
    mock_result.stream_events.return_value.__aiter__ = mock_aiter

    with patch("agents.Runner.run_streamed", return_value=mock_result):
        # Collect SSE events
        events = []
        async for sse_chunk in service.stream_chat(message, session_id):
            events.append(sse_chunk)

        # 1. Verify sequence
        assert "event: agent.message.delta" in events[0]
        assert "The answer" in events[0]
        
        assert "event: agent.message.delta" in events[1]
        assert " is 42." in events[1]
        
        assert "event: agent.message.done" in events[2]
        assert str(session_id) in events[2]

        # 2. Verify full content buffering
        assert service.get_last_full_content() == "The answer is 42."
