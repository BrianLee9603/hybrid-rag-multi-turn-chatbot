import uuid
import json
import asyncio
import time
import os
from typing import AsyncGenerator, Dict, Any, List, Optional
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

from app.core.config import settings
from app.models.chat import MessageRole, ChatMessage

class AssistantContent:
    """Simple class to track content across generator execution."""
    def __init__(self):
        self.content = ""

class ChatService:
    def __init__(self):
        """Initialize the OpenAI Agent."""
        # Ensure the SDK can find the key by setting it in the environment
        if settings.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            
        self.agent = Agent(
            name="Assistant",
            instructions=settings.AGENT_PERSONA,
            model=settings.AGENT_MODEL
        )

    async def summarize_history(self, history: List[ChatMessage], current_summary: Optional[str] = None) -> str:
        """
        Creates a concise summary of the conversation so far.
        Incorporates the existing summary to maintain continuity.
        """
        history_text = "\n".join([f"{m.role.value.upper()}: {m.content}" for m in history])
        
        prompt = (
            "Summarize the following chat history into a single, concise paragraph. "
            "Focus on key facts, user preferences, and the current state of the discussion. "
            "If a summary already exists, incorporate the new information into it.\n\n"
            f"EXISTING SUMMARY: {current_summary or 'None'}\n\n"
            f"NEW MESSAGES:\n{history_text}\n\n"
            "CONCISE SUMMARY:"
        )

        # Use a non-streaming call for the summary
        response = await self.openai_client.chat.completions.create(
            model=self.agent.model,
            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

    def _format_sse(self, event: str, data: Dict[Any, Any]) -> str:
        """Helper to format SSE wire format: event: name\ndata: JSON\n\n"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def stream_chat(
        self, 
        message: str,
        session_id: uuid.UUID,
        content_tracker: AssistantContent,
        history: List[ChatMessage] = None,
        summary: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Orchestrates the SSE stream with injected context.
        """
        start_time = asyncio.get_event_loop().time()
        ttft_recorded = False
        full_assistant_content: List[str] = []

        # 1. Prepare dynamic instructions (Persona + Summary)
        dynamic_instructions = settings.AGENT_PERSONA
        if summary:
            dynamic_instructions += f"\n\nCONTEXT FROM PREVIOUS CONVERSATION:\n{summary}"

        # 2. Format history for context
        history_context = ""
        if history:
            history_context = "PREVIOUS CONVERSATION HISTORY:\n"
            for m in history:
                history_context += f"{m.role.value.upper()}: {m.content}\n"
            history_context += "\n"

        # 3. Initialize Agent with dynamic context
        transient_agent = Agent(
            name=self.agent.name,
            instructions=dynamic_instructions,
            model=self.agent.model
        )

        # 4. Run the agent with context prepended to the current input
        full_input = f"{history_context}USER: {message}"
        
        result = Runner.run_streamed(
            transient_agent, 
            input=full_input
        )
        event_iterator = result.stream_events().__aiter__()

        last_event_time = asyncio.get_event_loop().time()
        heartbeat_interval = 15.0

        is_done = False
        while not is_done:
            try:
                time_to_heartbeat = heartbeat_interval - (asyncio.get_event_loop().time() - last_event_time)

                try:
                    event = await asyncio.wait_for(event_iterator.__anext__(), timeout=max(0.1, time_to_heartbeat))

                    if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                        # Capture TTFT (Time to First Token)
                        if not ttft_recorded:
                            ttft = (asyncio.get_event_loop().time() - start_time) * 1000
                            print(f"EVAL [session={session_id}]: TTFT={ttft:.2f}ms")
                            ttft_recorded = True

                        delta_text = event.data.delta
                        full_assistant_content.append(delta_text)
                        content_tracker.content = "".join(full_assistant_content)

                        last_event_time = asyncio.get_event_loop().time()
                        yield self._format_sse("agent.message.delta", {"text": delta_text})

                except asyncio.TimeoutError:
                    last_event_time = asyncio.get_event_loop().time()
                    yield self._format_sse("heartbeat", {"timestamp": int(time.time())})

                except StopAsyncIteration:
                    is_done = True
                    total_latency = (asyncio.get_event_loop().time() - start_time) * 1000
                    print(f"EVAL [session={session_id}]: Total Latency={total_latency:.2f}ms")
                    yield self._format_sse("agent.message.done", {"session_id": str(session_id)})

            except Exception as e:
                is_done = True
                yield self._format_sse("agent.workflow.failed", {"error": str(e)})
                raise e

chat_service = ChatService()
