import uuid
import json
import asyncio
import time
import os
import tiktoken
from typing import AsyncGenerator, Dict, Any, List, Optional
from agents import Agent, Runner, function_tool
from openai.types.responses import ResponseTextDeltaEvent

from app.core.config import settings
from app.models.chat import MessageRole, ChatMessage

class AssistantContent:
    """Simple class to track content and metadata across generator execution."""
    def __init__(self):
        self.content = ""
        self.finish_reason = "stop"
        self.token_count = 0

class ChatService:
    def __init__(self):
        """Initialize the OpenAI Agent."""
        # Ensure the SDK can find the key by setting it in the environment
        if settings.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            
        # Import tools from the dedicated tools module
        from app.services.tools import search_docs, search_web, get_current_time

        self.agent = Agent(
            name="Assistant",
            instructions=(
                f"{settings.AGENT_PERSONA}\n\n"
                "You have three specialized tools:\n"
                "1. search_docs: Use for project-specific info.\n"
                "2. search_web: Use for real-time or general web knowledge.\n"
                "3. get_current_time: Use to check today's date.\n\n"
                "IMPORTANT: If a user asks about a specific date, you MUST use the get_current_time tool first to orient yourself before searching or answering."
            ),
            model=settings.AGENT_MODEL,
            tools=[search_docs, search_web, get_current_time]
        )

    async def summarize_history(self, history: List[ChatMessage], current_summary: Optional[str] = None) -> str:
        """
        Creates a concise summary of the conversation so far.
        """
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        history_text = "\n".join([f"{m.role.value.upper()}: {m.content}" for m in history])
        
        prompt = (
            "Summarize the following chat history into a single, concise paragraph. "
            "Focus on key facts and user preferences.\n\n"
            f"EXISTING SUMMARY: {current_summary or 'None'}\n\n"
            f"NEW MESSAGES:\n{history_text}\n\n"
            "CONCISE SUMMARY:"
        )

        response = await client.chat.completions.create(
            model=self.agent.model,
            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

    def _format_sse(self, event: str, data: Dict[Any, Any]) -> str:
        """Helper to format SSE wire format."""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _count_tokens(self, text: str) -> int:
        """Calculate token count locally using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.agent.model)
            return len(encoding.encode(text))
        except:
            return len(text) // 4

    async def stream_chat(
        self, 
        message: str,
        session_id: uuid.UUID,
        content_tracker: AssistantContent,
        history: List[ChatMessage] = None,
        summary: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Orchestrates the SSE stream with metadata tracking.
        """
        start_time = asyncio.get_event_loop().time()
        ttft_recorded = False
        full_assistant_content: List[str] = []

        # 1. Prepare dynamic instructions (Fresh Persona with Date + Summary)
        from datetime import datetime
        current_date_str = datetime.now().strftime("%A, %B %d, %Y")
        
        # Inject the real date into the template from settings
        persona_with_date = settings.AGENT_PERSONA.format(current_date=current_date_str)
        
        dynamic_instructions = (
            f"{persona_with_date}\n\n"
            f"CRITICAL TEMPORAL CONTEXT: The absolute current date is {current_date_str}. "
            "Any date prior to this is the PAST. Any date after this is the FUTURE. "
            "When answering questions about specific dates, use this context to correctly identify if an event has already happened."
        )
        if summary:
            dynamic_instructions += f"\n\nLONG-TERM MEMORY (SUMMARY):\n{summary}"

        # 2. Format history
        history_context = ""
        if history:
            history_context = "PREVIOUS CONVERSATION HISTORY:\n"
            for m in history:
                history_context += f"{m.role.value.upper()}: {m.content}\n"
            history_context += "\n"

        # 3. Initialize Agent
        # We use transient agent to pass dynamic instructions
        transient_agent = Agent(
            name=self.agent.name,
            instructions=dynamic_instructions,
            model=self.agent.model,
            tools=self.agent.tools # Pass tools to the transient agent
        )

        # 4. Run agent
        full_input = f"{history_context}USER: {message}"
        result = Runner.run_streamed(transient_agent, input=full_input)
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
                        if not ttft_recorded:
                            ttft = (asyncio.get_event_loop().time() - start_time) * 1000
                            print(f"EVAL [session={session_id}]: TTFT={ttft:.2f}ms")
                            ttft_recorded = True

                        delta_text = event.data.delta
                        full_assistant_content.append(delta_text)
                        content_tracker.content = "".join(full_assistant_content)
                        
                        last_event_time = asyncio.get_event_loop().time()
                        yield self._format_sse("agent.message.delta", {"text": delta_text})
                    
                    if event.type == "raw_response_event" and hasattr(event.data, "finish_reason") and event.data.finish_reason:
                        content_tracker.finish_reason = event.data.finish_reason

                except asyncio.TimeoutError:
                    last_event_time = asyncio.get_event_loop().time()
                    yield self._format_sse("heartbeat", {"timestamp": int(time.time())})

                except StopAsyncIteration:
                    is_done = True
                    content_tracker.token_count = self._count_tokens(content_tracker.content)
                    total_latency = (asyncio.get_event_loop().time() - start_time) * 1000
                    print(f"EVAL [session={session_id}]: Total={total_latency:.2f}ms | Tokens={content_tracker.token_count}")
                    yield self._format_sse("agent.message.done", {"session_id": str(session_id)})

            except Exception as e:
                is_done = True
                yield self._format_sse("agent.workflow.failed", {"error": str(e)})
                raise e

chat_service = ChatService()
