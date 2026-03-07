# Take-Home Assignment — AI Core Service Engineer

**Role:** Backend Engineer (Junior–Mid Level)
**Estimated Time:** 4–6 hours
**Submission:** Link to a private GitHub repository

---

## The Goal

Build a small AI chat service in Python. At its core, we want to see three things:

1. **A working AI agent** — accepts a user message, runs it through an LLM using the OpenAI Agents SDK, and returns a response.
2. **Real-time streaming** — responses stream back to the client via Server-Sent Events (SSE) as they are generated, not as a single payload at the end.
3. **Persistent chat history** — every conversation is stored in a PostgreSQL database, scoped per user and per session.

We will look at your git commit history to understand how you approached the problem progressively. Commit often with clear messages.

---

## What You Will Need

- Your own **OpenAI API key**. (Expected cost: cents for this assignment.)
- **Docker** — a `docker-compose.yml` stub with PostgreSQL wired up will be provided. Extend it to include your FastAPI service.

---

## What to Build

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/chat/stream` | Accept a user message, run the agent, stream the response via SSE |
| `GET` | `/api/v1/sessions/{session_id}/history` | Return a session's full message history |
| `DELETE` | `/api/v1/sessions/{session_id}` | Delete a session and all its messages |
| `GET` | `/api/v1/health` | Health check — returns `200 OK` |

---

### Streaming

`POST /api/v1/chat/stream` must return a `text/event-stream` response. Stream the agent response incrementally — do not buffer the full reply and send it at the end.

Emit at minimum these SSE event types:

| Event | When to emit |
|---|---|
| `agent.message.delta` | Each text chunk as it arrives from the model |
| `agent.message.done` | When the full response is complete |
| `agent.workflow.failed` | On any unhandled error |
| `heartbeat` | Every 15 seconds while the stream is open |

Each event is a JSON object. Example wire format:

```
event: agent.message.delta
data: {"text": "The answer is 42."}

event: agent.message.done
data: {"session_id": "abc-123"}
```

---

### Agent

- Use the **OpenAI Agents SDK** to create and run the agent.
- Run the agent in **streaming mode** so text can be emitted incrementally.
- Write a system prompt that gives the agent a clear, simple persona.

---

### Database

Use **PostgreSQL** (provided via docker-compose). Use **async SQLAlchemy** with the `asyncpg` driver.

**Schema:**

**`chat_sessions`**

| Column | Type |
|---|---|
| `id` | UUID, primary key |
| `user_id` | String |
| `created_at` | Timestamp |
| `updated_at` | Timestamp |

**`chat_messages`**

| Column | Type |
|---|---|
| `id` | UUID, primary key |
| `session_id` | UUID, foreign key → `chat_sessions.id` |
| `role` | Enum: `user` / `assistant` |
| `content` | Text |
| `created_at` | Timestamp |

**Rules:**
- If the `session_id` in the request doesn't exist yet, create it automatically.
- Persist the user message **before** running the agent.
- Persist the assistant reply **after** streaming completes.
- A user must only be able to access their own sessions — always filter queries by both `session_id` and `user_id`.
- Provide **Alembic migrations**. Running `alembic upgrade head` must produce the full schema from scratch.

---

### Tests

Write tests using `pytest`:
- At least one **unit test** — verify SSE events are emitted in the correct order (mock the agent so no real API call is made).
- At least one **integration test** — verify that messages are correctly persisted to the database after a chat turn (use a real test DB, mock the LLM).

---

## Request / Response Contract

```json
POST /api/v1/chat/stream
{
  "session_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
  "user_id":    "user-123",
  "message":    "What is the capital of France?"
}
→ text/event-stream
```

```
GET /api/v1/sessions/{session_id}/history?user_id=user-123
→ {
    "session_id": "...",
    "messages": [
      { "role": "user",      "content": "...", "created_at": "..." },
      { "role": "assistant", "content": "...", "created_at": "..." }
    ]
  }
```

---

## Tech Stack

| Concern | Requirement |
|---|---|
| Language | Python 3.11+ |
| Framework | FastAPI |
| Agent SDK | OpenAI Agents SDK |
| Database | PostgreSQL + async SQLAlchemy (`asyncpg`) |
| Migrations | Alembic |
| Runtime | `docker compose up` starts your service + PostgreSQL |

---

## Deliverables

1. **Private GitHub repo** — share access with us.
2. **`README.md`** — how to run it, how to run tests, and a short write-up (2–3 paragraphs) of your design choices and any trade-offs you made.
3. **Alembic migrations** that build the full schema from `alembic upgrade head`.
4. **Clean git history** — we will read your commits as part of the review.

---

## What We Are Not Looking For

- A frontend or UI.
- Auth middleware — passing `user_id` as a plain string in the request body is fine.
- A deployed or production-hardened system.
- Perfect code — we care more about how you reason about the problem than a polished implementation.
