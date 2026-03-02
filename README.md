# AI Core Service Engineer — Take-Home Implementation

This repository contains my implementation of the AI chat service. The project focuses on reliable real-time streaming, persistent chat history, and a modular architecture that separates data access from business logic.

## Setup and Execution

### 1. Prerequisites
- Docker and Docker Compose
- OpenAI API Key

### 2. Configuration
Create a `.env` file in the root directory and provide your credentials:
```bash
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/chat_db
```

### 3. Running the Service
To build and start the application along with the PostgreSQL database, use:
```bash
docker compose up --build
```
The service will be accessible at `http://localhost:8000`. You can check the status via the health endpoint at `/api/v1/health`.

### 4. Running the Tests
I have included a suite of integration and unit tests. To run them locally using an in-memory database:
```bash
pytest
```

---

## Architectural Decisions and Rationale

### Service-Repository Pattern
For this project, I chose the Service-Repository pattern to ensure the codebase remains maintainable as complexity grows.
- **Repository Layer (app/crud/):** I encapsulated all database logic within a dedicated CRUD layer. Every query is strictly filtered by both `session_id` and `user_id`. This was a deliberate choice to enforce data isolation at the storage level.
- **Service Layer (app/services/):** The `ChatService` handles the orchestration of the AI Agent and the streaming logic. By decoupling this from the API routes, the core business logic remains testable in isolation from the web framework.

### Concurrent Streaming and Heartbeats
Implementing the required 15-second heartbeat presented a specific concurrency challenge. I chose to use an asynchronous generator that races the next AI event against a heartbeat timer using `asyncio.wait_for`. This ensures that heartbeats are dispatched during periods of LLM latency without blocking the arrival of new message deltas.

### Persistence Strategy
The requirements mandated specific timing for message persistence. I implemented the following flow:
- **Pre-stream:** The user's message is persisted and committed before the AI runner is initialized. This ensures we never lose the user's input even if the LLM call fails.
- **Post-stream:** I utilized FastAPI's `BackgroundTasks` to persist the assistant's full response after the stream closes. This allows the connection to terminate immediately for the user while the database write finishes asynchronously.

### Technical Stack
- **Async SQLAlchemy:** Given the long-lived nature of SSE connections, an asynchronous stack was essential. I used the `asyncpg` driver to ensure the database layer doesn't become a bottleneck during concurrent streaming.
- **Pydantic Settings:** All configuration is managed via Pydantic, ensuring that environment variables are validated at startup.

---

## Project Structure
```text
app/
├── api/            # API routers and SSE endpoint logic
├── core/           # Configuration management and system persona
├── crud/           # Data access layer (scoping and persistence)
├── db/             # Database engine and session management
├── models/         # SQLAlchemy ORM models
├── schemas/        # Pydantic validation schemas
└── services/       # AI orchestration and stream generation
```
