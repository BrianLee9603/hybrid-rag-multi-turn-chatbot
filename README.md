# AI Core Service Engineer — Take-Home Implementation

This repository contains my implementation of the AI chat service. The project focuses on reliable real-time streaming, persistent chat history, and a modular architecture that separates data access from business logic.

## Setup and Execution

### 1. Prerequisites
- Docker and Docker Compose
- OpenAI API Key
- Tavily API Key (for web search augmentation)

### 2. Installation
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/BinhOiDungNghien/VCAPTECH_ASSIGNMENT.git
cd VCAPTECH_ASSIGNMENT
```

### 3. Configuration
Create a .env file in the root directory and provide your credentials:
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/chat_db
```

### 3. Running the Service
To build and start the application along with the PostgreSQL database, use:
```bash
docker compose up --build
```
The service will be accessible at http://localhost:8000. You can check the status via the health endpoint at /api/v1/health.

### 4. Running the Tests
I have included a suite of integration and unit tests. To run them locally using an in-memory database:
```bash
pytest
```

---

## Architectural Decisions and Rationale

### Service-Repository Pattern
I chose the Service-Repository pattern to ensure the codebase remains maintainable as complexity grows.
- Repository Layer (app/crud/): Encapsulates all database logic. Every query is strictly filtered by both session_id and user_id to enforce data isolation.
- Service Layer (app/services/): Handles the orchestration of the AI Agent and the streaming logic. By decoupling this from the API routes, the core business logic remains testable in isolation.

### Multi-turn Memory: Summary Buffer Strategy
To ensure the AI maintains continuity in long conversations without exceeding token limits, I implemented a hybrid memory system. The service retains a precise window of the last 10 messages while periodically condensing older history into a persistent summary. This summary is stored in the database and injected into the AI's persona, providing a balance of short-term nuance and long-term recall.

### Agentic Knowledge Augmentation
The system uses an autonomous tool-calling pattern. Rather than injecting all possible information into the prompt, the Agent is equipped with specialized tools:
- Local RAG: A semantic search tool using ChromaDB to query local repository documentation.
- Web Search: A high-precision search tool powered by Tavily for real-time information.
- Temporal Grounding: A tool that provides the current system time, ensuring the AI can correctly process date-relative queries.

### Concurrent Streaming and Performance
- Asynchronous Core: The service is built entirely on asynchronous patterns (FastAPI, SQLAlchemy with asyncpg) to handle long-lived SSE connections efficiently.
- Heartbeat Mechanism: I used an asynchronous generator that races the next AI event against a heartbeat timer using asyncio.wait_for. This ensures that heartbeats are dispatched during periods of LLM latency.
- Background Persistence: User messages are persisted immediately, while assistant responses are saved via FastAPI BackgroundTasks. This allows the connection to terminate immediately for the user while the database write finishes asynchronously.

---

## Basic UI/UX Implementation

Although the assignment focused on the backend, I have included a single-file web interface (chat_ui.html) to demonstrate the service's capabilities in a real-world context. Just need to open the file with your native browser.
- Real-time Markdown: AI responses are rendered with full Markdown support, including code blocks and formatting.
- Session Management: Users can generate new session IDs or load history from the PostgreSQL database directly through the UI.
- Telemetry Display: Metadata such as token counts and finish reasons are visible when loading session history.

---

## Project Structure
```text
app/
├── api/            # API routers and SSE endpoint logic
├── core/           # Configuration management and system persona
├── crud/           # Data access layer (scoping and persistence)
├── db/             # Database engine and connection pooling
├── models/         # SQLAlchemy ORM models
├── schemas/        # Pydantic validation schemas
└── services/       # AI orchestration, RAG, and Web Search logic
```
