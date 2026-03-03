from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
from app.services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Index repository documents for RAG
    try:
        rag_service.reindex_docs()
    except Exception as e:
        print(f"RAG: Error indexing docs on startup: {e}")
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect all routes to the main app instance
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    """Root redirect to health check for quick validation."""
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Visit {settings.API_V1_STR}/health for status."}

if __name__ == "__main__":
    import uvicorn
    # Allow manual execution if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
