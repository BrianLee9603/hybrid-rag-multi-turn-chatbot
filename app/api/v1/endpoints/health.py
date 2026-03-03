from fastapi import APIRouter
from app.services.chat_service import chat_service

router = APIRouter()

@router.get("/health")
def health_check():
    """Basic health check endpoint to verify service is alive."""
    return {"status": "ok"}
