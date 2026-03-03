from agents import function_tool
from app.services.rag_service import rag_service
from app.services.search_service import search_service

@function_tool
def search_docs(query: str) -> str:
    """
    Search internal repository documentation (README, requirements, etc.).
    Use this ONLY when the user asks about the internal project setup, tech stack, or goal.
    """
    try:
        print(f"RAG: Searching for context: {query}")
        return rag_service.search(query)
    except Exception as e:
        print(f"RAG: Search failed: {e}")
        return f"Error searching documents: {str(e)}"

@function_tool
def search_web(query: str) -> str:
    """
    Search the live internet for real-time information, news, or general knowledge.
    Use this when the user asks about anything outside of this specific repository.
    """
    try:
        print(f"WEB SEARCH: Searching for: {query}")
        return search_service.search(query)
    except Exception as e:
        print(f"WEB SEARCH: Search failed: {e}")
        return f"Error performing web search: {str(e)}"

@function_tool
def get_current_time() -> str:
    """
    Returns the current system date and time. 
    Use this to orient yourself before answering questions about dates, schedules, or time-relative facts.
    """
    from datetime import datetime
    return datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
