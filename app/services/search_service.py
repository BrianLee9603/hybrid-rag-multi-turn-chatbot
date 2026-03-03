from duckduckgo_search import DDGS
from typing import List, Dict

class SearchService:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, max_results: int = 5) -> str:
        """
        Perform a live web search and return formatted results.
        """
        print(f"WEB SEARCH: Searching for: {query}")
        try:
            results = self.ddgs.text(query, max_results=max_results)
            
            if not results:
                return "No web search results found for this query."

            formatted_results = []
            for i, res in enumerate(results):
                title = res.get('title', 'No Title')
                snippet = res.get('body', 'No Content')
                link = res.get('href', 'No Link')
                formatted_results.append(f"[{i+1}] {title}\nContent: {snippet}\nSource: {link}")

            return "\n\n".join(formatted_results)
            
        except Exception as e:
            print(f"WEB SEARCH: Error during search: {e}")
            return f"Error performing web search: {str(e)}"

search_service = SearchService()
