import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from app.core.config import settings

class RAGService:
    def __init__(self):
        # 1. Initialize Persistent Chroma Client
        # This saves the vector data to a local folder named '.db/chroma'
        self.client = chromadb.PersistentClient(path=".db/chroma")
        
        # 2. Use OpenAI for Embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        
        # 3. Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name="repo_docs",
            embedding_function=self.embedding_fn
        )

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better semantic retrieval."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def reindex_docs(self):
        """Scan the repository for documentation and index them."""
        print("RAG: Starting documentation re-indexing...")
        
        # Find all .md and .txt files in root
        files = glob.glob("*.md") + glob.glob("*.txt")
        
        documents = []
        metadatas = []
        ids = []
        
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                filename = os.path.basename(file_path)
                
                chunks = self._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": filename, "chunk": idx})
                    ids.append(f"{filename}_{idx}")

        if documents:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"RAG: Successfully indexed {len(documents)} chunks from {len(files)} files.")
        else:
            print("RAG: No documents found to index.")

    def search(self, query: str, n_results: int = 3) -> str:
        """Query the vector database and return a concatenated string of results."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            context_parts.append(f"--- FROM {source} ---\n{doc}")
            
        return "\n\n".join(context_parts) if context_parts else "No relevant documents found."

rag_service = RAGService()
