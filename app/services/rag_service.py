import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from app.core.config import settings

class RAGService:
    def __init__(self):
        # 1. Initialize Persistent Chroma Client
        # Use path from settings for environment flexibility
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        
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

        # 4. Initialize Reranker (Cross-Encoder)
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight but powerful reranker
            self.reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
            print("RAG: Reranker initialized successfully.")
        except Exception as e:
            print(f"RAG: Warning - Reranker failed to initialize: {e}")
            self.reranker = None

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """
        Split text into overlapping chunks using a recursive approach.
        Accumulates parts until chunk_size is reached to avoid tiny fragments.
        """
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        def split_recursive(text_to_split: str, seps: List[str]) -> List[str]:
            if len(text_to_split) <= chunk_size:
                return [text_to_split]
            
            # Find the best separator
            separator = seps[0] if seps else ""
            for s in seps:
                if s in text_to_split:
                    separator = s
                    break
            
            # Split by separator
            if separator:
                splits = text_to_split.split(separator)
            else:
                # No more separators, hard split
                return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size)]
            
            final_chunks = []
            current_chunk = ""
            
            for i, part in enumerate(splits):
                # Add the separator back except for the last element
                if i < len(splits) - 1:
                    part += separator
                
                if len(current_chunk) + len(part) <= chunk_size:
                    current_chunk += part
                else:
                    # Current part doesn't fit
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    
                    # If the part itself is too big, recurse deeper
                    if len(part) > chunk_size:
                        remaining_seps = seps[seps.index(separator)+1:]
                        final_chunks.extend(split_recursive(part, remaining_seps))
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                final_chunks.append(current_chunk)
            
            return final_chunks

        raw_chunks = split_recursive(text, separators)
        
        # Apply overlap properly
        if overlap <= 0 or len(raw_chunks) <= 1:
            return [c.strip() for c in raw_chunks if c.strip()]
            
        final_results = []
        for i in range(len(raw_chunks)):
            chunk = raw_chunks[i]
            if i > 0:
                # Take the end of the previous chunk
                prev_chunk = raw_chunks[i-1]
                overlap_prefix = prev_chunk[-overlap:]
                chunk = overlap_prefix + chunk
            
            final_results.append(chunk.strip())
            
        return [c for c in final_results if c]

    def _read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"RAG: Error reading PDF {file_path}: {e}")
            return ""

    def reindex_docs(self):
        """Scan the repository for documentation and index them."""
        print(f"RAG: Starting documentation re-indexing from {settings.DOCS_DIR} and root...")
        
        # 1. Clear existing documents to avoid stale/irrelevant data
        try:
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
                print(f"RAG: Cleared {len(all_ids)} existing chunks.")
        except Exception as e:
            print(f"RAG: Warning - could not clear collection: {e}")
        
        # 2. Identify files to index
        files = []
        
        # A. Recursive scan in DOCS_DIR
        if os.path.exists(settings.DOCS_DIR):
            for root, dirs, filenames in os.walk(settings.DOCS_DIR):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__']]
                for filename in filenames:
                    if filename.endswith((".md", ".txt", ".pdf")):
                        files.append(os.path.join(root, filename))
        
        # B. Specific root files for high-level project context
        root_files = ["README.md", "requirements.txt"]
        for f in root_files:
            if os.path.exists(f):
                files.append(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for file_path in files:
            try:
                content = ""
                if file_path.endswith(".pdf"):
                    content = self._read_pdf(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                
                if not content or not content.strip():
                    continue

                # Use relative path as source name
                rel_path = os.path.relpath(file_path, ".")
                
                chunks = self._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": rel_path, "chunk": idx})
                    ids.append(f"{rel_path}_{idx}")
            except Exception as e:
                print(f"RAG: Error processing {file_path}: {e}")

        if documents:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"RAG: Successfully indexed {len(documents)} chunks from {len(files)} files.")
        else:
            print("RAG: No documents found to index.")

    def _rerank(self, query: str, documents: List[str], metadatas: List[Dict], top_n: int = 3) -> List[Dict]:
        """Rerank retrieved documents using a Cross-Encoder."""
        if not self.reranker or not documents:
            return [{"doc": d, "metadata": m} for d, m in zip(documents, metadatas)][:top_n]

        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)

        scored_results = sorted(
            zip(scores, documents, metadatas),
            key=lambda x: x[0],
            reverse=True
        )

        return [{"doc": d, "metadata": m, "score": float(s)} for s, d, m in scored_results[:top_n]]

    async def search(self, query: str, n_results: int = 3) -> str:
        """Query the vector database with HyDE, rerank, and return results."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # 1. HyDE: Generate a hypothetical answer to improve embedding relevance
        try:
            hyde_response = await client.chat.completions.create(
                model=settings.AGENT_MODEL,
                messages=[
                    {"role": "system", "content": "Write a brief, hypothetical technical answer to the user's question. Focus on keywords and technical facts."},
                    {"role": "user", "content": query}
                ],
                max_tokens=200
            )
            hypothetical_answer = hyde_response.choices[0].message.content
            search_query = f"{query}\n{hypothetical_answer}"
            print(f"RAG: HyDE generated hypothetical answer (first 50 chars): {hypothetical_answer[:50]}...")
        except Exception as e:
            print(f"RAG: Warning - HyDE failed: {e}")
            search_query = query

        # 2. Broad retrieval using the expanded query
        fetch_k = max(n_results * 3, 10)
        results = self.collection.query(
            query_texts=[search_query],
            n_results=fetch_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found."

        # 3. Rerank the top results using the ORIGINAL query for precision
        reranked = self._rerank(
            query=query,
            documents=results['documents'][0],
            metadatas=results['metadatas'][0],
            top_n=n_results
        )
        
        # 4. Format output
        context_parts = []
        for res in reranked:
            source = res['metadata']['source']
            context_parts.append(f"--- FROM {source} ---\n{res['doc']}")
            
        return "\n\n".join(context_parts)

rag_service = RAGService()
