import os
import glob
from typing import Any
from tools.base import Tool, ToolResult

# Try importing RAG dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class ScoutTool(Tool):
    name = "codebase_scout"
    description = "Search the codebase semantically using RAG (Vector DB)."
    
    # Define schema for the agent
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural language search query"
            }
        },
        "required": ["query"]
    }

    # BTP FIX: Accept config argument to satisfy ToolRegistry
    def __init__(self, config=None, **kwargs):
        self.client = None
        self.collection = None
        if RAG_AVAILABLE:
            try:
                # Use a local folder for the database
                db_path = os.path.join(os.getcwd(), ".ai_agent_rag_db")
                self.client = chromadb.PersistentClient(path=db_path)
                
                # Standard embedding model
                self.ef = embedding_functions.DefaultEmbeddingFunction()
                
                self.collection = self.client.get_or_create_collection(
                    name="btp_codebase",
                    embedding_function=self.ef
                )
                # Auto-index on startup if empty
                self._index_codebase()
            except Exception as e:
                print(f"[RAG Warning] Init failed: {e}")

    def _index_codebase(self):
        """Reads project files and stores them in the vector DB."""
        if not self.collection or self.collection.count() > 0:
            return 

        print("[RAG] Indexing codebase... this happens once.")
        documents = []
        ids = []
        metadatas = []
        
        files = glob.glob("**/*.py", recursive=True) + glob.glob("**/*.md", recursive=True)
        
        count = 0
        for f in files:
            if any(x in f for x in [".git", "venv", "__pycache__", ".chroma", ".ai_agent_rag_db"]):
                continue
                
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    if not content.strip(): continue
                    
                    # Split into chunks
                    chunk_size = 1000
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    
                    for i, chunk in enumerate(chunks):
                        documents.append(f"File: {f}\nContent:\n{chunk}")
                        ids.append(f"{f}_{i}")
                        metadatas.append({"path": f})
                        count += 1
            except Exception:
                pass
                
        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end],
                    ids=ids[i:end],
                    metadatas=metadatas[i:end]
                )
        print(f"[RAG] Indexed {count} chunks.")

    async def execute(self, invocation: Any) -> ToolResult:
        # Handle arguments safely
        try:
            if hasattr(invocation, 'arguments'):
                query = invocation.arguments.get("query")
            elif isinstance(invocation, dict):
                query = invocation.get("arguments", {}).get("query")
            else:
                query = getattr(invocation, 'parameters', {}).get("query")
        except Exception:
            return ToolResult(error="Could not parse query arguments.")

        if not query:
            return ToolResult(error="No query provided.")

        if not RAG_AVAILABLE or not self.collection:
            return ToolResult(error="RAG Error: ChromaDB not available.")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            output = f"RAG Search Results for '{query}':\n\n"
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    path = results['metadatas'][0][i]['path']
                    output += f"--- Match {i+1} (File: {path}) ---\n{doc}\n\n"
            else:
                output += "No relevant code found."
                
            return ToolResult(success=output)
            
        except Exception as e:
            return ToolResult(error=f"Search failed: {str(e)}")