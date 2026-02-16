import os
import json
from typing import Dict, Any
from tools.base import Tool, ToolResult, ToolInvocation
import chromadb

class ScoutTool(Tool):
    name: str = "codebase_scout"
    description: str = "Semantic search assistant. Searches the vector database for code snippets relevant to your query."
    
    _parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural language question about the codebase"
            }
        },
        "required": ["query"]
    }

    @property
    def schema(self) -> Dict[str, Any]:
        return self._parameters

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        # 1. Robust Argument Extraction
        args = {}
        if hasattr(invocation, 'arguments'):
            args = invocation.arguments
        elif hasattr(invocation, 'parameters'):
            args = invocation.parameters
        elif isinstance(invocation, dict):
            args = invocation.get('arguments', {})
        
        # Handle case where args is a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                pass

        # 2. Flexible Key Search (LLMs sometimes use 'question' or 'input' instead of 'query')
        query = args.get("query") or args.get("question") or args.get("input") or args.get("content")
        
        if not query:
            # DEBUG: Tell the user exactly what keys we received
            return ToolResult(
                success=False, 
                output=f"Error: No query found. Received keys: {list(args.keys())}",
                error="Missing query parameter"
            )

        try:
            # 3. Connect to the local Vector DB
            db_path = os.path.join(os.getcwd(), ".ai_agent_rag_db")
            
            if not os.path.exists(db_path):
                return ToolResult(
                    success=False, 
                    output=f"RAG Database not found at {db_path}. Please run 'python knowledge/ingest.py' to parse your code first."
                )

            client = chromadb.PersistentClient(path=db_path)
            
            # 4. Get the collection
            collections = client.list_collections()
            if not collections:
                return ToolResult(success=False, output="Database is empty. Run ingestion script.")
            
            # Try to find 'codebase' collection, otherwise take the first one
            collection = next((c for c in collections if c.name == "codebase"), collections[0])
            
            # 5. Perform Semantic Search
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if not results['ids'][0]:
                return ToolResult(success=True, output="No relevant code snippets found in the database.")

            # 6. Format the results
            output_text = f"Found relevant snippets in collection '{collection.name}':\n\n"
            
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                file_path = meta.get('file_path', 'unknown_file')
                output_text += f"--- Result {i+1} ({file_path}) ---\n{doc}\n\n"

            return ToolResult(success=True, output=output_text)

        except Exception as e:
            return ToolResult(success=False, output=f"Database Error: {str(e)}")
