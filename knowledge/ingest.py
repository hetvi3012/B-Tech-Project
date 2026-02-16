import os
import chromadb
from chromadb.utils import embedding_functions

class CodeIndexer:
    def __init__(self, repo_path="./", db_path="./chroma_db"):
        self.repo_path = repo_path
        self.client = chromadb.PersistentClient(path=db_path)
        # Using a local embedding model that doesn't require an API key
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="codebase", 
            embedding_function=self.emb_fn
        )

    def simple_code_splitter(self, code, chunk_size=1000):
        """Basic splitter that tries to break at newlines to avoid cutting lines."""
        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            if current_size > chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        return chunks

    def run(self):
        ids = []
        documents = []
        metadatas = []
        
        count = 0
        for root, _, files in os.walk(self.repo_path):
            if any(x in root for x in ["venv", ".git", "__pycache__", "chroma_db"]):
                continue
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.normpath(os.path.join(root, file))
                    with open(full_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        chunks = self.simple_code_splitter(code)
                        
                        for i, chunk in enumerate(chunks):
                            ids.append(f"{full_path}_{i}")
                            documents.append(chunk)
                            metadatas.append({"path": full_path, "chunk_id": i})
                            count += 1

        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ Success! Indexed {count} code snippets from {self.repo_path}")

if __name__ == "__main__":
    indexer = CodeIndexer()
    indexer.run()