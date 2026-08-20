import json
import os
import chromadb
from chromadb.utils import embedding_functions

# Load chunks
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
kb_data_dir = os.path.join(ROOT_DIR, "data", "knowledge_base")
chunks_path = os.path.join(kb_data_dir, "chunks.json")

with open(chunks_path, "r") as f:
    chunks = json.load(f)

# Initialize ChromaDB client (local persistent)
db_dir = os.path.join(kb_data_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_dir)

# Initialize embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collection
collection = client.get_or_create_collection(
    name="policy_knowledge_base",
    embedding_function=sentence_transformer_ef
)

# Prepare data for insertion
ids = []
documents = []
metadatas = []

for chunk in chunks:
    ids.append(chunk["chunk_id"])
    documents.append(chunk["text"])
    metadatas.append({"doc_id": chunk["doc_id"]})

# Add to collection
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"Successfully embedded and indexed {len(chunks)} chunks into ChromaDB at {db_dir}")
