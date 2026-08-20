import json
import os
import chromadb
from chromadb.utils import embedding_functions

kb_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_data")
client = chromadb.PersistentClient(path=os.path.join(kb_data_dir, "chroma_db"))
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_collection(name="policy_knowledge_base", embedding_function=ef)

with open(os.path.join(kb_data_dir, "queries_eval.json"), "r") as f:
    eval_data = json.load(f)

print("=" * 70)
print("RETRIEVAL EVALUATION: Precision@3 and Recall@3 (k=3)")
print("=" * 70)

p3_list = []
r3_list = []

for idx, item in enumerate(eval_data, 1):
    qid = item["query_id"]
    query = item["query"]
    rel_doc_ids = set(item["relevant_doc_ids"])
    
    results = col.query(query_texts=[query], n_results=3)
    retrieved_doc_ids = [m["doc_id"] for m in results["metadatas"][0]]
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    
    # Relevant chunks/docs retrieved in top-3
    relevant_retrieved = [doc_id for doc_id in retrieved_doc_ids if doc_id in rel_doc_ids]
    # Unique relevant documents retrieved
    num_unique_rel_retrieved = len(set(relevant_retrieved))
    total_rel_docs = len(rel_doc_ids)
    
    # Precision@3 = (Number of relevant items in top-3) / 3
    # Note: If multiple chunks from the same relevant document are retrieved, how is Precision@3 defined?
    # At chunk level: relevant_chunks / 3
    # At doc level: if any chunk is relevant, each relevant chunk retrieved in top-3 is relevant.
    p3 = len(relevant_retrieved) / 3.0
    # Recall@3 = (Number of relevant documents found in top-3) / (Total relevant documents for query)
    r3 = num_unique_rel_retrieved / float(total_rel_docs)
    
    p3_list.append(p3)
    r3_list.append(r3)
    
    print(f"\nQuery {idx} [{qid}]: \"{query}\"")
    print(f"  Target Relevant Document(s): {list(rel_doc_ids)}")
    print(f"  Top-3 Retrieved Documents:  {retrieved_doc_ids}")
    print(f"  ChromaDB Distances:         {[round(d, 4) for d in distances]}")
    print(f"  Relevant Matches in Top-3:  {relevant_retrieved} ({len(relevant_retrieved)} of 3)")
    print(f"  - Precision@3: {len(relevant_retrieved)} / 3 = {p3:.4f} ({p3*100:.2f}%)")
    print(f"  - Recall@3:    {num_unique_rel_retrieved} / {total_rel_docs} = {r3:.4f} ({r3*100:.2f}%)")

avg_p3 = sum(p3_list) / len(p3_list)
avg_r3 = sum(r3_list) / len(r3_list)

print("\n" + "=" * 70)
print("AGGREGATE SUMMARY ACROSS ALL 5 EVALUATION QUERIES")
print("=" * 70)
p3_str = " + ".join([f"{p:.4f}" for p in p3_list])
r3_str = " + ".join([f"{r:.4f}" for r in r3_list])
print(f"Mean Precision@3 = ({p3_str}) / {len(p3_list)} = {avg_p3:.4f} ({avg_p3*100:.2f}%)")
print(f"Mean Recall@3    = ({r3_str}) / {len(r3_list)} = {avg_r3:.4f} ({avg_r3*100:.2f}%)")
