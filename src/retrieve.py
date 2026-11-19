import sys

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import MODEL_NAME

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "policy_chunks"  # Must match the name used in index.py

# Load the model and open the collection once, when this file is imported
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def vector_search(question, k=5):
    """Return the k chunks whose embeddings are closest to the question."""
    # Embed the question with the same model used for the chunks
    question_embedding = model.encode(question).tolist()

    results = collection.query(query_embeddings=[question_embedding], n_results=k)

    # Chroma returns one list per question, so [0] selects our only question
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Results are already sorted, closest first (a lower distance means more similar)
    hits = []
    for rank, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        hits.append({
            "text": text,
            "doc_name": metadata["doc_name"],
            "section": metadata["section"],
            "section_code": metadata["section_code"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance,
            "rank": rank,
        })
    return hits


if __name__ == "__main__":
    # Everything typed after the script name is the question
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/retrieve.py "your question"')
    question = " ".join(sys.argv[1:])

    print(f"Question: {question}\n")
    for hit in vector_search(question):
        preview = hit["text"][:150].replace("\n", " ")  # Keep each result on one line
        print(f"{hit['rank']}. distance={hit['distance']:.4f} | {hit['doc_name']} | {hit['section_code']}")
        print(f"   {preview}\n")
