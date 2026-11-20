import re
import sys

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunker import MODEL_NAME

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "policy_chunks"  # Must match the name used in index.py

RRF_K = 60  # Standard constant in reciprocal rank fusion: score = 1 / (RRF_K + rank)
CANDIDATES = 20  # How many results each search contributes before fusion
CODE_BONUS = 10  # A matching code like PTO-3.1 outweighs up to 10 matching words
CODE_PATTERN = re.compile(r"\b[A-Z][A-Z&]*-\d+(?:\.\d+)?\b")  # PTO-3.1, FIN-118, T&E-07
STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "many",
    "much", "my", "of", "on", "or", "say", "the", "to", "what", "when", "with",
}  # fmt: skip

# Load the models and connect to the database once, when this file is imported
model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    # Look the collection up on every search: index.py deletes and recreates it,
    # so a handle kept from import time could point at a deleted collection
    return client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def make_hit(text, metadata, rank, **scores):
    return {
        "text": text,
        "doc_name": metadata["doc_name"],
        "section": metadata["section"],
        "section_code": metadata["section_code"],
        "chunk_id": metadata["chunk_id"],
        "rank": rank,
        **scores,
    }


def vector_search(question, k=5):
    """Return the k chunks whose embeddings are closest to the question."""
    # Embed the question with the same model used for the chunks
    question_embedding = model.encode(question).tolist()

    results = get_collection().query(query_embeddings=[question_embedding], n_results=k)

    # Chroma returns one list per question, so [0] selects our only question.
    # Results are already sorted, closest first (a lower distance means more similar).
    rows = zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
    )
    return [
        make_hit(text, metadata, rank, distance=distance)
        for rank, (text, metadata, distance) in enumerate(rows, start=1)
    ]


def keyword_search(question, k=5):
    """Score chunks by substring matches of the question's words and codes."""
    words = {word.strip("?.,!:;\"'()") for word in question.lower().split()} - STOPWORDS - {""}
    codes = CODE_PATTERN.findall(question)

    stored = get_collection().get(include=["documents", "metadatas"])
    scored = []
    for text, metadata in zip(stored["documents"], stored["metadatas"], strict=True):
        lower_text = text.lower()
        score = sum(word in lower_text for word in words)
        score += CODE_BONUS * sum(code in text for code in codes)
        if score > 0:
            scored.append((score, text, metadata))

    scored.sort(key=lambda item: item[0], reverse=True)  # Stable: ties keep stored order
    return [
        make_hit(text, metadata, rank, score=score)
        for rank, (score, text, metadata) in enumerate(scored[:k], start=1)
    ]


def rrf_combine(*ranked_lists):
    """Merge ranked lists: each list adds 1 / (RRF_K + rank) to a chunk's score."""
    scores, results = {}, {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
            results.setdefault(chunk_id, result)

    order = sorted(scores, key=scores.get, reverse=True)
    return [{**results[chunk_id], "rrf_score": scores[chunk_id]} for chunk_id in order]


def hybrid_search(question, k=5):
    """Fuse the top vector and keyword results with RRF and return the top k."""
    fused = rrf_combine(vector_search(question, CANDIDATES), keyword_search(question, CANDIDATES))
    return [{**result, "rank": rank} for rank, result in enumerate(fused[:k], start=1)]


def rerank(question, chunks, top_n=3):
    """Score each (question, chunk) pair with the cross-encoder and keep the best top_n."""
    if not chunks:
        return []
    # The cross-encoder reads question and chunk together; a higher score means more relevant
    scores = reranker.predict([(question, chunk["text"]) for chunk in chunks])
    scored = [
        {**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores, strict=True)
    ]
    scored.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
    return scored[:top_n]


def retrieve(question, candidates=10, top_n=3):
    return rerank(question, hybrid_search(question, k=candidates), top_n)


if __name__ == "__main__":
    # Everything typed after the script name is the question
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/retrieve.py "your question"')
    question = " ".join(sys.argv[1:])

    print(f"Question: {question}\n")
    for hit in vector_search(question):
        preview = hit["text"][:150].replace("\n", " ")  # Keep each result on one line
        score = f"{hit['rank']}. distance={hit['distance']:.4f}"
        print(f"{score} | {hit['doc_name']} | {hit['section_code']}")
        print(f"   {preview}\n")
