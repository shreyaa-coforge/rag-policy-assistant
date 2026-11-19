import sys

from retrieve import hybrid_search, vector_search


def label(hit):
    return f"{hit['doc_name']} | {hit['section_code']}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/compare_hybrid.py "your question"')
    question = " ".join(sys.argv[1:])

    print(f"Question: {question}\n")
    print(f"{'#':<3}{'Vector only':<50}{'Hybrid'}")
    for vector_hit, hybrid_hit in zip(
        vector_search(question, k=5), hybrid_search(question, k=5), strict=True
    ):
        print(f"{vector_hit['rank']:<3}{label(vector_hit):<50}{label(hybrid_hit)}")
