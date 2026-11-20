import sys
from itertools import zip_longest

from retrieve import hybrid_search, rerank

if len(sys.argv) < 2:
    raise SystemExit('Usage: python src/compare_rerank.py "your question"')
question = " ".join(sys.argv[1:])

hybrid = hybrid_search(question, k=10)
reranked = rerank(question, hybrid, top_n=3)

print(f"Question: {question}\n")
print(f"{'#':<3}{'Hybrid top 10':<45}Reranked top 3 (rerank_score)")
# Hybrid has 10 rows and reranked has 3, so pad the shorter column with blanks
for rank, (hit, best) in enumerate(zip_longest(hybrid, reranked), start=1):
    left = f"{hit['doc_name']} | {hit['section_code']}"
    right = (
        f"{best['doc_name']} | {best['section_code']} ({best['rerank_score']:.2f})" if best else ""
    )
    print(f"{rank:<3}{left:<45}{right}")
