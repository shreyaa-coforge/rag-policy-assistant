import json
from pathlib import Path

EVAL_SET = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
TYPES = sorted({case["type"] for case in EVAL_SET})


def found(search_fn, case, k):
    # Match on doc_name + section_code, never chunk_id (chunk ids change when chunking changes)
    return any(
        result["doc_name"] == case["expected_doc"]
        and result["section_code"] == case["expected_section_code"]
        for result in search_fn(case["question"], k)
    )


def recall_at_k(search_fn, k=5):
    return sum(found(search_fn, case, k) for case in EVAL_SET) / len(EVAL_SET)


def report(name, search, k):
    """Print recall@k overall and per type, then the missed questions."""
    hit_by_case = [(case, found(search, case, k)) for case in EVAL_SET]
    missed = [case for case, hit in hit_by_case if not hit]

    print(f"{name} recall@{k}")
    for kind in ["overall", *TYPES]:
        hits = [hit for case, hit in hit_by_case if kind in ("overall", case["type"])]
        print(f"  {kind:<10} {sum(hits) / len(hits):.2f} ({sum(hits)}/{len(hits)})")
    missed_lines = [f"[{case['type']}] {case['question']}" for case in missed]
    print("  Missed:", *missed_lines or ["none"], sep="\n    - ")


def test_recall_at_5(retrieve_module):
    print()
    for name in ("vector_search", "hybrid_search"):
        search = getattr(retrieve_module, name)
        report(name, search, 5)
        assert recall_at_k(search, k=5) >= 0.9


def test_rerank_recall_at_1_and_3(retrieve_module):
    # retrieve() always reranks the hybrid top 10; k only sets how many it keeps
    def reranked(question, k):
        return retrieve_module.retrieve(question, top_n=k)

    print()
    for k in (1, 3):
        report("hybrid_search", retrieve_module.hybrid_search, k)
        report("retrieve (rerank)", reranked, k)
