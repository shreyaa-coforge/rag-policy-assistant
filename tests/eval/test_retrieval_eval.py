import json
from pathlib import Path

EVAL_SET = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))


def found(search_fn, case, k):
    # Match on doc_name + section_code, never chunk_id (chunk ids change when chunking changes)
    return any(
        result["doc_name"] == case["expected_doc"]
        and result["section_code"] == case["expected_section_code"]
        for result in search_fn(case["question"], k)
    )


def recall_at_k(search_fn, k=5):
    return sum(found(search_fn, case, k) for case in EVAL_SET) / len(EVAL_SET)


def test_vector_search_recall_at_5(retrieve_module):
    search = retrieve_module.vector_search
    recall = recall_at_k(search, k=5)
    missed = [case["question"] for case in EVAL_SET if not found(search, case, 5)]

    print(f"\nvector_search recall@5: {recall:.2f} ({len(EVAL_SET) - len(missed)}/{len(EVAL_SET)})")
    print("Missed:", *missed or ["none"], sep="\n  - ")
    assert recall >= 0.6
