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


def report(label, hits, total):
    print(f"  {label:<10} recall@5: {hits / total:.2f} ({hits}/{total})")


def test_recall_at_5(retrieve_module):
    print()
    for name in ("vector_search", "hybrid_search"):
        search = getattr(retrieve_module, name)
        hit_by_case = [(case, found(search, case, 5)) for case in EVAL_SET]
        missed = [case for case, hit in hit_by_case if not hit]

        print(name)
        report("overall", len(EVAL_SET) - len(missed), len(EVAL_SET))
        for kind in TYPES:
            of_type = [hit for case, hit in hit_by_case if case["type"] == kind]
            report(kind, sum(of_type), len(of_type))
        missed_lines = [f"[{case['type']}] {case['question']}" for case in missed]
        print("  Missed:", *missed_lines or ["none"], sep="\n    - ")

        assert recall_at_k(search, k=5) >= 0.9
