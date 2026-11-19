import pytest

CODE_QUESTION = "What does section PTO-3.1 say?"


def is_pto_3_1(result):
    return result["doc_name"] == "pto_policy.md" and result["section_code"] == "PTO-3.1"


def test_keyword_search_ranks_exact_section_code_first(retrieve_module):
    results = retrieve_module.keyword_search(CODE_QUESTION)

    assert results
    assert is_pto_3_1(results[0])


def test_keyword_search_ignores_stopwords(retrieve_module):
    results = retrieve_module.keyword_search("what is the a an of to and or in")

    assert [result for result in results if result["score"] > 0] == []


def test_rrf_combine_orders_by_reciprocal_rank_fusion(retrieve_module):
    vector_results = [{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}]
    keyword_results = [{"chunk_id": "C"}, {"chunk_id": "A"}, {"chunk_id": "D"}]

    # Each list adds 1 / (60 + rank) for every chunk it contains (rank starts at 1):
    #   A: 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522  (rank 1 and 2)
    #   C: 1/(60+3) + 1/(60+1) = 0.015873 + 0.016393 = 0.032266  (rank 3 and 1)
    #   B: 1/(60+2)            = 0.016129                       (vector list only)
    #   D: 1/(60+3)            = 0.015873                       (keyword list only)
    # So the combined order is A, C, B, D.
    combined = retrieve_module.rrf_combine(vector_results, keyword_results)

    assert [result["chunk_id"] for result in combined] == ["A", "C", "B", "D"]
    expected_scores = [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    assert [result["rrf_score"] for result in combined] == pytest.approx(expected_scores)


def test_hybrid_search_finds_section_code_that_vector_search_misses(retrieve_module):
    vector_results = retrieve_module.vector_search(CODE_QUESTION, k=5)
    hybrid_results = retrieve_module.hybrid_search(CODE_QUESTION, k=5)

    assert not any(is_pto_3_1(result) for result in vector_results)
    assert any(is_pto_3_1(result) for result in hybrid_results)
