EXPECTED_KEYS = {"text", "doc_name", "section", "section_code", "chunk_id", "distance", "rank"}


def test_vector_search_returns_k_results_with_expected_keys(retrieve_module):
    results = retrieve_module.vector_search("How many days a week can I work from home?", k=3)

    assert len(results) == 3
    for result in results:
        assert set(result) == EXPECTED_KEYS
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_vector_search_results_are_sorted_by_distance(retrieve_module):
    results = retrieve_module.vector_search("What is the daily meal allowance?", k=5)

    distances = [result["distance"] for result in results]
    assert distances == sorted(distances)  # Closest (smallest distance) first
