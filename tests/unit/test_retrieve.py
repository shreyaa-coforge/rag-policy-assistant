from pathlib import Path

EXPECTED_KEYS = {"text", "doc_name", "section", "section_code", "chunk_id", "distance", "rank"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_tests_use_a_private_index_outside_the_project(retrieve_module):
    # A shared ./chroma_db can be rebuilt by another process mid-run, so tests use their own
    index_path = Path(retrieve_module.CHROMA_PATH)

    assert index_path.is_absolute()
    assert not index_path.is_relative_to(PROJECT_ROOT)


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
