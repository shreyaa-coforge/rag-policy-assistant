QUESTION = "How many PTO days do new employees get?"

# Small fake chunks with extra keys, to check rerank keeps every original key
FAKE_CHUNKS = [
    {
        "chunk_id": "a",
        "doc_name": "a.md",
        "text": "The office closes at 6 PM on Fridays.",
        "rank": 1,
    },
    {
        "chunk_id": "b",
        "doc_name": "b.md",
        "text": "New employees get 15 PTO days per year.",
        "rank": 2,
    },
    {
        "chunk_id": "c",
        "doc_name": "c.md",
        "text": "Meals are reimbursed up to $75 a day.",
        "rank": 3,
    },
    {"chunk_id": "d", "doc_name": "d.md", "text": "PTO requests need manager approval.", "rank": 4},
]


def test_rerank_returns_at_most_top_n(retrieve_module):
    assert len(retrieve_module.rerank(QUESTION, FAKE_CHUNKS, top_n=3)) == 3
    assert len(retrieve_module.rerank(QUESTION, FAKE_CHUNKS[:2], top_n=3)) == 2


def test_rerank_keeps_keys_adds_score_and_sorts_descending(retrieve_module):
    results = retrieve_module.rerank(QUESTION, FAKE_CHUNKS, top_n=3)

    originals = {chunk["chunk_id"]: chunk for chunk in FAKE_CHUNKS}
    for result in results:
        assert {**originals[result["chunk_id"]], "rerank_score": result["rerank_score"]} == result
    scores = [result["rerank_score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_list_returns_empty_list(retrieve_module):
    assert retrieve_module.rerank(QUESTION, []) == []


def test_table_to_sentences_rewrites_rows_and_keeps_other_lines(retrieve_module):
    text = (
        "PTO Policy > Accrual Rates\n"
        "PTO accrues each pay period.\n"
        "\n"
        "| Years of Service | Annual PTO |\n"
        "|---|---|\n"
        "| 0 to 2 years | 15 days |\n"
        "| 3 to 5 years | 20 days |\n"
        "\n"
        "Years are counted from the hire date."
    )
    expected = (
        "PTO Policy > Accrual Rates\n"
        "PTO accrues each pay period.\n"
        "\n"
        "Years of Service: 0 to 2 years; Annual PTO: 15 days.\n"
        "Years of Service: 3 to 5 years; Annual PTO: 20 days.\n"
        "\n"
        "Years are counted from the hire date."
    )

    assert retrieve_module.table_to_sentences(text) == expected



