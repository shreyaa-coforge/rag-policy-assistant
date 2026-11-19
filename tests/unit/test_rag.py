from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FAKE_ANSWER = "Rule one applies [a.md | A-1 Rule]."


def make_hit(doc_name, section, chunk_id, text, rank):
    return {
        "text": text,
        "doc_name": doc_name,
        "section": section,
        "section_code": section.split(" ")[0],
        "chunk_id": chunk_id,
        "distance": 0.5 + rank,
        "rank": rank,
    }


# Two chunks share the same doc and section, so sources must be de-duplicated
FAKE_HITS = [
    make_hit("a.md", "A-1 Rule", "a-0", "Policy A > A-1 Rule\nRule one text.", 1),
    make_hit("b.md", "B-2 Other", "b-0", "Policy B > B-2 Other\nOther rule text.", 2),
    make_hit("a.md", "A-1 Rule", "a-1", "Policy A > A-1 Rule\nRule one continued.", 3),
]


@pytest.fixture
def mock_client(rag_module, monkeypatch):
    """Replace retrieval with fixed hits and the OpenAI client with a mock."""
    monkeypatch.setattr(rag_module, "vector_search", lambda question, k: FAKE_HITS)

    client = MagicMock()
    # Shape the mock like a real response: response.choices[0].message.content
    message = SimpleNamespace(content=FAKE_ANSWER)
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    monkeypatch.setattr(rag_module, "client", client)
    return client


def sent_prompt(client):
    """Join every message the mock was called with into one string."""
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    return "\n".join(message["content"] for message in messages)


def test_prompt_contains_labeled_chunks_and_context_only_instruction(rag_module, mock_client):
    rag_module.answer("What is rule one?")

    prompt = sent_prompt(mock_client)
    for hit in FAKE_HITS:
        assert f"[{hit['doc_name']} | {hit['section']}]\n{hit['text']}" in prompt
    assert "Answer ONLY from the context provided" in prompt
    assert "I don't know based on the provided policies" in prompt
    assert "What is rule one?" in prompt


def test_model_is_called_once_with_configured_model_at_temperature_0(rag_module, mock_client):
    rag_module.answer("What is rule one?")

    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == rag_module.OPENAI_MODEL
    assert kwargs["temperature"] == 0


def test_answer_returns_mocked_text_and_deduplicated_sources(rag_module, mock_client):
    result = rag_module.answer("What is rule one?")

    assert result == {
        "answer": FAKE_ANSWER,
        "sources": [
            {"doc_name": "a.md", "section": "A-1 Rule"},
            {"doc_name": "b.md", "section": "B-2 Other"},
        ],
    }
