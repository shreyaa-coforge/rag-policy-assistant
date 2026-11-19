# RAG Policy Assistant

This is a training lab: build a RAG system over the policy docs in `docs/`.

## Stack (fixed)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- Vector store: chromadb
- Reranking: sentence-transformers CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Answer generation: OpenAI API (`gpt-4o-mini`, temperature 0)
- Tests: pytest
- CI: GitHub Actions

## Scope rules
- Do NOT add features or libraries beyond the stack above.
- No web UI.
- No agentic retrieval.
- No LangChain or LlamaIndex.
- No BM25 libraries.
- No logging frameworks.
- Keyword search must be a simple substring match over chunk text.

## Code style
- Keep code simple and readable, with short comments.
- The author must be able to understand and explain every line.
- Keep code simple and sized to the problem. No speculative features.
- Comments only where the reason is not obvious from the code.
  No docstrings on small, self-explanatory functions.
- Run `ruff format .` and `ruff check .` before every commit.

## Workflow
- Work one step at a time. Do not build ahead.

- Follow TDD. For new behavior: write the test first, run it and show it
  fails, then implement the minimum code to pass, then rerun. Unit tests
  must not call the OpenAI API; mock the client instead.
