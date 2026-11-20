import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def retrieve_module():
    """Build the index once per test session, then return the retrieve module.

    Run pytest from the project root: index.py and retrieve.py both use ./chroma_db.
    """
    # Run index.py as a script, the same way "python src/index.py" does
    runpy.run_path(str(PROJECT_ROOT / "src" / "index.py"))
    import retrieve

    return retrieve


@pytest.fixture(scope="session")
def rag_module(retrieve_module):
    # Importing rag must work without OPENAI_API_KEY (CI runs unit tests without it)
    import rag

    return rag
