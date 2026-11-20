import os
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def retrieve_module(tmp_path_factory):
    """Build a private index once per test session, then return the retrieve module.

    CHROMA_PATH sends index.py and retrieve.py to a temporary folder, so tests never
    share or rebuild the project's ./chroma_db. The path must be absolute so it
    still points at the same place whatever the current folder is.
    """
    os.environ["CHROMA_PATH"] = str(tmp_path_factory.mktemp("chroma_db").resolve())
    # Run index.py as a script, the same way "python src/index.py" does
    runpy.run_path(str(PROJECT_ROOT / "src" / "index.py"))
    # Import after setting CHROMA_PATH: retrieve reads it once, when imported
    import retrieve

    return retrieve


@pytest.fixture(scope="session")
def rag_module(retrieve_module):
    # Importing rag must work without OPENAI_API_KEY (CI runs unit tests without it)
    import rag

    return rag
