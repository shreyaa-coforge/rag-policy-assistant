import os
import runpy
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def retrieve_module(tmp_path_factory):
    """Build the index once in a temporary folder and return the retrieve module.

    index.py and retrieve.py use the relative path ./chroma_db, so we switch
    into a temporary folder first. This keeps the project's real chroma_db untouched.
    """
    work_dir = tmp_path_factory.mktemp("index")
    shutil.copytree(PROJECT_ROOT / "docs", work_dir / "docs")

    original_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        # Run index.py as a script, the same way "python src/index.py" does
        runpy.run_path(str(PROJECT_ROOT / "src" / "index.py"))
        # retrieve opens ./chroma_db when imported, so import it here
        import retrieve
    finally:
        os.chdir(original_dir)

    return retrieve


@pytest.fixture(scope="session")
def rag_module(retrieve_module):
    # Importing rag must work without OPENAI_API_KEY (CI runs unit tests without it)
    import rag

    return rag
