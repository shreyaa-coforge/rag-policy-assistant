from itertools import pairwise
from pathlib import Path

import pytest

from chunker import chunk_documents, count_tokens, get_section_code, last_unit, split_blocks

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# A small policy with a header block, a heading with no body, and two real sections
SMALL_DOC = """# Test Policy

**Policy Owner:** Testing
**Version:** 9.9
**Effective Date:** January 1, 2030

## T-1 Parent Only

### T-1.1 First Child

Child body text.

## T-2 Standalone

Standalone body text.
"""


@pytest.fixture(scope="module")
def real_chunks():
    """Chunk the real policy documents once for all tests in this file."""
    return chunk_documents(DOCS_DIR)


def write_doc(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def body(chunk):
    """Return the chunk text without its first line (the contextual header)."""
    return chunk["text"].split("\n", 1)[1]


# --- Size limit ---


def test_every_real_chunk_is_at_most_200_tokens(real_chunks):
    for chunk in real_chunks:
        assert count_tokens(chunk["text"]) <= 200, chunk["metadata"]["chunk_id"]


# --- Headings with no body ---


def test_real_heading_without_body_is_not_a_chunk(real_chunks):
    sections = {chunk["metadata"]["section"] for chunk in real_chunks}
    assert "EXP-4 Meals" not in sections
    assert "EXP-4.2 Daily Meal Allowance" in sections  # Its subsection still is


def test_synthetic_heading_without_body_is_not_a_chunk(tmp_path):
    write_doc(tmp_path, "test.md", SMALL_DOC)
    sections = [chunk["metadata"]["section"] for chunk in chunk_documents(tmp_path)]
    assert sections == ["T-1.1 First Child", "T-2 Standalone"]


# --- Header block ---


def test_header_block_never_appears_in_real_chunks(real_chunks):
    for chunk in real_chunks:
        assert "**Version:**" not in chunk["text"]
        assert "**Effective Date:**" not in chunk["text"]


def test_header_block_never_appears_in_synthetic_chunks(tmp_path):
    write_doc(tmp_path, "test.md", SMALL_DOC)
    for chunk in chunk_documents(tmp_path):
        assert "9.9" not in chunk["text"]
        assert "January 1, 2030" not in chunk["text"]
        assert "Policy Owner" not in chunk["text"]


# --- Contextual header ---


def test_synthetic_chunks_start_with_exact_contextual_header(tmp_path):
    write_doc(tmp_path, "test.md", SMALL_DOC)
    first_lines = [chunk["text"].split("\n")[0] for chunk in chunk_documents(tmp_path)]
    assert first_lines == [
        "Test Policy > T-1 Parent Only > T-1.1 First Child",  # ### section: has a parent
        "Test Policy > T-2 Standalone",  # ## section: no parent
    ]


def test_real_chunks_start_with_contextual_header(real_chunks):
    for chunk in real_chunks:
        doc_path = DOCS_DIR / chunk["metadata"]["doc_name"]
        doc_title = doc_path.read_text(encoding="utf-8").splitlines()[0][2:]  # Drop "# "
        first_line = chunk["text"].split("\n")[0]
        assert first_line.startswith(f"{doc_title} > ")
        assert first_line.endswith(f" > {chunk['metadata']['section']}")


# --- Section code ---


def test_get_section_code():
    assert get_section_code("EXP-4.2 Daily Meal Allowance") == "EXP-4.2"


# --- Oversized table ---


def test_oversized_table_is_split_by_rows_with_header_in_every_piece(tmp_path):
    rows = [
        f"| Item {i} with several descriptive words | Value {i} with more detail |"
        for i in range(40)
    ]
    table = "\n".join(["| Name | Value |", "|---|---|"] + rows)
    write_doc(tmp_path, "test.md", f"# Test Policy\n\n## T-3 Big Table\n\n{table}\n")

    chunks = chunk_documents(tmp_path)

    assert len(chunks) > 1
    for chunk in chunks:
        assert body(chunk).startswith("| Name | Value |\n|---|---|\n")
        assert count_tokens(chunk["text"]) <= 200
    # Splitting by rows must not lose any row
    all_text = "\n".join(chunk["text"] for chunk in chunks)
    for row in rows:
        assert row in all_text


# --- Overlap ---


def test_split_section_piece_2_starts_with_last_unit_of_piece_1(tmp_path):
    sentences = [
        f"Sentence number {i} explains one rule for employees in detail." for i in range(40)
    ]
    write_doc(
        tmp_path, "test.md", "# Test Policy\n\n## T-4 Long Rule\n\n" + " ".join(sentences) + "\n"
    )

    chunks = chunk_documents(tmp_path)

    assert len(chunks) > 1
    for piece_1, piece_2 in pairwise(chunks):
        overlap = last_unit(split_blocks(body(piece_1))[-1])
        assert overlap in sentences  # The overlap is one whole sentence
        assert body(piece_2).startswith(overlap)


def test_real_table_overlap_carries_the_table_header(real_chunks):
    # EXP-4.2 piece 1 ends with the meal table, so piece 2 starts with its last row
    pieces = [
        chunk
        for chunk in real_chunks
        if chunk["metadata"]["doc_name"] == "expense_reimbursement_policy.md"
        and chunk["metadata"]["section_code"] == "EXP-4.2"
    ]
    assert len(pieces) == 2
    assert body(pieces[1]).startswith(
        "| Meal | Daily Limit |\n|---|---|\n| **Total per day** | **$75** |"
    )


# --- Skipped files ---


def test_data_quality_issue_file_is_skipped(tmp_path):
    write_doc(tmp_path, "policy.md", SMALL_DOC)
    write_doc(
        tmp_path,
        "DATA_QUALITY_ISSUE.md",
        "# Issue\n\n## DQ-1 Problem\n\nThis must not be chunked.\n",
    )

    doc_names = {chunk["metadata"]["doc_name"] for chunk in chunk_documents(tmp_path)}
    assert doc_names == {"policy.md"}
