from pathlib import Path

import pytest

from chunker import chunk_documents, count_tokens

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
    return chunk_documents(DOCS_DIR)


def write_doc(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def body(chunk):
    # Everything after the first line, which is the contextual header
    return chunk["text"].split("\n", 1)[1]


# --- Size limit ---


def test_every_real_chunk_is_at_most_200_tokens(real_chunks):
    for chunk in real_chunks:
        assert count_tokens(chunk["text"]) <= 200, chunk["metadata"]["chunk_id"]


# --- Splitting between whole blocks ---


def test_long_section_is_split_only_between_whole_blocks(tmp_path):
    paragraphs = [
        f"Paragraph {i} says employees must follow this rule carefully every day." * 4
        for i in range(8)
    ]
    write_doc(
        tmp_path, "test.md", "# Test Policy\n\n## T-3 Long Rule\n\n" + "\n\n".join(paragraphs)
    )

    chunks = chunk_documents(tmp_path)

    assert len(chunks) > 1
    pieces = [block for chunk in chunks for block in body(chunk).split("\n\n")]
    assert pieces == paragraphs  # Every paragraph whole, in order, exactly once (no overlap)


def test_oversized_block_is_kept_whole_with_a_warning(tmp_path, capsys):
    paragraph = " ".join(f"Rule number {i} applies to all staff." for i in range(60))
    write_doc(tmp_path, "test.md", f"# Test Policy\n\n## T-4 Huge Rule\n\n{paragraph}\n")

    chunks = chunk_documents(tmp_path)

    assert len(chunks) == 1
    assert body(chunks[0]) == paragraph
    warning = capsys.readouterr().out
    assert "test.md" in warning
    assert "T-4 Huge Rule" in warning


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


def test_section_code_comes_from_the_heading(tmp_path):
    write_doc(tmp_path, "test.md", SMALL_DOC)
    codes = [chunk["metadata"]["section_code"] for chunk in chunk_documents(tmp_path)]
    assert codes == ["T-1.1", "T-2"]


def test_real_section_code(real_chunks):
    codes = {
        chunk["metadata"]["section_code"]
        for chunk in real_chunks
        if chunk["metadata"]["section"] == "EXP-4.2 Daily Meal Allowance"
    }
    assert codes == {"EXP-4.2"}


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
