from itertools import pairwise
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


# --- Splitting and overlap ---


def sentence(i, j):
    return f"Paragraph {i} sentence {j} says employees must follow this rule."


PARAGRAPHS = [" ".join(sentence(i, j) for j in range(3)) for i in range(8)]
LONG_DOC = "# Test Policy\n\n## T-3 Long Rule\n\n" + "\n\n".join(PARAGRAPHS) + "\n"


def test_long_section_is_split_only_between_whole_blocks(tmp_path):
    write_doc(tmp_path, "test.md", LONG_DOC)

    chunks = chunk_documents(tmp_path)

    assert len(chunks) > 1
    # Drop the overlap (first block) of every later piece; what is left is the whole section
    blocks = body(chunks[0]).split("\n\n")
    for chunk in chunks[1:]:
        blocks += body(chunk).split("\n\n")[1:]
    assert blocks == PARAGRAPHS


def test_split_after_paragraph_repeats_its_last_sentence(tmp_path):
    write_doc(tmp_path, "test.md", LONG_DOC)

    chunks = chunk_documents(tmp_path)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        i = PARAGRAPHS.index(body(previous).split("\n\n")[-1])
        assert body(current).startswith(sentence(i, 2) + "\n\n")


def test_split_after_table_repeats_header_separator_and_last_row(tmp_path):
    rows = [f"| Item {i} | Value {i} |" for i in range(12)]  # 224 tokens in total, so it splits
    table = "\n".join(["| Name | Value |", "|---|---|", *rows])
    closing = " ".join(f"Closing sentence {j} adds more detail about this rule." for j in range(11))
    intro = "This intro explains the table."
    write_doc(
        tmp_path,
        "test.md",
        f"# Test Policy\n\n## T-5 Table Rule\n\n{intro}\n\n{table}\n\n{closing}\n",
    )

    chunks = chunk_documents(tmp_path)

    assert len(chunks) == 2
    assert body(chunks[0]).endswith(table)
    assert body(chunks[1]) == f"| Name | Value |\n|---|---|\n| Item 11 | Value 11 |\n\n{closing}"


def test_section_that_fits_has_no_overlap(tmp_path):
    write_doc(tmp_path, "test.md", SMALL_DOC)
    texts = [chunk["text"] for chunk in chunk_documents(tmp_path)]
    assert texts == [
        "Test Policy > T-1 Parent Only > T-1.1 First Child\nChild body text.",
        "Test Policy > T-2 Standalone\nStandalone body text.",
    ]


def test_no_chunk_body_is_empty_or_only_a_heading(real_chunks):
    for chunk in real_chunks:
        lines = chunk["text"].split("\n", 1)
        assert len(lines) == 2, chunk["metadata"]["chunk_id"]
        assert lines[1].strip(), chunk["metadata"]["chunk_id"]
        assert not lines[1].lstrip().startswith("#"), chunk["metadata"]["chunk_id"]


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
