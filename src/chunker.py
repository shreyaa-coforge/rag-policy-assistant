# Chunking rationale:
# - Chunk size (200 tokens, measured with the embedding model's own tokenizer):
#   keeps chunks under all-MiniLM-L6-v2's 256-token limit, with room for the
#   contextual header. The header counts toward the 200 tokens.
# - Sections: chunks follow the documents' own ## and ### structure.
# - Tables and lists: stay intact. Only a table that is over the limit by
#   itself is split, by rows, with its header row repeated in every piece.
# - Overlap (one sentence, list item, or table row): applies only when a
#   section is split, so context is preserved exactly where the text is cut.
#   An overlapping table row carries its table's header row with it.

import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 200
SKIP_FILES = {"DATA_QUALITY_ISSUE.md"}

# Only the tokenizer is needed here, to measure chunk size
tokenizer = SentenceTransformer(MODEL_NAME).tokenizer


def count_tokens(text):
    """Count tokens the way the embedding model will see them."""
    return len(tokenizer.encode(text, verbose=False))


def fits(header, text):
    """Check whether the header plus this text stays within the limit."""
    return count_tokens(f"{header}\n{text}") <= MAX_TOKENS


def is_table(block):
    return block.startswith("|")


def is_list(block):
    # Matches "- item", "* item", "+ item", and "1. item"
    return re.match(r"([-*+]|\d+\.) ", block) is not None


def parse_sections(text):
    """Return the document title and a list of (parent, heading, body)."""
    doc_title = ""
    parent = ""
    sections = []

    for line in text.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            # A ## heading has no parent, and becomes the parent of its ### headings
            parent = line[3:].strip()
            sections.append({"parent": "", "heading": parent, "lines": []})
        elif line.startswith("### "):
            sections.append({"parent": parent, "heading": line[4:].strip(), "lines": []})
        elif sections:
            # Lines before the first ## heading are skipped (owner, version, date)
            sections[-1]["lines"].append(line)

    results = []
    for section in sections:
        body = "\n".join(section["lines"]).strip()
        # A heading with no body of its own is not a chunk
        if body:
            results.append((section["parent"], section["heading"], body))
    return doc_title, results


def split_blocks(body):
    """Split a section body into paragraphs, tables, and lists."""
    return [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]


def split_sentences(paragraph):
    """Split a paragraph at sentence endings."""
    # Split after . ! or ? when a new sentence starts next.
    # The (?<!...) part avoids splitting after abbreviations like "U.S." or "e.g."
    pattern = r"(?<!\.[A-Za-z]\.)(?<=[.!?])\s+(?=[A-Z0-9*\"(])"
    return re.split(pattern, paragraph)


def last_unit(block):
    """Return the last table row, list item, or sentence of a block."""
    lines = block.splitlines()
    if is_table(block):
        if len(lines) > 2:
            # Carry the header and separator rows so the row is never orphaned
            return "\n".join(lines[:2] + [lines[-1]])
        # A single row (inside split_table): the header is added as a prefix there
        return lines[-1]
    if is_list(block):
        return lines[-1]
    return split_sentences(block)[-1]


def pack_units(units, header, separator, prefix=""):
    """Group units into pieces under the limit, overlapping one unit between pieces."""
    pieces = []
    current = []

    for unit in units:
        # Keep adding units while they fit. A piece always holds at least one unit.
        if not current or fits(header, prefix + separator.join(current + [unit])):
            current.append(unit)
            continue

        # This unit does not fit, so finish the piece and start the next one
        pieces.append(prefix + separator.join(current))
        overlap = last_unit(current[-1])
        with_overlap = [overlap, unit]

        # Skip the overlap if it breaks the limit or the unit already contains it
        if overlap not in unit and fits(header, prefix + separator.join(with_overlap)):
            current = with_overlap
        else:
            current = [unit]

    pieces.append(prefix + separator.join(current))
    return pieces


def split_table(table, header):
    """Split an oversized table by rows, repeating its header row in every piece."""
    lines = table.splitlines()
    # The first two lines are the column names and the |---|---| separator
    table_header = "\n".join(lines[:2]) + "\n"
    return pack_units(lines[2:], header, "\n", prefix=table_header)


def break_block(block, header):
    """Return a block whole if it fits, otherwise break it into smaller units."""
    if fits(header, block):
        return [block]
    if is_table(block):
        return split_table(block, header)
    if is_list(block):
        return block.splitlines()
    return split_sentences(block)


def split_section(header, body):
    """Return a section body as one piece, or several if it is over the limit."""
    if fits(header, body):
        return [body]

    units = []
    for block in split_blocks(body):
        units.extend(break_block(block, header))
    return pack_units(units, header, "\n\n")


def get_section_code(heading):
    """Pull the code from a heading, e.g. "EXP-4.2 Meals" -> "EXP-4.2"."""
    match = re.match(r"[A-Z]+-\d+(\.\d+)?", heading)
    return match.group() if match else ""


def chunk_documents(docs_dir="docs"):
    """Read every policy file in docs_dir and return a list of chunks."""
    chunks = []

    for path in sorted(Path(docs_dir).glob("*.md")):
        if path.name in SKIP_FILES:
            continue

        doc_title, sections = parse_sections(path.read_text(encoding="utf-8"))
        chunk_number = 0  # Numbers the chunks within this document

        for parent, heading, body in sections:
            # Build "<Document title> > <parent heading> > <section heading>"
            # A ## section has no parent, so that part is left out
            parts = [doc_title, parent, heading]
            header = " > ".join(part for part in parts if part)

            for piece in split_section(header, body):
                chunks.append(
                    {
                        "text": f"{header}\n{piece}",
                        "metadata": {
                            "doc_name": path.name,
                            "section": heading,
                            "section_code": get_section_code(heading),
                            "chunk_id": f"{path.stem}-{chunk_number}",
                        },
                    }
                )
                chunk_number += 1

    return chunks


if __name__ == "__main__":
    chunks = chunk_documents()

    # Group the chunks by document and by section
    chunks_per_doc = {}
    tokens_per_section = {}
    for chunk in chunks:
        doc_name = chunk["metadata"]["doc_name"]
        section = chunk["metadata"]["section"]
        chunks_per_doc[doc_name] = chunks_per_doc.get(doc_name, 0) + 1
        tokens_per_section.setdefault((doc_name, section), []).append(count_tokens(chunk["text"]))

    print("Total chunks per document:")
    for doc_name, total in chunks_per_doc.items():
        print(f"  {doc_name}: {total}")

    print("\nSections that were split (tokens per piece):")
    for (doc_name, section), token_counts in tokens_per_section.items():
        if len(token_counts) > 1:
            print(f"  {doc_name} | {section}: {token_counts}")

    print("\nSample chunk containing a table:")
    # A line starting with "|" means the chunk holds a table
    sample = next(chunk for chunk in chunks if "\n|" in chunk["text"])
    print(f"Metadata: {sample['metadata']}")
    print(sample["text"])
