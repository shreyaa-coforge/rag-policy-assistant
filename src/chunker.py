# Chunking rationale:
# - 200 tokens (measured with the model's own tokenizer) keeps chunks under
#   all-MiniLM-L6-v2's 256-token limit, with room for the contextual header.
# - Chunks follow the documents' ## and ### section structure.
# - Long sections are split only between whole blocks (paragraphs, tables,
#   lists), so no sentence or table row is ever cut.
# - Every piece repeats the section header.
# - Overlap applies only when a section is split: the next piece starts with
#   the last sentence of a paragraph, or a table's header, separator, and last
#   row, so nothing is orphaned. Corpus analysis of the 11 split points: overlap
#   needs 24-38 tokens (median 33), so OVERLAP_TOKENS = 40. The largest later
#   piece is 134 tokens, and 134 + 40 = 174 stays under the 200-token cap.

import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 200
OVERLAP_TOKENS = 40
SKIP_FILES = {"DATA_QUALITY_ISSUE.md"}

tokenizer = SentenceTransformer(MODEL_NAME).tokenizer


def count_tokens(text):
    return len(tokenizer.encode(text, verbose=False))


def parse_sections(text):
    """Return the document title and a list of (parent, heading, body)."""
    title, parent, sections = "", "", []
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            parent = line[3:].strip()
            sections.append(("", parent, []))
        elif line.startswith("### "):
            sections.append((parent, line[4:].strip(), []))
        elif sections:  # Lines before the first ## (owner, version, date) are skipped
            sections[-1][2].append(line)

    with_bodies = [(p, heading, "\n".join(lines).strip()) for p, heading, lines in sections]
    return title, [section for section in with_bodies if section[2]]  # Drop empty headings


def split_blocks(body):
    return [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]


def overlap_for(block):
    if block.startswith("|"):  # Table: header row, separator row, last row
        lines = block.splitlines()
        return "\n".join([*lines[:2], lines[-1]])
    return re.split(r"(?<=[.!?])\s+", block)[-1]  # Paragraph: last sentence


def pack_blocks(header, blocks):
    """Group whole blocks into pieces that fit under the cap together with the header."""
    pieces, current = [], []
    for block in blocks:
        candidate = "\n\n".join([*current, block])
        if current and count_tokens(f"{header}\n{candidate}") > MAX_TOKENS:
            pieces.append("\n\n".join(current))
            overlap = overlap_for(current[-1])
            small = count_tokens(overlap) <= OVERLAP_TOKENS
            fits = count_tokens(f"{header}\n{overlap}\n\n{block}") <= MAX_TOKENS
            current = [overlap, block] if small and fits else [block]
        else:
            current.append(block)
    pieces.append("\n\n".join(current))
    return pieces


def chunk_documents(docs_dir="docs"):
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        if path.name in SKIP_FILES:
            continue
        title, sections = parse_sections(path.read_text(encoding="utf-8"))
        chunk_number = 0
        for parent, heading, body in sections:
            header = " > ".join(part for part in (title, parent, heading) if part)
            code = re.match(r"[A-Z]+-\d+(\.\d+)?", heading)
            for piece in pack_blocks(header, split_blocks(body)):
                text = f"{header}\n{piece}"
                # Only a single block that is too big on its own can exceed the cap
                if count_tokens(text) > MAX_TOKENS:
                    print(f"Warning: block over {MAX_TOKENS} tokens in {path.name} | {heading}")
                metadata = {
                    "doc_name": path.name,
                    "section": heading,
                    "section_code": code.group() if code else "",
                    "chunk_id": f"{path.stem}-{chunk_number}",
                }
                chunks.append({"text": text, "metadata": metadata})
                chunk_number += 1
    return chunks


if __name__ == "__main__":
    per_doc, per_section = {}, {}
    for chunk in chunk_documents():
        doc_name, section = chunk["metadata"]["doc_name"], chunk["metadata"]["section"]
        per_doc[doc_name] = per_doc.get(doc_name, 0) + 1
        per_section.setdefault((doc_name, section), []).append(count_tokens(chunk["text"]))

    for doc_name, total in per_doc.items():
        print(f"{doc_name}: {total} chunks")
    for (doc_name, section), counts in per_section.items():
        if len(counts) > 1:
            print(f"Split: {doc_name} | {section}: {counts} tokens")
    print(f"Largest chunk: {max(max(counts) for counts in per_section.values())} tokens")
