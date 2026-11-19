import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import hybrid_search

OPENAI_MODEL = "gpt-4o-mini"
NOT_FOUND_ANSWER = "I don't know based on the provided policies"

# Read OPENAI_API_KEY from the .env file into the environment (if the file exists)
load_dotenv()

# The rules the model must follow for every question
SYSTEM_PROMPT = f"""You answer questions about company policies.
Rules:
- Answer ONLY from the context provided. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND_ANSWER}".
- Cite sources inline after each fact, in the form [doc_name | section],
  copied from the label above the chunk you used."""


def build_prompt(question, hits):
    """Combine the labelled context chunks and the question into one message."""
    context_blocks = []
    for hit in hits:
        # Label each chunk so the model can cite it
        label = f"[{hit['doc_name']} | {hit['section']}]"
        context_blocks.append(f"{label}\n{hit['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def answer(question, k=5):
    """Retrieve relevant chunks, ask the model, and return the answer with sources."""
    # Step 1: Find the most relevant chunks (vector + keyword search, fused)
    hits = hybrid_search(question, k)

    # Step 2: Ask the model, using only those chunks as context.
    # The client is created here, not at import, so importing rag needs no API key.
    client = OpenAI()  # Finds OPENAI_API_KEY in the environment by itself
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,  # Always pick the most likely words, so answers are repeatable
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
    )

    # Step 3: List each retrieved (doc_name, section) once, in rank order
    sources = []
    for hit in hits:
        source = {"doc_name": hit["doc_name"], "section": hit["section"]}
        if source not in sources:
            sources.append(source)

    return {"answer": response.choices[0].message.content, "sources": sources}


if __name__ == "__main__":
    # Everything typed after the script name is the question
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/rag.py "your question"')
    question = " ".join(sys.argv[1:])

    result = answer(question)
    print(f"Question: {question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for source in result["sources"]:
        print(f"- {source['doc_name']} | {source['section']}")
