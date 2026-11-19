import chromadb
from sentence_transformers import SentenceTransformer

from chunker import MODEL_NAME, chunk_documents

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "policy_chunks"

# Step 1: Split the policy documents into chunks
chunks = chunk_documents()

# Stop here if there is nothing to index, so an existing collection is not lost
if not chunks:
    raise SystemExit("No chunks found. Add .md policy files to docs/ and run again.")

# Step 2: Embed every chunk, including its contextual header
model = SentenceTransformer(MODEL_NAME)  # Same model the chunker measured tokens with
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts).tolist()  # Chroma expects plain Python lists

# Step 3: Delete the old collection and create an empty one
client = chromadb.PersistentClient(path=CHROMA_PATH)  # Saves data to disk
client.get_or_create_collection(name=COLLECTION_NAME)  # Makes sure the delete works
client.delete_collection(name=COLLECTION_NAME)
# embedding_function=None: Chroma must not embed anything itself,
# because we always pass our own embeddings
collection = client.create_collection(name=COLLECTION_NAME, embedding_function=None)

# Step 4: Store the chunks with their embeddings and metadata
# (doc_name, section, section_code, chunk_id)
collection.add(
    ids=[chunk["metadata"]["chunk_id"] for chunk in chunks],
    documents=texts,
    embeddings=embeddings,
    metadatas=[chunk["metadata"] for chunk in chunks],
)

# Step 5: Print the total and 2 samples, one from the start and one from the middle
print(f"Total chunks stored: {collection.count()}")

for chunk in [chunks[0], chunks[len(chunks) // 2]]:
    print(f"\nMetadata: {chunk['metadata']}")
    print(chunk["text"])
