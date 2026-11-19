import chromadb
from sentence_transformers import SentenceTransformer

# Section 1: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Section 2: Create an in-memory vector store (data is lost when the script ends)
client = chromadb.EphemeralClient()
collection = client.create_collection(name="minimal_loop")

# Section 3: Embed and store the first text
text_1 = "Employees may work remotely up to three days per week."
embedding_1 = model.encode(text_1).tolist()  # Chroma expects a plain Python list
collection.add(ids=["text_1"], documents=[text_1], embeddings=[embedding_1])
print(f"Stored text_1. Items in collection: {collection.count()}")
print(f"Embedding length: {len(embedding_1)}")

# Section 4: Embed and store the second, unrelated text
text_2 = "Office parking permits are issued by the facilities team."
embedding_2 = model.encode(text_2).tolist()
collection.add(ids=["text_2"], documents=[text_2], embeddings=[embedding_2])
print(f"Stored text_2. Items in collection: {collection.count()}")

# Section 5: Embed the question and find the closest stored texts
question = "How many days a week can I work from home?"
question_embedding = model.encode(question).tolist()
results = collection.query(query_embeddings=[question_embedding], n_results=2)

# Chroma returns one list per question, so [0] selects our only question
documents = results["documents"][0]
distances = results["distances"][0]

# Results are already sorted, closest first (a lower distance means more similar)
print(f"\nQuestion: {question}")
for rank, (document, distance) in enumerate(zip(documents, distances, strict=True), start=1):
    print(f"{rank}. distance={distance:.4f} | {document}")
