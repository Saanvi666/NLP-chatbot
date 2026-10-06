import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# -----------------------------
# 1. Load knowledge base
# -----------------------------
df = pd.read_csv("college_knowledge_base_dummy.csv")

print("Knowledge base loaded:", len(df), "records")


# -----------------------------
# 2. Load Sentence Transformer
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# -----------------------------
# 3. Create embeddings
# -----------------------------
print("Creating embeddings...")

embeddings = model.encode(
    df["question"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# Convert to float32 for FAISS
embeddings = embeddings.astype("float32")


# -----------------------------
# 4. Normalize embeddings
# -----------------------------
faiss.normalize_L2(embeddings)


# -----------------------------
# 5. Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# 6. Save FAISS index
# -----------------------------
faiss.write_index(index, "college_faiss.index")


# -----------------------------
# 7. Save knowledge base
# -----------------------------
with open("knowledge_base.pkl", "wb") as f:
    pickle.dump(df, f)


print("\n--------------------------------")
print("INDEX BUILDING COMPLETED!")
print("--------------------------------")

print("Embedding dimension:", dimension)
print("Total vectors:", index.ntotal)
print("FAISS index saved as: college_faiss.index")
print("Knowledge base saved as: knowledge_base.pkl")