import faiss
import pickle
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Load saved files
# --------------------------------

print("Loading model and FAISS index...")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("college_faiss.index")

with open("knowledge_base.pkl", "rb") as f:
    df = pickle.load(f)

print("Loaded successfully!")
print("Total records:", len(df))


# --------------------------------
# 2. Function for semantic search
# --------------------------------

def search(query, top_k=3):

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        print(f"\nRank {rank}")
        print("Similarity Score:", round(float(score), 4))
        print("Category:", df.iloc[idx]["category"])
        print("Question:", df.iloc[idx]["question"])
        print("Answer:", df.iloc[idx]["answer"])


# --------------------------------
# 3. Test queries
# --------------------------------

search("How can I apply to the college?")

search("How much do I need to pay for tuition?")

search("When are the college exams conducted?")

search("Does the college have a library?")