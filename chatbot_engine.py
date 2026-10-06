import faiss
import pickle
from sentence_transformers import SentenceTransformer


# --------------------------------
# Load model and saved data
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("college_faiss.index")

with open("knowledge_base.pkl", "rb") as f:
    df = pickle.load(f)


# --------------------------------
# Semantic search function
# --------------------------------

def search(query, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({
            "score": float(score),
            "category": df.iloc[idx]["category"],
            "question": df.iloc[idx]["question"],
            "answer": df.iloc[idx]["answer"]
        })

    return results


# --------------------------------
# Generate chatbot response
# --------------------------------

def get_answer(query):

    results = search(query, top_k=3)

    best_result = results[0]

    score = best_result["score"]

    # Confidence threshold
    THRESHOLD = 0.70

    if score < THRESHOLD:

        return {
            "answer": (
                "I'm sorry, I couldn't find reliable information "
                "about that in the college knowledge base."
            ),
            "category": "Unknown",
            "confidence": score,
            "results": results
        }

    return {
        "answer": best_result["answer"],
        "category": best_result["category"],
        "confidence": score,
        "results": results
    }


# --------------------------------
# Test chatbot
# --------------------------------

if __name__ == "__main__":

    test_queries = [
        "How can I apply to the college?",
        "How much is the tuition fee?",
        "When are semester exams?",
        "Does the college have a library?",
        "What is the hostel fee?"
    ]

    for query in test_queries:

        result = get_answer(query)

        print("\n" + "=" * 60)
        print("USER:", query)
        print("CATEGORY:", result["category"])
        print("CONFIDENCE:", round(result["confidence"], 4))
        print("BOT:", result["answer"])