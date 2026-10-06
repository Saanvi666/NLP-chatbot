from flask import Flask, render_template, request, jsonify, session
from chatbot_engine import get_answer

app = Flask(__name__)

# Secret key for session management
app.secret_key = "college-chatbot-secret-key"


# --------------------------------
# Home page
# --------------------------------

@app.route("/")
def home():

    # Start conversation history
    if "history" not in session:
        session["history"] = []

    return render_template("index.html")


# --------------------------------
# Chat API
# --------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question.",
            "category": "Unknown",
            "confidence": 0
        })

    # Get answer from AI engine
    result = get_answer(question)

    # Get existing conversation history
    history = session.get("history", [])

    # Add current conversation
    history.append({
        "user": question,
        "bot": result["answer"],
        "category": result["category"],
        "confidence": result["confidence"]
    })

    # Save updated history
    session["history"] = history

    return jsonify({
        "answer": result["answer"],
        "category": result["category"],
        "confidence": result["confidence"]
    })


# --------------------------------
# Conversation history API
# --------------------------------

@app.route("/history")
def history():

    return jsonify({
        "history": session.get("history", [])
    })


# --------------------------------
# Clear conversation
# --------------------------------

@app.route("/clear", methods=["POST"])
def clear():

    session["history"] = []

    return jsonify({
        "message": "Conversation cleared successfully."
    })


# --------------------------------
# Run application
# --------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)