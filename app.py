"""
DocMind RAG API — Flask backend
Endpoints:
  POST /api/ingest      — upload text or plain document content
  POST /api/query       — ask a question, get grounded answer
  GET  /api/stats       — vector store stats
  POST /api/reset       — clear history
  DELETE /api/documents — clear all documents
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import RAGPipeline

app    = Flask(__name__)
CORS(app)

# One global pipeline instance (for demo — production would be per-session)
pipeline = RAGPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Pre-load a sample document so the demo works immediately
SAMPLE_DOC = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
natural intelligence displayed by animals including humans. AI research has been defined
as the field of study of intelligent agents, which refers to any system that perceives its
environment and takes actions that maximize its chance of achieving its goals.

Machine learning is a subfield of AI focused on algorithms that learn from data. Supervised
learning requires labeled examples; the model learns a mapping from inputs to outputs.
Unsupervised learning finds hidden structure in unlabeled data. Reinforcement learning
trains agents via reward signals from an environment.

Natural language processing (NLP) is the branch of AI dealing with understanding and
generating human language. Transformers are the dominant architecture, using self-attention
to model relationships between all tokens simultaneously. BERT learns bidirectional context;
GPT predicts the next token autoregressively.

Computer vision applies deep learning to image and video understanding. CNNs extract
hierarchical spatial features. Object detection models like YOLO predict bounding boxes
and classes in real time. Generative models like GANs and diffusion models synthesize
realistic images.

Retrieval-Augmented Generation (RAG) grounds language model outputs in a retrieved
knowledge base, reducing hallucinations and enabling up-to-date responses without
retraining. Documents are split into chunks, embedded, and stored in a vector index.
Queries retrieve the most relevant chunks, which are injected into the LLM prompt as context.
"""

pipeline.ingest(SAMPLE_DOC, "AI Overview (preloaded)")


@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.json or {}
    text = data.get("text", "").strip()
    name = data.get("filename", "document.txt")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 50:
        return jsonify({"error": "Text too short (min 50 chars)"}), 400
    if len(text) > 200_000:
        return jsonify({"error": "Text too long (max 200k chars)"}), 400
    
    result = pipeline.ingest(text, name)
    return jsonify({
        "success": True,
        "message": f"Ingested {result['chunks']} chunks from '{name}'",
        **result
    })


@app.route("/api/query", methods=["POST"])
def query():
    data  = request.json or {}
    question = data.get("question", "").strip()
    top_k    = int(data.get("top_k", 4))
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 chars)"}), 400
    
    result = pipeline.answer(question, top_k=min(top_k, 8))
    return jsonify(result)


@app.route("/api/stats")
def stats():
    s = pipeline.store.stats()
    return jsonify({
        **s,
        "history_turns": len(pipeline.history) // 2,
    })


@app.route("/api/reset", methods=["POST"])
def reset_history():
    pipeline.reset_history()
    return jsonify({"success": True, "message": "Conversation history cleared"})


@app.route("/api/documents", methods=["DELETE"])
def clear_documents():
    pipeline.store.clear()
    pipeline.reset_history()
    return jsonify({"success": True, "message": "All documents and history cleared"})


@app.route("/")
def index():
    return jsonify({
        "service":  "DocMind RAG API",
        "version":  "1.0.0",
        "endpoints": ["/api/ingest", "/api/query", "/api/stats", "/api/reset"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5051)