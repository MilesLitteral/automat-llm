import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer

from automat_llm.core import (
    load_personality_file,
    init_interactions,
    generate_response
)

import faiss

# ---------------------------
# Directories & Globals
# ---------------------------
if getattr(sys, 'frozen', False):
    current_dir = Path(sys.executable).parent
else:
    current_dir = Path(__file__).parent.resolve()

input_dir = current_dir / "Input_JSON"
logs_dir = current_dir / "Logs"

input_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

console = Console()
documents = []            # list of original JSON items
faiss_docs = []           # list of Document objects
embedding_index = None    # FAISS index
model = None              # SentenceTransformer model
embedding_dim = None
personality_data = None
user_interactions = None
rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]

# Logging
logging.basicConfig(
    filename=logs_dir / "chatbot_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Helper: render Markdown
# ---------------------------
def render_llm(text: str, title: str = "LLM Response"):
    md = Markdown(text, code_theme="monokai", hyperlinks=True)
    panel = Panel(md, title=title, border_style="cyan", padding=(1,2))
    console.print(panel, file=sys.stderr)

# ---------------------------
# Initialize FAISS system
# ---------------------------
def initialize_system_faiss():
    global faiss_docs, documents, embedding_index, embedding_dim, model, personality_data, user_interactions

    personality_data = load_personality_file()
    user_interactions = init_interactions()

    documents.clear()
    faiss_docs.clear()

    # Load JSON documents
    for file in input_dir.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                content = json.dumps(item)
                faiss_docs.append({"content": content, "source": file.name})
                documents.append(item)

    # Load embedding model
    model = SentenceTransformer("hkunlp/instructor-large")

    # Create embeddings
    doc_texts = [d["content"] for d in faiss_docs]
    embeddings = model.encode(doc_texts, convert_to_numpy=True).astype('float32')

    # FAISS index
    embedding_dim = embeddings.shape[1]
    embedding_index = faiss.IndexFlatL2(embedding_dim)
    embedding_index.add(embeddings)

    print(f"✅ Loaded {len(documents)} documents into FAISS.", flush=True)

# ---------------------------
# Retrieve top-k docs
# ---------------------------
def retrieve_top_k(query: str, k: int = 5):
    if embedding_index is None:
        return []

    query_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    D, I = embedding_index.search(query_emb, k)
    return [faiss_docs[i] for i in I[0]]

# ---------------------------
# Chat Loop
# ---------------------------
def chat_loop():
    global personality_data, user_interactions

    char_name = personality_data.get("char_name", "AI")
    print("READY", flush=True)  # Signal Electron that backend is ready

    while True:
        user_input = sys.stdin.readline()
        if not user_input:
            continue
        user_input = user_input.strip()
        if user_input.lower() == "quit":
            print("EXIT", flush=True)
            break

        # Optional image generation
        if "image" in user_input.lower():
            print("IMAGE_DONE", flush=True)
            continue

        # Retrieve relevant docs
        retrieved_docs = retrieve_top_k(user_input, k=10)
        context = "\n".join([d["content"] for d in retrieved_docs])

        # Generate response
        response = generate_response(
            user_id="Automat-User-Id",
            user_interactions=user_interactions,
            user_input=user_input,
            rude_keywords=rude_keywords,
            personality_data=personality_data,
            context=context  # raw text context from FAISS
        )

        print(response, flush=True)
        sys.stdout.flush()
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    initialize_system_faiss()
    chat_loop()
