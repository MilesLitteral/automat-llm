import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------------------
# Globals & Directories
# ---------------------------
CURRENT_DIR = Path(__file__).parent.resolve()
FAISS_DB_PATH = CURRENT_DIR / "faiss_db"
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

logs_dir = CURRENT_DIR / "Logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=logs_dir / "chatbot_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Conversation Utilities
# ---------------------------
def get_history(user_id: str, max_messages: int = 10):
    conversation_histories.setdefault(user_id, [])
    return conversation_histories[user_id][-max_messages:]


def add_to_history(user_id: str, role: str, content: str):
    conversation_histories.setdefault(user_id, [])
    conversation_histories[user_id].append({"role": role, "content": content})


def format_history(history):
    return "\n".join(
        f"User: {msg['content']}" if msg["role"] == "user"
        else f"Cybel: {msg['content']}"
        for msg in history
    )
current_dir = os.getcwd()

conversation_histories = {}  

def get_conversation_history(user_id, max_messages=10):
    """Get the last N messages from conversation history"""
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    return conversation_histories[user_id][-max_messages:]

def add_to_conversation_history(user_id, role, content):
    """Add a message to conversation history"""
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    conversation_histories[user_id].append({"role": role, "content": content})

def format_conversation_history(history):
    """Format conversation history as a readable string"""
    formatted = []
    for msg in history:
        if msg["role"] == "user":
            formatted.append(f"User: {msg['content']}")
        else:
            formatted.append(f"Cybel: {msg['content']}")
    return "\n".join(formatted)


def load_json_as_documents(client, directory):
    documents = []
    collection = client.collections.use("MyCollection") #TBA: use(f"{user_id}_Collection")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    raw_content = f.read()
                    # Optionally, reformat it as pretty-printed JSON
                    parsed = json.loads(raw_content)
                    pretty_json = json.dumps(parsed, indent=2)
                    documents.append(Document(page_content=pretty_json, metadata={"source": filename}))
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

    # Extract list of 'Entry' strings if the JSON is a list of dicts
    entries = [item['Entry'] for item in documents if 'Entry' in item]

    with collection.batch.fixed_size(batch_size=200) as batch:
        for d in entries:
            print(d)
            batch.add_object(
                {
                    "entry": d
                }
            )

            with open("uploaded_docs_log.json", "a", encoding="utf-8") as log_f:
                log_f.write(json.dumps({
                    "entry": d,
                    "timestamp": time.time()
                }) + "\n")

            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")


    return documents

def init_interactions():
    # Load or initialize user interactions
    try:
        user_interactions_file = f"{current_dir}/user_interactions.json"
        with open(user_interactions_file, 'r', encoding='utf-8') as f:
            user_interactions = json.load(f)
            return user_interactions
    except FileNotFoundError:
        user_interactions = {"users": {}}
        with open(user_interactions_file, 'w', encoding='utf-8') as f:
            json.dump(user_interactions, f, indent=4)


def load_personality_file():
    # Load the personality from robot_personality.json
    try:
        personality_file = f"{current_dir}/robot_personality.json"
        with open(personality_file, 'r', encoding='utf-8') as f:
            personality_data = json.load(f)
            return personality_data
    except FileNotFoundError:
        print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
        logging.error(f"Personality file not found at {personality_file}.")
        exit(1)


# ---------------------------
# Document Loader
# ---------------------------
def load_json_as_documents(directory: Path) -> List[Document]:
    documents = []

    for filename in directory.glob("*.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                parsed = json.load(f)

            if isinstance(parsed, list):
                for item in parsed:
                    if "Entry" in item:
                        documents.append(
                            Document(
                                page_content=item["Entry"],
                                metadata={"source": filename.name}
                            )
                        )
            else:
                documents.append(
                    Document(
                        page_content=json.dumps(parsed, indent=2),
                        metadata={"source": filename.name}
                    )
                )
        except Exception as e:
            print(f"Skipping {filename.name}: {e}")

    return documents

# ---------------------------
# Build Modern RAG
# ---------------------------
def create_faiss_rag(documents: List[Document]):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if FAISS_DB_PATH.exists():
        vectorstore = FAISS.load_local(
            str(FAISS_DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(FAISS_DB_PATH))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatGroq(
        temperature=0.5,
        model="openai/gpt-oss-20b",
        max_tokens=2048,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Cybel — a snarky but helpful AI assistant.
        Use retrieved memory context to answer accurately.
        Stay conversational and confident."""),
                ("human", """
        User input:
        {input}

        Conversation history:
        {conversation_history}

        Retrieved memory:
        {context}
        """)
    ])

    # Format retrieved docs into text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "conversation_history": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Modern LCEL FAISS RAG ready.")
    return rag_chain

# ---------------------------
# Response Generation
# ---------------------------
def generate_response(
    user_id: str,
    user_input: str,
    rude_keywords: list,
    personality_data: dict,
    rag_chain
) -> str:

    input_lower = user_input.lower()

    # Basic rudeness handling
    if any(keyword in input_lower for keyword in rude_keywords):
        return "Wow. Hostile. Try again with basic human decency."

    try:
        add_to_history(user_id, "user", user_input)
        history = get_history(user_id, max_messages=8)
        formatted_history = format_history(history[:-1])

        result = rag_chain.invoke({
            "input": user_input,
            "conversation_history": formatted_history
        })

        response = result
        #response = result["answer"]

        add_to_history(user_id, "assistant", response)

        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")

        return response

    except Exception as e:
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "Something broke internally. Give me a second."

# ---------------------------
# Example Main
# ---------------------------
if __name__ == "__main__":

    input_dir = CURRENT_DIR / "Input_JSON"
    documents = load_json_as_documents(input_dir)

    rag_chain = create_faiss_rag(documents)

    rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "quit":
            break

        response = generate_response(
            "user1",
            user_input,
            rude_keywords,
            {},
            rag_chain
        )

        print(f"Cybel: {response}")
